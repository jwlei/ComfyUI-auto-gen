"""
The panel in-process, starting the real main.py as a real process, which talks to
a fake ComfyUI and a fake Claude and runs real ffmpeg. Stop really kills it.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import close, fake_claude, fake_comfy, make_clip

from PIL import Image

import webapp
from modules import paths
from modules.run_state import RunState, list_runs


def png_bytes(color=(10, 200, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (640, 368), color).save(buffer, format="PNG")
    return buffer.getvalue()


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.comfy = fake_comfy(self.temp)
        self.addCleanup(close, self.comfy)
        self.claude = fake_claude()
        self.addCleanup(close, self.claude)
        patcher = mock.patch.dict(os.environ, {
            "COMFYUI_URL": self.comfy.url, "ANTHROPIC_BASE_URL": self.claude.url, "ANTHROPIC_API_KEY": "test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        webapp._comfyui_cache["at"] = 0.0
        webapp._runs_cache["at"] = 0.0
        # The fake renders instantly, which would otherwise teach the ETA that all renders do.
        if os.path.exists(webapp.STAGE_TIMINGS):
            os.remove(webapp.STAGE_TIMINGS)
        self.client = webapp.app.test_client()
        self.addCleanup(self.clear_everything)

    def clear_everything(self):
        webapp.stop_run()
        self.wait_idle()
        for queue in webapp.QUEUES.values():
            queue.items, queue.held, queue.results = [], False, []
            queue._save()
        for folder in (paths.RUNS_DIR, paths.OUTPUT_DIR, paths.UPLOADED_DIR, paths.INPUTS_DIR):
            shutil.rmtree(folder, ignore_errors=True)

    def wait_until(self, condition, what, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.05)
        self.fail(f"Timed out waiting for {what}. Log: {[l['text'] for l in webapp.job.lines][-15:]}")

    def wait_idle(self):
        def idle():
            webapp.run_queue.pending()
            return webapp.job.idle and not webapp.run_queue.current
        self.wait_until(idle, "the job to end")

    def generate(self, **fields):
        form = {"prompt": "a storm rolls in", "duration": "60", "llm": "claude", "size": "64x64",
                "image": (io.BytesIO(png_bytes()), "harbour.png"), **fields}
        return self.client.post("/generate", data={k: v for k, v in form.items() if v is not None},
                                content_type="multipart/form-data")

    def status(self):
        webapp._runs_cache["at"] = 0.0
        return self.client.get("/api/status").get_json()

    def only_run(self):
        runs = list_runs()
        self.assertEqual(len(runs), 1)
        return runs[0]

    def texts_rendered(self):
        return [p["prompt"]["121"]["inputs"]["text"] for p in self.comfy.prompts]

    # --- The whole way through ---

    def test_review_change_a_prompt_render_watch_download_delete(self):
        reply = self.generate(review="1")
        self.assertEqual(reply.get_json(), {"ok": True, "queued": False})
        self.wait_idle()

        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(self.comfy.prompts, [], "reviewing first renders nothing")
        self.assertEqual(status["progress"]["stage"][:2], [1, 1])
        self.assertIn("--plan-only", status["command"])
        self.assertEqual(self.claude.requests[0]["body"]["messages"][0]["content"][0]["type"], "image")

        run_id = status["progress"]["run_id"]
        self.assertEqual([run["id"] for run in status["stopped_runs"]], [run_id])
        self.assertIn("Plan ready", status["stopped_runs"][0]["detail"])
        self.assertGreater(status["stopped_runs"][0]["estimate"], 60 * 20, "it knows the render is still ahead")

        detail = self.client.get(f"/runs/{run_id}").get_json()
        self.assertEqual(detail["title"], "The Harbour Storm")
        self.assertEqual(detail["llm"], "claude")
        self.assertEqual(len(detail["shots"]), 8)
        self.assertFalse(detail["shots"][0]["rendered"])
        self.assertEqual(self.client.get(f"/runs/{run_id}/image").status_code, 200)
        self.assertEqual(self.client.get(f"/runs/{run_id}/clips/1").status_code, 404)

        prompts = [shot["prompt"] for shot in detail["shots"]]
        prompts[3] = "My better fourth prompt."
        self.assertTrue(self.client.post(f"/runs/{run_id}/shots", json={"prompts": prompts}).get_json()["ok"])
        self.assertEqual(self.client.post(f"/runs/{run_id}/shots", json={"prompts": ["too few"]}).status_code, 400)
        self.assertEqual(self.client.post(f"/runs/{run_id}/shots", json={"prompts": [""] * 8}).status_code, 400)

        self.assertTrue(self.client.post(f"/runs/{run_id}/continue").get_json()["ok"])
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(len(self.comfy.prompts), 8)
        self.assertEqual(self.texts_rendered()[3], "My better fourth prompt.")
        self.assertEqual(status["stopped_runs"], [], "a finished run isn't waiting for anything")
        self.assertEqual(status["progress"]["stage"][:2], [3, 3])

        film = status["videos"]["output"][0]
        self.assertEqual(film["title"], "The Harbour Storm")
        self.assertEqual(film["run"], run_id)
        video = self.client.get(f"/videos/output/{film['name']}")
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.mimetype, "video/mp4")
        video.close()
        self.assertEqual(self.client.get(f"/runs/{run_id}/clips/8").status_code, 200)

        # A shot redone from a finished film: the shots after it too, and the film anew.
        self.assertTrue(self.client.post(f"/runs/{run_id}/redo/7").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(self.status()["returncode"], 0)
        self.assertEqual(self.texts_rendered()[8:], ["Prompt number 7.", "Prompt number 8."])
        self.assertEqual(self.client.post(f"/runs/{run_id}/redo/99").status_code, 400)

        self.assertTrue(self.client.post(f"/delete/output/{film['name']}").get_json()["ok"])
        status = self.status()
        self.assertEqual(status["videos"]["output"], [])
        self.assertEqual(status["stopped_runs"], [], "deleting a film takes its clips with it")
        self.assertFalse(os.path.exists(os.path.join(paths.RUNS_DIR, run_id)))

    def test_stop_kills_it_cancels_the_clip_and_continue_picks_up(self):
        self.comfy.script = ["ok", "ok", "hang"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the third clip to start")
        self.wait_until(lambda: webapp.job.progress.get("detail", "").startswith("Clip 3"), "the panel to see it")

        status = self.status()
        self.assertTrue(status["running"])
        progress = status["progress"]
        self.assertEqual(progress["stage"][2], "Clips")
        self.assertRegex(progress["detail"], r"^Clip 3 of 8: Shot 3, \ds$")
        self.assertEqual([step["state"] for step in progress["eta"]["steps"]], ["done", "now", "next"])
        self.assertGreater(progress["eta"]["total_left"], 0)

        run_id = progress["run_id"]
        self.assertEqual(status["stopped_runs"], [], "a run that's going can't be continued")
        for refused in (self.client.post(f"/runs/{run_id}/continue"), self.client.post(f"/runs/{run_id}/discard"),
                        self.client.post(f"/runs/{run_id}/redo/1"),
                        self.client.post(f"/runs/{run_id}/shots", json={"prompts": ["x"] * 8})):
            self.assertEqual(refused.status_code, 409)
        self.assertTrue(self.client.get(f"/runs/{run_id}").get_json()["busy"])

        # A second film waits its turn, and Stop puts it on hold.
        self.assertEqual(self.generate(prompt="a second film").get_json(), {"ok": True, "queued": True})
        queue_status = self.status()["queue"]
        self.assertEqual(queue_status["items"][0]["label"], "a second film")
        self.assertGreater(queue_status["items"][0]["estimate"], 60)

        pid = webapp.job.process.pid
        self.assertTrue(self.client.post("/stop").get_json()["ok"])
        self.assertFalse(webapp.job.running)
        self.assertNotIn(str(pid), subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout)
        self.assertIn(("/interrupt", {"prompt_id": "p3"}), self.comfy.cancels)

        self.wait_until(lambda: webapp.job.idle, "the job to wind up")
        status = self.status()
        self.assertTrue(status["stopped"])
        self.assertTrue(status["queue"]["held"], "nothing starts by itself after a Stop")
        self.assertEqual(len(self.comfy.prompts), 3)
        waiting = status["stopped_runs"][0]
        self.assertIn("2 of 8 clips kept", waiting["detail"])
        self.assertEqual(waiting["clips"], 2)

        # Take the second film out, and continue the first.
        item = status["queue"]["items"][0]["id"]
        self.assertTrue(self.client.post(f"/queue/run/remove/{item}").get_json()["ok"])
        self.assertTrue(self.client.post(f"/runs/{run_id}/continue").get_json()["ok"])
        self.wait_until(lambda: webapp.job.progress.get("stage"), "the stage to show")
        self.wait_until(lambda: webapp.job.eta.number, "the ETA to see the stage")
        self.assertIn("Plan", webapp.job.eta.kept_stages)
        # Eight 8s shots fitted to 60s lose a second each from the front, so the two kept are 7s.
        self.assertEqual(webapp.job.eta.kept_seconds, 14)
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(self.texts_rendered()[3:], [f"Prompt number {n}." for n in range(3, 9)],
                         "the two finished clips aren't rendered again")
        self.assertEqual(len(status["videos"]["output"]), 1)

    def test_a_run_that_fails_can_be_continued(self):
        self.comfy.script = ["ok", "error", "error"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 1)
        self.assertFalse(status["stopped"])
        self.assertIn("1 of 8 clips kept", status["stopped_runs"][0]["detail"])
        self.assertTrue(any("Kept what's done" in line["text"] for line in status["lines"]))
        changes = webapp.job.eta.changes
        self.assertTrue(any("rendering again" in change["text"] for change in changes), changes)

    def test_a_queued_film_starts_by_itself_with_its_own_image(self):
        self.comfy.script = ["ok"] * 8
        self.assertFalse(self.generate().get_json()["queued"])
        self.assertTrue(self.generate(prompt="next film", review="1").get_json()["queued"])
        self.wait_until(lambda: len(self.claude.requests) == 2, "the second film to start")
        self.wait_idle()
        self.assertEqual(len(list_runs()), 2)
        self.assertEqual({run.brief for run in list_runs()}, {"a storm rolls in", "next film"})
        results = self.status()["queue"]["results"]
        self.assertEqual([result["outcome"] for result in results], ["done", "done"])

    def test_discard(self):
        self.assertTrue(self.generate(review="1").get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(self.client.post(f"/runs/{run.id}/discard").get_json()["ok"])
        self.assertEqual(list_runs(), [])
        self.assertEqual(self.client.post(f"/runs/{run.id}/discard").status_code, 404)

    def test_inputs_dont_pile_up(self):
        for _ in range(2):
            self.assertTrue(self.generate(review="1").get_json()["ok"])
            self.wait_idle()
        self.assertEqual(len(os.listdir(paths.INPUTS_DIR)), 1, "only the last one, until the next film")

    def test_failed_plan_says_why_and_leaves_nothing(self):
        self.claude.status = 401
        self.assertTrue(self.generate(review="1").get_json()["ok"])
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 1)
        self.assertTrue(any("ANTHROPIC_API_KEY" in line["text"] for line in status["lines"]), status["lines"])
        self.assertEqual(list_runs(), [])

    def test_check_setup(self):
        self.assertTrue(self.client.post("/check", data={"llm": "claude"}).get_json()["ok"])
        self.wait_until(lambda: webapp.job.idle, "the check to end")
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(self.comfy.prompts, [])
        self.assertEqual(self.claude.requests, [], "a check spends nothing")

    # --- Refusals ---

    def test_bad_forms_are_refused(self):
        for fields, expected in (
            ({"prompt": ""}, "what should happen"),
            ({"image": None}, "image"),
            ({"image": (io.BytesIO(b"not an image"), "x.png")}, "isn't an image"),
            ({"duration": "30"}, "60-300"),
            ({"duration": "soon"}, "whole number"),
            ({"start": "sideways"}, "start mode"),
            ({"llm": "gpt"}, "Unknown LLM"),
            ({"size": "big"}, "1280x736"),
            ({"upload": "1", "privacy": "shouted"}, "privacy"),
        ):
            reply = self.generate(**fields)
            self.assertEqual(reply.status_code, 400, fields)
            self.assertIn(expected, reply.get_json()["error"])
        self.assertEqual(self.claude.requests, [])

    def test_chosen_llm_without_a_key_is_refused(self):
        with mock.patch.object(webapp.planner, "llm_configured", return_value=False):
            reply = self.generate()
        self.assertEqual(reply.status_code, 400)
        self.assertIn("no credentials", reply.get_json()["error"])

    def test_an_oversized_image_says_so(self):
        reply = self.generate(image=(io.BytesIO(b"x" * (webapp.MAX_UPLOAD_BYTES + 1)), "huge.png"))
        self.assertEqual(reply.status_code, 413)
        self.assertIn("MB", reply.get_json()["error"])

    def test_nothing_outside_the_folders_can_be_reached(self):
        secret = os.path.join(paths.DATA_ROOT, "secret.mp4")
        with open(secret, "w") as f:
            f.write("x")
        self.addCleanup(os.remove, secret)
        for url in ("/videos/output/..%2Fsecret.mp4", "/download/output/..%2Fsecret.mp4", "/videos/nowhere/secret.mp4"):
            self.assertEqual(self.client.get(url).status_code, 404, url)
        self.assertEqual(self.client.post("/delete/output/..%2Fsecret.mp4").status_code, 404)
        self.assertTrue(os.path.exists(secret))
        for bad_id in ("..", "run_1", "..%2F..%2Fetc"):
            self.assertEqual(self.client.get(f"/runs/{bad_id}").status_code, 404, bad_id)

    def test_upload_needs_keys(self):
        self.assertEqual(self.client.post("/publish", data={"privacy": "private"}).status_code, 400)
        self.assertIn(b"disabled title", self.client.get("/").data)

    def test_upload_queue(self):
        os.makedirs(paths.OUTPUT_DIR, exist_ok=True)
        name = "Film_run_20261101_120000.mp4"
        make_clip(os.path.join(paths.OUTPUT_DIR, name), 1)
        fake_upload = [sys.executable, "-c", "print('uploading')"]
        keys = {"YOUTUBE_CLIENT_ID": "a", "YOUTUBE_CLIENT_SECRET": "b", "YOUTUBE_REFRESH_TOKEN": "c"}
        with mock.patch.dict(os.environ, keys):
            with mock.patch.object(webapp.uploads, "start", wraps=lambda argv, env=None, keep_log=False: webapp.Job.start(webapp.uploads, fake_upload)) as start:
                self.assertEqual(self.client.post("/publish", data={"video": name, "privacy": "unlisted"}).get_json(), {"ok": True, "queued": False})
                self.assertEqual(self.client.post("/publish", data={"video": name}).status_code, 409, "it's already claimed")
                self.wait_until(lambda: webapp.uploads.idle, "the upload to end")
                self.assertEqual(start.call_args.args[0][-3:], ["--privacy", "unlisted", name])
        webapp.upload_queue.pending()

    # --- Sign in and restarts ---

    def test_password(self):
        with mock.patch.dict(os.environ, {"APP_PASSWORD": "open sesame"}):
            client = webapp.app.test_client()
            self.assertEqual(client.get("/").status_code, 302)
            self.assertEqual(client.get("/api/status").status_code, 302)
            self.assertEqual(client.post("/generate").status_code, 302)
            self.assertIn(b"Wrong password", client.post("/login", data={"password": "nope"}).data)
            self.assertEqual(client.get("/").status_code, 302)
            reply = client.post("/login?next=//evil.example", data={"password": "open sesame"})
            self.assertEqual(reply.status_code, 302)
            self.assertNotIn("evil", reply.headers["Location"], "a link can't send you elsewhere after signing in")
            self.assertEqual(client.get("/").status_code, 200)
            self.assertIn(b"Sign out", client.get("/").data)

    def test_queue_comes_back_on_hold_after_a_restart(self):
        path = os.path.join(self.temp, "queue.json")
        item = {"id": 4, "label": "a film", "detail": "", "argv": ["x"], "env": {}, "estimate": 60}
        with open(path, "w", encoding="utf-8") as f:
            json.dump([item], f)
        queue = webapp.JobQueue(webapp.Job(), lambda item: None, path=path)
        self.assertTrue(queue.held)
        self.assertEqual(queue.snapshot()["items"][0]["label"], "a film")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{broken")
        self.assertEqual(webapp.JobQueue(webapp.Job(), lambda item: None, path=path).items, [])

    def test_shutdown_stops_everything_and_keeps_the_queue(self):
        self.comfy.script = ["hang"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the clip to start")
        self.assertTrue(self.generate(prompt="still queued").get_json()["queued"])
        run_id = webapp.job.progress["run_id"]

        def reopen():
            for job in (webapp.job, webapp.uploads):
                job.closed = False
        self.addCleanup(reopen)
        with mock.patch.object(webapp.threading, "Timer") as timer:
            self.assertTrue(self.client.post("/shutdown").get_json()["ok"])
        timer.assert_called_once()
        self.assertFalse(webapp.job.running)
        self.assertIn(("/interrupt", {"prompt_id": "p1"}), self.comfy.cancels)
        with open(webapp.run_queue.path, encoding="utf-8") as f:
            self.assertEqual([item["label"] for item in json.load(f)], ["still queued"])
        self.assertTrue(RunState.load(run_id).shots, "it can continue after the restart")

    # --- The page ---

    def test_page_script(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        page = os.path.join(self.temp, "page.html")
        with open(page, "wb") as f:
            f.write(self.client.get("/").data)
        result = subprocess.run([node, os.path.join(ROOT, "tests", "page_check.cjs"), page],
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stale_server_only_restarts_when_idle(self):
        with mock.patch.object(webapp.threading, "Timer") as timer:
            old = self.client.post("/launcher/refresh", data={"code_changed_at": "1"})
            self.assertEqual(old.get_json(), {"restarting": False})
            timer.assert_not_called()
            new = self.client.post("/launcher/refresh", data={"code_changed_at": str(time.time() + 60)})
            self.assertEqual(new.get_json(), {"restarting": True})
            timer.assert_called_once()


if __name__ == "__main__":
    unittest.main()

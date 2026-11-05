"""
The panel in-process, starting the real main.py as a real process, which talks to
a fake ComfyUI and a fake Claude and runs real ffmpeg. Stop really kills it.
"""
import io
import json
import os
import types
from datetime import datetime, timedelta
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

from fakes import (OBJECT_INFO, close, fake_claude, fake_comfy, make_clip, plan_reply, reference_workflow, saved_workflow,
                   text_only_workflow)

import ffmpeg
from PIL import Image

from panel import app as _both_pipelines  # noqa: F401
from panel import server as webapp
from panel import video as video_panel
from pipelines.shorts.paths import OUTPUT_DIR as SHORTS_READY, UPLOADED_DIR as SHORTS_UPLOADED
from pipelines.video import media, paths
from pipelines.video.run_state import RunState, list_runs


def make_clip_audio(path, seconds):
    ffmpeg.input(f"sine=f=440:d={seconds}", f="lavfi").output(path).run(overwrite_output=True, quiet=True)


def png_bytes(color=(10, 200, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (640, 368), color).save(buffer, format="PNG")
    return buffer.getvalue()


SAVED_OPTIONS = [{"id": "steps", "kind": "number", "label": "Steps", "default": "20", "min": 1, "max": 150, "step": 1, "advanced": True},
                 {"id": "negative", "kind": "text", "label": "Negative prompt", "default": "saved negative", "advanced": True}]

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
        webapp._saved_workflows_cache["at"] = 0.0
        # The fake renders instantly, which would otherwise teach the ETA that all renders do.
        if os.path.exists(webapp.VIDEO_TIMINGS):
            os.remove(webapp.VIDEO_TIMINGS)
        self.client = webapp.app.test_client()
        self.addCleanup(self.clear_everything)

    def clear_everything(self):
        webapp.stop_run()
        self.wait_idle()
        for queue in webapp.QUEUES.values():
            queue.items, queue.held, queue.results = [], False, []
            queue._save()
        for folder in (paths.RUNS_DIR, paths.OUTPUT_DIR, paths.INPUTS_DIR):
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
        return self.client.post("/generate/video", data={k: v for k, v in form.items() if v is not None},
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
        # Eight clips to even out for stitching take 48 seconds, and this tiny test film's render comes on top.
        self.assertGreater(status["stopped_runs"][0]["estimate"], 48, "it knows the render is still ahead")

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

        film = status["videos"]["video"][0]
        self.assertEqual(film["title"], "The Harbour Storm")
        self.assertEqual(film["run"], run_id)
        video = self.client.get(f"/videos/video/{film['name']}")
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

        self.assertTrue(self.client.post(f"/delete/video/{film['name']}").get_json()["ok"])
        status = self.status()
        self.assertEqual(status["videos"]["video"], [])
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
        self.assertEqual(queue_status["items"][0]["label"], "1 video", "titled like every side's")
        self.assertEqual(queue_status["items"][0]["facts"]["prompt"], "a second film", "the prompt is what the title opens")
        # The first film's plan and clips were timed: the fakes are quick, so it is short, but it's there.
        self.assertGreater(queue_status["items"][0]["estimate"], 0)

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
        self.assertEqual(len(status["videos"]["video"]), 1)

    # --- Generate never pays for the same plan twice ---

    def test_generate_continues_the_same_film_without_a_new_plan(self):
        self.assertTrue(self.generate(review="1").get_json()["ok"])
        self.wait_idle()
        run_id = self.only_run().id
        asked = len(self.claude.requests)

        # Asked to read the prompts again: they're there, and nothing starts.
        reply = self.generate(review="1").get_json()
        self.assertEqual(reply["shots"], run_id)
        self.assertEqual(reply["continued"], self.only_run().title)
        self.assertEqual(len(self.claude.requests), asked)

        # Generate, with a stopped run on the same brief and image: that run, and no new plan.
        reply = self.generate().get_json()
        self.assertEqual(reply["continued"], self.only_run().title)
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(len(self.claude.requests), asked, "the plan was already paid for")
        self.assertEqual(len(status["videos"]["video"]), 1)

    def test_generate_plans_anew_when_the_film_differs(self):
        self.assertTrue(self.generate(review="1").get_json()["ok"])
        self.wait_idle()
        asked = len(self.claude.requests)
        for change in ({"prompt": "something else"}, {"duration": "90"},
                       {"image": (io.BytesIO(png_bytes((200, 10, 10))), "other.png")}):
            self.assertNotIn("continued", self.generate(review="1", **change).get_json(), change)
            self.wait_idle()
        self.assertGreaterEqual(len(self.claude.requests), asked + 3, "each one needed its own plan")

    def test_generate_keeps_the_plan_when_the_size_changes(self):
        self.assertTrue(self.generate(review="1").get_json()["ok"])
        self.wait_idle()
        asked = len(self.claude.requests)
        reply = self.generate(size="96x96").get_json()
        self.assertIn("continued", reply)
        self.wait_idle()
        self.assertEqual(len(self.claude.requests), asked)
        self.assertEqual(self.status()["returncode"], 0)

    def audio_file(self, seconds=20):
        path = os.path.join(self.temp, "song.wav")
        make_clip_audio(path, seconds)
        with open(path, "rb") as f:
            return io.BytesIO(f.read()), "song.wav"

    def test_a_film_with_audio_is_as_long_as_it(self):
        self.claude.reply = plan_reply(shots=3, seconds=7)
        reply = self.generate(duration="60", audio_mode="file", audio_file=self.audio_file()).get_json()
        self.assertTrue(reply["ok"], reply)
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        film = os.path.join(paths.OUTPUT_DIR, status["videos"]["video"][0]["name"])
        self.assertAlmostEqual(media.duration(film), 20, delta=1.5)
        streams = ffmpeg.probe(film)["streams"]
        self.assertEqual(sorted(stream["codec_type"] for stream in streams), ["audio", "video"])

    def test_the_same_audio_keeps_the_plan_and_other_audio_does_not(self):
        self.claude.reply = plan_reply(shots=3, seconds=7)
        self.assertTrue(self.generate(review="1", audio_mode="file", audio_file=self.audio_file()).get_json()["ok"])
        self.wait_idle()
        asked = len(self.claude.requests)
        self.assertIn("continued", self.generate(review="1", audio_mode="file", audio_file=self.audio_file()).get_json())
        self.assertEqual(len(self.claude.requests), asked)
        self.assertNotIn("continued", self.generate(review="1", audio_mode="file", audio_file=self.audio_file(21)).get_json())
        self.wait_idle()
        self.assertEqual(len(self.claude.requests), asked + 1)

    def test_audio_options_are_checked(self):
        self.assertIn("Write the text", self.generate(audio_mode="text").get_json()["error"])
        self.assertIn("Choose the audio", self.generate(audio_mode="file").get_json()["error"])
        self.assertIn("Unknown voice", self.generate(audio_mode="text", narration="Hello", voice="nobody").get_json()["error"])

    def test_a_film_of_one_clip(self):
        self.claude.reply = plan_reply(shots=1, seconds=5)
        self.assertTrue(self.generate(duration="5").get_json()["ok"])
        self.wait_idle()
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(len(self.comfy.prompts), 1)
        self.assertEqual(self.comfy.prompts[0]["prompt"]["291"]["inputs"]["value"], 5)
        self.assertEqual(len(status["videos"]["video"]), 1)
        film = os.path.join(paths.OUTPUT_DIR, status["videos"]["video"][0]["name"])
        self.assertAlmostEqual(media.duration(film), 5, delta=0.1, msg="a first clip keeps all its frames")

    def test_the_estimate_covers_the_whole_range(self):
        short = self.client.post("/api/estimate/video", data={"duration": "5"}).get_json()
        long = self.client.post("/api/estimate/video", data={"duration": "600"}).get_json()
        self.assertEqual((short["clips"], long["clips"]), (1, 75))
        self.assertGreater(long["total"], 5 * 3600 * 0.9, "about 5 hours at the default render speed")
        self.assertLess(short["total"], 600)

    def test_the_estimate_follows_the_resolution(self):
        clips = lambda **size: self.client.post("/api/estimate/video", data={"duration": "60", **size}).get_json()["steps"][1]["seconds"]
        highest = clips()
        self.assertEqual(clips(size="1280x736"), highest, "the default is the highest preset")
        self.assertEqual(clips(size="736x1280"), highest, "turning it doesn't change the pixels")
        self.assertAlmostEqual(clips(size="640x352") / highest, 640 * 352 / (1280 * 736), places=2)

    def test_a_preset_is_what_gets_rendered(self):
        self.comfy.script = []
        self.claude.reply = plan_reply(shots=1, seconds=5)
        self.assertTrue(self.generate(duration="5", size="352x640").get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertEqual((run.width, run.height), (352, 640))
        sent = self.comfy.prompts[0]["prompt"]
        self.assertEqual((sent["292"]["inputs"]["value"], sent["293"]["inputs"]["value"]), (352, 640))

    def test_a_short_film_says_so_in_the_queue(self):
        self.comfy.script = ["hang"]
        self.claude.reply = plan_reply(shots=1, seconds=5)
        self.assertTrue(self.generate(duration="5").get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the clip to start")
        self.assertTrue(self.generate(duration="150", review="1").get_json()["queued"])
        self.assertTrue(self.generate(duration="5", review="1").get_json()["queued"])
        details = [item["detail"] for item in self.status()["queue"]["items"]]
        self.assertTrue(details[0].startswith("2:30 film"), details)
        self.assertTrue(details[1].startswith("5s film"), details)

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

        # Stopped runs says it failed and why, until it makes its film; Output only lists what was made.
        self.wait_until(lambda: self.status()["stopped_runs"][0].get("outcome"), "the failure to be noted")
        failed = self.status()["stopped_runs"][0]
        self.assertEqual(self.status()["videos"]["video"], [])
        self.assertEqual(failed["outcome"]["outcome"], "failed")
        self.assertIn("failed", failed["outcome"]["why"])
        # The fake ComfyUI fails with "out of memory": said in plain words first, its lines under them, and in Finished.
        self.assertTrue(failed["outcome"]["why"].startswith("Out of GPU memory"), failed["outcome"]["why"])
        self.assertIn("From its log:", failed["outcome"]["why"])
        self.assertTrue(self.status()["queue"]["results"][-1]["note"].startswith("Out of GPU memory"))
        self.comfy.script = ["ok"] * 7
        self.assertTrue(self.client.post(f"/runs/{failed['id']}/continue").get_json()["ok"])
        self.wait_until(lambda: self.status()["videos"]["video"], "the film")
        self.wait_until(lambda: not os.path.exists(os.path.join(RunState.load(failed["id"]).folder, webapp.OUTCOME_FILE)),
                        "the note to go once it worked")

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
        # Asked to stop at its prompts: listed as not rendered, saying so, rather than missing.
        self.wait_until(lambda: any(run.get("outcome") for run in self.status()["stopped_runs"]), "it to be noted")
        self.assertEqual([run["outcome"]["outcome"] for run in self.status()["stopped_runs"] if run.get("outcome")], ["not rendered"])

    def test_skip_starts_the_next_film_and_continues_later(self):
        self.comfy.script = ["ok", "hang"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the second clip to start")
        self.wait_until(lambda: webapp.job.progress.get("detail", "").startswith("Clip 2"), "the panel to see it")
        skipped = webapp.job.progress["run_id"]
        self.assertTrue(self.generate(prompt="next film", review="1").get_json()["queued"])

        self.assertEqual(self.client.post("/skip").get_json(), {"ok": True, "error": None})
        self.assertIn(("/interrupt", {"prompt_id": "p2"}), self.comfy.cancels)
        self.wait_until(lambda: len(self.claude.requests) == 2, "the next film to start by itself")
        self.wait_idle()
        status = self.status()
        self.assertFalse(status["queue"]["held"])
        self.assertEqual([result["outcome"] for result in status["queue"]["results"]], ["stopped", "done"])
        waiting = next(run for run in status["stopped_runs"] if run["id"] == skipped)
        self.assertIn("1 of 8 clips kept", waiting["detail"])

        self.assertTrue(self.client.post(f"/runs/{skipped}/continue").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(self.status()["returncode"], 0)
        self.assertEqual(len(self.comfy.prompts), 9, "only the cut-off clip and those after it were rendered")

    def test_skip_refuses_a_run_with_nothing_saved_yet(self):
        self.comfy.script = ["hang"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the clip to start")
        self.wait_until(lambda: webapp.job.progress.get("run_id"), "the panel to see the run")
        self.assertTrue(self.generate(prompt="next film", review="1").get_json()["queued"])
        with mock.patch.object(webapp, "saved_runs", return_value=[]):
            reply = self.client.post("/skip")
        self.assertEqual(reply.status_code, 409)
        self.assertTrue(webapp.job.running, "stopping it now would lose it")
        self.assertFalse(self.status()["queue"]["held"])

    def test_presets_are_kept_by_name_per_side(self):
        if os.path.exists(webapp.PRESETS_PATH):
            os.remove(webapp.PRESETS_PATH)
        self.addCleanup(lambda: os.path.exists(webapp.PRESETS_PATH) and os.remove(webapp.PRESETS_PATH))
        post = lambda body: self.client.post("/api/presets", json=body)
        self.assertEqual(post({"side": "image", "name": "Phone wallpaper", "values": {"aspect": "1272x2772", "count": "4"}}).status_code, 200)
        post({"side": "image", "name": "Edits", "values": {"workflow": "image_edit.json"}})
        presets = self.client.get("/api/presets").get_json()["presets"]
        self.assertEqual([p["name"] for p in presets["image"]], ["Edits", "Phone wallpaper"], "by name")
        self.assertEqual(presets["video"], [])
        post({"side": "image", "name": "Edits", "values": {"workflow": "other.json"}})
        presets = self.client.get("/api/presets").get_json()["presets"]
        self.assertEqual(next(p for p in presets["image"] if p["name"] == "Edits")["values"], {"workflow": "other.json"}, "the same name saves over it")
        post({"side": "image", "name": "Edits", "delete": True})
        self.assertEqual([p["name"] for p in self.client.get("/api/presets").get_json()["presets"]["image"]], ["Phone wallpaper"])
        self.assertEqual(post({"side": "nowhere", "name": "x", "values": {}}).status_code, 400)

    def test_a_runs_title_opens_its_workflow_and_resolution(self):
        image = {"pipeline": "image", "argv": ["--count", "3", "--size", "1272x2772"],
                 "env": {"COMFYUI_WORKFLOW": "C:/x/logs/workflows/image_krea2_SFW_NSFW.json"}}
        self.assertEqual(webapp.run_facts(image), {"workflow": "image_krea2_SFW_NSFW", "resolution": "1272x2772"})
        by_aspect = {"pipeline": "image", "argv": ["--aspect", "16:9", "--megapixels", "2"], "env": {}}
        self.assertEqual(webapp.run_facts(by_aspect), {"workflow": "Built-in", "resolution": "16:9 at 2 MP"})
        shorts = {"pipeline": "shorts", "argv": ["--runs", "3"], "env": {"AVATAR_SIZE": "576x1024", "COMFYUI_WORKFLOW": "w/avatar_lipsync.json"}}
        self.assertEqual(webapp.run_facts(shorts), {"workflow": "avatar_lipsync", "resolution": "576x1024"})
        self.assertEqual(webapp.run_facts({"pipeline": "video", "argv": ["--resume", "r1"], "env": {}})["resolution"], "As the run was set up")

    def test_a_side_has_one_default_preset_at_most(self):
        if os.path.exists(webapp.PRESETS_PATH):
            os.remove(webapp.PRESETS_PATH)
        self.addCleanup(lambda: os.path.exists(webapp.PRESETS_PATH) and os.remove(webapp.PRESETS_PATH))
        post = lambda body: self.client.post("/api/presets", json=body)
        defaults = lambda: [p["name"] for p in self.client.get("/api/presets").get_json()["presets"]["image"] if p.get("default")]
        post({"side": "image", "name": "Edits", "values": {"workflow": "image_edit.json"}})
        post({"side": "image", "name": "Wallpaper", "values": {"aspect": "16:9"}})
        post({"side": "image", "name": "Edits", "default": True})
        self.assertEqual(defaults(), ["Edits"])
        post({"side": "image", "name": "Wallpaper", "default": True})
        self.assertEqual(defaults(), ["Wallpaper"], "making one the default unmakes the last")
        post({"side": "image", "name": "Wallpaper", "values": {"aspect": "21:9"}})
        self.assertEqual(defaults(), ["Wallpaper"], "saved over, it stays the default")
        post({"side": "image", "name": "Wallpaper", "default": False})
        self.assertEqual(defaults(), [])
        self.assertEqual(post({"side": "image", "name": "Nothing", "default": True}).status_code, 404)

    def test_a_job_started_later_waits_at_the_end_until_it_is_started(self):
        self.assertTrue(self.generate(start_later="1").get_json()["queued"], "it waits, though nothing is running")
        waiting = self.status()["queue"]["items"]
        self.assertEqual([item["later"] for item in waiting], [True])
        self.assertFalse(webapp.job.running)
        self.comfy.script = ["ok"] * 16
        self.assertTrue(self.generate(prompt="busy", review="1").get_json() is not None)
        self.wait_idle()
        self.assertEqual([item["label"] for item in self.status()["queue"]["items"]], [waiting[0]["label"]], "the one added after went first")
        # Held again by something running, a new one goes ahead of it, and it stays last.
        self.assertTrue(self.client.post(f"/queue/run/start/{waiting[0]['id']}").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(self.status()["queue"]["items"], [], "started, it ran")

    def test_your_phone_is_told_when_a_job_ends_if_ntfy_is_set(self):
        sent = []
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "my-topic"}), \
                mock.patch.object(webapp.requests, "post", side_effect=lambda url, **kw: sent.append((url, kw))):
            webapp.notify_ended({"label": "A storm", "pipeline": "video"}, "done")
            webapp.notify_ended({"label": "Stopped", "pipeline": "video"}, "stopped")
            self.wait_until(lambda: sent, "the message")
        self.assertEqual(len(sent), 1, "only finished or failed")
        url, kw = sent[0]
        self.assertEqual(url, "https://ntfy.sh/my-topic")
        self.assertEqual(kw["headers"]["Title"].decode(), "Finished: A storm")
        sent.clear()
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": ""}), mock.patch.object(webapp.requests, "post", side_effect=lambda *a, **k: sent.append(1)):
            webapp.notify_ended({"label": "x"}, "done")
        self.assertEqual(sent, [], "nothing without a topic")

    def test_a_failure_we_know_is_said_in_plain_words(self):
        job = types.SimpleNamespace(first_line=0, lines=[])
        say = lambda *texts: (job.lines.clear(), job.lines.extend({"n": n, "text": t} for n, t in enumerate(texts, 1)), webapp.explain_failure(job, 1))[2]
        self.assertTrue(say("🎬 Clip 1/3", "❌ ComfyUI reported an error: torch.OutOfMemoryError: CUDA out of memory.")[0].startswith("Out of GPU memory"))
        self.assertTrue(say("⚠️ Gemini: 429 RESOURCE_EXHAUSTED")[0].startswith("The LLM's quota is used up"))
        self.assertTrue(say("❌ ComfyUI isn't running at http://127.0.0.1:8188.")[0].startswith("ComfyUI wasn't running"))
        self.assertTrue(say("Needs the node LayerColor: Brightness Contrast, which isn't installed.")[0].startswith("A node this workflow needs"))
        self.assertIsNone(say("⚠️ Could not write the metadata (no quota). Using the film's title.")[0], "a warning about metadata isn't the LLM's quota")
        plain, said = say("❌ Something odd happened.")
        self.assertIsNone(plain, "one we don't know is left as it said")
        self.assertIn("Something odd happened", said)
        # Its own lines stay under the plain words.
        self.assertIn("CUDA out of memory", say("❌ CUDA out of memory.")[1])

    def test_an_exact_prompt_goes_to_every_clip_without_an_llm(self):
        self.comfy.script = ["ok"] * 16
        self.assertTrue(self.generate(prompt_mode="direct", prompt="A red kite over a grey sea", duration="20").get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertEqual(self.claude.requests, [], "no LLM is asked")
        self.assertTrue(run.shots and all(shot["prompt"] == "A red kite over a grey sea" for shot in run.shots))
        self.assertEqual(sum(shot["seconds"] for shot in run.shots), 20)

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

    def test_bad_forms_are_refused(self):
        for fields, expected in (
            ({"prompt": ""}, "what should happen"),
            ({"image": None}, "image"),
            ({"image": (io.BytesIO(b"not an image"), "x.png")}, "isn't an image"),
            ({"duration": "4"}, "5-600"),
            ({"duration": "605"}, "5-600"),
            ({"duration": "soon"}, "whole number"),
            ({"start": "sideways"}, "start mode"),
            ({"llm": "gpt"}, "Unknown LLM"),
            ({"size": "big"}, "1280x736"),
        ):
            reply = self.generate(**fields)
            self.assertEqual(reply.status_code, 400, fields)
            self.assertIn(expected, reply.get_json()["error"])
        self.assertEqual(self.claude.requests, [])

    def test_chosen_llm_without_a_key_is_refused(self):
        with mock.patch.object(video_panel.planner, "llm_configured", return_value=False):
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
        for url in ("/videos/output/..%2Fsecret.mp4", "/download/video/..%2Fsecret.mp4", "/videos/nowhere/secret.mp4"):
            self.assertEqual(self.client.get(url).status_code, 404, url)
        self.assertEqual(self.client.post("/delete/video/..%2Fsecret.mp4").status_code, 404)
        self.assertTrue(os.path.exists(secret))
        for bad_id in ("..", "run_1", "..%2F..%2Fetc"):
            self.assertEqual(self.client.get(f"/runs/{bad_id}").status_code, 404, bad_id)

    # --- Workflows saved in ComfyUI ---

    def save_workflows(self):
        broken = saved_workflow()
        next(n for n in broken["nodes"] if n["id"] == 1)["widgets_values"] = ["gone.safetensors"]
        self.comfy.saved, self.comfy.object_info = {"wan.json": saved_workflow(), "broken.json": broken,
                                                    "r2v.json": reference_workflow(), "t2v.json": text_only_workflow()}, OBJECT_INFO

    def test_saved_workflows_only_list_what_runs_and_say_why_elsewhere(self):
        self.save_workflows()
        reply = self.client.get("/api/workflows?pipeline=video").get_json()
        self.assertTrue(reply["online"])
        with open(os.path.join(webapp.SAVED_WORKFLOWS_DIR, "not_listed.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"broken": "Needs gone.safetensors, which isn't installed."})
        self.assertEqual(self.client.get("/api/workflows?pipeline=shorts").get_json()["workflows"], [], "none are tuned for Shorts")
        self.assertEqual(reply["workflows"], [
            {"name": "r2v.json", "label": "r2v", "takes": {"min_images": 0, "max_images": 20}, "clip_seconds": [5, 10, 8]},
            {"name": "t2v.json", "label": "t2v", "takes": {"min_images": 0, "max_images": 0, "options": SAVED_OPTIONS}, "clip_seconds": [5, 10, 8]},
            {"name": "wan.json", "label": "wan", "takes": {"min_images": 0, "max_images": 1, "options": SAVED_OPTIONS}, "clip_seconds": [5, 10, 8]}],
            "each with the workflow's own options: a KSampler's steps here")

    def test_a_films_workflow_options_are_used_and_kept(self):
        self.save_workflows()
        self.assertTrue(self.generate(workflow="wan.json", opt_steps="12").get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(run.options, {"steps": "12"})
        self.assertTrue(all(p["prompt"]["6"]["inputs"]["steps"] == 12 for p in self.comfy.prompts), "every clip")
        clips = len(self.comfy.prompts)
        film = os.path.basename(run.final)
        self.assertTrue(film.endswith("_wan.mp4") and run.id not in film, "<title>_<workflow>.mp4")
        listed = next(v for v in webapp.list_videos("video") if v["name"] == film)
        # The same again: what it was made with, and the image it was given.
        settings = self.client.get(f"/runs/{run.id}").get_json()["settings"]
        self.assertEqual((settings["workflow"], settings["options"], settings["images"]), ("wan.json", {"steps": "12"}, 1))
        self.assertEqual(self.client.get(f"/runs/{run.id}/inputs/image/1").status_code, 200)
        self.assertEqual(self.client.get(f"/runs/{run.id}/inputs/image/2").status_code, 404)
        self.assertEqual(self.client.get(f"/runs/{run.id}/inputs/secret/1").status_code, 404)
        self.assertEqual((listed["run"], listed["model"]), (run.id, "wan"), "the film still finds its run, and shows its workflow")
        self.assertTrue(self.client.post(f"/runs/{run.id}/redo/1").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(os.path.basename(self.only_run().final), film, "made again, it keeps its name")
        self.assertEqual(self.comfy.prompts[clips]["prompt"]["6"]["inputs"]["steps"], 12, "a shot made again is made the same way")
        self.assertEqual(self.generate(workflow="wan.json", opt_steps="0").status_code, 400)

    def test_a_run_renders_with_the_chosen_workflow_and_keeps_it(self):
        self.save_workflows()
        self.assertEqual(self.generate(workflow="broken.json").status_code, 400)
        self.assertEqual(self.generate(workflow="nowhere.json").status_code, 400)
        self.assertTrue(self.generate(workflow="wan.json").get_json()["ok"])
        self.wait_idle()

        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(os.path.basename(run.workflow), "wan.json")
        sent = self.comfy.prompts[0]["prompt"]
        self.assertEqual(sent["2"]["inputs"]["text"], "Prompt number 1.")
        self.assertEqual(sent["3"]["inputs"]["text"], "saved negative")
        self.assertEqual(sent["4"]["inputs"]["image"], "uploaded.png")
        seconds = run.shots[0]["seconds"]
        self.assertEqual(sent["5"]["inputs"]["length"], seconds * 24 + 1, "in frames, at the workflow's 24 fps")
        self.assertEqual((sent["5"]["inputs"]["width"], sent["5"]["inputs"]["height"]), (64, 64))
        self.assertIs(sent["11"]["inputs"]["save_output"], False)

        rendered = len(self.comfy.prompts)
        self.assertTrue(self.client.post(f"/runs/{run.id}/redo/8").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(len(self.comfy.prompts), rendered + 1)
        self.assertIn("5", self.comfy.prompts[-1]["prompt"], "the redo uses it too")

    def test_without_an_image_the_input_comes_out_of_the_workflow(self):
        self.save_workflows()
        self.assertTrue(self.generate(workflow="wan.json", image=None).get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(run.image_count, 0)
        first, second = (p["prompt"] for p in self.comfy.prompts[:2])
        self.assertNotIn("start_image", first["5"]["inputs"], "text only, so it doesn't load the saved example")
        self.assertEqual(second["5"]["inputs"]["start_image"], ["4", 0], "after that, it continues from the last frame")
        self.assertEqual(second["4"]["inputs"]["image"], "uploaded.png")
        self.assertEqual(len(self.comfy.uploads), len(self.comfy.prompts) - 1)

    def test_a_text_only_workflow_takes_no_image(self):
        self.save_workflows()
        self.assertEqual(self.generate(workflow="t2v.json").status_code, 400, "it has nowhere to put an image")
        self.assertTrue(self.generate(workflow="t2v.json", image=None).get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(run.start, "image", "nothing to start the next clip on")
        self.assertEqual(self.comfy.uploads, [])

    def test_a_reference_workflow_gets_every_clip_all_its_images(self):
        self.save_workflows()
        image = lambda: (io.BytesIO(png_bytes()), "ref.png")
        too_many = {f"image_{number}": image() for number in range(2, 22)}
        self.assertEqual(self.generate(workflow="r2v.json", **too_many).status_code, 400, "twenty references at most")
        self.assertTrue(self.generate(workflow="r2v.json", image_2=image()).get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual((run.image_count, run.start), (2, "image"))
        for sent in (p["prompt"] for p in self.comfy.prompts):
            self.assertEqual((sent["2"]["inputs"]["image"], sent["1"]["inputs"]["image"]), ("uploaded.png", "uploaded.png"))
            self.assertEqual(sent["6"]["inputs"]["width"], 64, "the size picker's link is replaced")
            self.assertEqual(sent["3"]["inputs"]["value"][:6], "Prompt")
            added = [node_id for node_id, node in sent.items() if "added by the pipeline" in node["_meta"]["title"]]
            self.assertEqual(added, [], "the references left empty come out, with their links")
            self.assertEqual(sorted(name for name in sent["6"]["inputs"] if name.startswith("ref_images")),
                             ["ref_images.ref_image_0", "ref_images.ref_image_1"])
        self.assertTrue(os.path.exists(run.image_paths[1]))
        guide = self.claude.requests[0]["body"]["messages"][0]["content"][-1]["text"]
        self.assertIn("<Picture 2> adds to it", guide, "extra context, by default")
        self.assertNotIn("for you alone", guide, "the video model sees this one")

    def test_the_built_in_workflow_still_needs_its_image(self):
        self.assertEqual(self.generate(image=None).status_code, 400)
        image = lambda: (io.BytesIO(png_bytes()), "b.png")
        self.assertEqual(self.generate(image_2=image(), image_3=image()).status_code, 400)

    def test_image_2_needs_an_image_1_to_add_to(self):
        self.save_workflows()
        image = lambda: (io.BytesIO(png_bytes()), "b.png")
        for workflow in ("", "wan.json", "r2v.json"):
            reply = self.generate(workflow=workflow, image=None, image_2=image())
            self.assertEqual(reply.status_code, 400, workflow)
            self.assertIn("in order", reply.get_json()["error"])
        self.assertEqual(self.generate(workflow="t2v.json", image_2=image()).status_code, 400)
        self.assertEqual(self.claude.requests, [])

    def test_an_unknown_role_for_image_2_is_refused(self):
        reply = self.generate(image_2_role="villain")
        self.assertEqual(reply.status_code, 400)
        self.assertIn("image 2", reply.get_json()["error"])

    def test_with_one_input_image_2_reaches_the_clips_through_the_prompts(self):
        first, second = png_bytes((200, 10, 30)), png_bytes((10, 30, 200))
        self.assertTrue(self.generate(image=(io.BytesIO(first), "a.png"), image_2=(io.BytesIO(second), "b.png")).get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.final, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual((run.image_count, run.start), (1, "chain"), "still a continuous take from image 1")
        self.assertEqual(len(run.image_paths), 1)

        content = self.claude.requests[0]["body"]["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["image", "image", "text"], "the LLM sees both")
        self.assertIn("for you alone", content[-1]["text"])
        self.assertIn("It shows a place", content[-1]["text"], "extra context, by default")
        # Every clip after the first starts on the last frame, and none of them starts on image 2.
        with Image.open(run.image_path) as kept:
            self.assertEqual(kept.getpixel((0, 0)), (200, 10, 30), "the run keeps image 1 as its first frame")
        with open(run.image_path, "rb") as f:
            self.assertTrue(self.comfy.uploads[0].startswith(f.read()))
        self.assertEqual(len(self.comfy.uploads), len(self.comfy.prompts))
        self.assertTrue(any("only reaches them through" in l["text"] for l in webapp.job.lines))

    def test_image_2_can_be_another_character(self):
        self.assertTrue(self.generate(image_2=(io.BytesIO(png_bytes()), "b.png"), image_2_role="character").get_json()["ok"])
        self.wait_idle()
        guide = self.claude.requests[0]["body"]["messages"][0]["content"][-1]["text"]
        self.assertIn("another character", guide)
        self.assertIn("describe how they look in full", guide)
        self.assertNotIn("It shows a place", guide)

    def test_each_image_sends_its_role(self):
        flags, error = video_panel.build_options({"image_2_role": "view", "image_3_role": "place", "image_5_role": "character"})
        self.assertIsNone(error)
        self.assertEqual([flags[i + 1] for i, flag in enumerate(flags) if flag == "--image-role"], ["view", "place"],
                         "in order, up to the first box without one")
        self.assertIn("image 3", video_panel.build_options({"image_2_role": "view", "image_3_role": "villain"})[1])

    def test_saved_workflows_stay_listed_with_comfyui_off(self):
        self.save_workflows()
        self.client.get("/api/workflows")
        webapp._comfyui_cache.update(at=0.0)
        with mock.patch.dict(os.environ, {"COMFYUI_URL": "http://127.0.0.1:9"}):
            reply = self.client.get("/api/workflows").get_json()
            self.assertFalse(reply["online"])
            self.assertEqual(reply["workflows"], [
                {"name": "r2v.json", "label": "r2v", "takes": {"min_images": 0, "max_images": 20}, "clip_seconds": [5, 10, 8]},
                {"name": "t2v.json", "label": "t2v", "takes": {"min_images": 0, "max_images": 0, "options": SAVED_OPTIONS}, "clip_seconds": [5, 10, 8]},
                {"name": "wan.json", "label": "wan", "takes": {"min_images": 0, "max_images": 1, "options": SAVED_OPTIONS}, "clip_seconds": [5, 10, 8]}])
            self.assertEqual(os.path.basename(video_panel.chosen_workflow({"workflow": "wan.json"})[0]["COMFYUI_WORKFLOW"]), "wan.json")
        webapp._comfyui_cache.update(at=0.0)

    def test_the_log_says_what_each_line_is(self):
        self.assertEqual([webapp.line_kind(text) for text in ("▶ Stage 1/3: Plan", "❌ failed", "⚠️ careful", "✅ Done", "🎛️ ComfyUI · Sampling · 4s", "plain")],
                         ["stage", "err", "warn", "ok", "comfy", ""])
        webapp.job.log("🎛️ ComfyUI · Sampling · 4s")
        self.assertEqual(webapp.job.lines[-1]["k"], "comfy")

    def test_the_workflow_of_the_last_run_can_be_looked_at(self):
        self.save_workflows()
        self.client.get("/api/workflows?pipeline=video")
        path = os.path.join(webapp.SAVED_WORKFLOWS_DIR, "wan.json")
        saved = webapp.job.workflow
        try:
            webapp.job.workflow = None
            self.assertEqual(self.client.get("/api/run-workflow").status_code, 404)
            webapp.job.workflow = {"path": path, "pipeline": "video", "options": {"steps": "7"}}
            reply = self.client.get("/api/run-workflow").get_json()
            self.assertEqual(reply["name"], "wan")
            self.assertEqual(next(option for option in reply["options"] if option["label"] == "Steps")["value"], "7")
            sampler = next(node for node in reply["nodes"] if node["type"] == "KSampler")
            self.assertEqual(sampler["inputs"]["steps"], "7", "the options this run was started with")
        finally:
            webapp.job.workflow = saved

    def test_the_built_in_workflow_is_still_the_default(self):
        self.assertEqual(video_panel.chosen_workflow({}), ({}, video_panel.BUILT_IN_TAKES, None))
        self.assertEqual(video_panel.chosen_workflow({"workflow": ""}), ({}, video_panel.BUILT_IN_TAKES, None))
        self.assertIn(b'<select id="videoWorkflow"', self.client.get("/").data)

    @unittest.skipUnless(os.name == "nt", "File Explorer is Windows only")
    def test_open_folder_only_on_this_pc(self):
        with mock.patch.object(webapp.os, "startfile") as startfile:
            self.assertTrue(self.client.post("/folders/uploaded/open").get_json()["ok"])
            startfile.assert_called_once_with(SHORTS_UPLOADED)
            self.assertTrue(os.path.isdir(SHORTS_UPLOADED), "it exists before the first upload")
            startfile.reset_mock()
            self.assertEqual(self.client.post("/folders/nowhere/open").status_code, 404)
            phone = {"REMOTE_ADDR": "192.168.1.20"}
            self.assertEqual(self.client.post("/folders/output/open", environ_base=phone).status_code, 403)
            startfile.assert_not_called()
            self.assertIn(b'File Explorer." hidden>', self.client.get("/", environ_base=phone).data)
        self.assertNotIn(b'File Explorer." hidden>', self.client.get("/").data)

    def test_upload_needs_keys(self):
        self.assertEqual(self.client.post("/publish", data={"privacy": "private"}).status_code, 400)
        self.assertIn(b"disabled title", self.client.get("/").data)

    def test_upload_queue(self):
        os.makedirs(SHORTS_READY, exist_ok=True)
        name = "short_20261101_120000_ltx.mp4"
        make_clip(os.path.join(SHORTS_READY, name), 1)
        self.addCleanup(lambda: os.path.exists(os.path.join(SHORTS_READY, name)) and os.remove(os.path.join(SHORTS_READY, name)))
        fake_upload = [sys.executable, "-c", "print('uploading')"]
        keys = {"YOUTUBE_CLIENT_ID": "a", "YOUTUBE_CLIENT_SECRET": "b", "YOUTUBE_REFRESH_TOKEN": "c"}
        with mock.patch.dict(os.environ, keys):
            with mock.patch.object(webapp.uploads, "start", wraps=lambda argv, env=None, keep_log=False, pipeline="shorts": webapp.Job.start(webapp.uploads, fake_upload)) as start:
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
            self.assertEqual(client.post("/generate/video").status_code, 302)
            self.assertIn(b"Wrong password", client.post("/login", data={"password": "nope"}).data)
            self.assertEqual(client.get("/").status_code, 302)
            reply = client.post("/login?next=//evil.example", data={"password": "open sesame"})
            self.assertEqual(reply.status_code, 302)
            self.assertNotIn("evil", reply.headers["Location"], "a link can't send you elsewhere after signing in")
            self.assertEqual(client.get("/").status_code, 200)
            self.assertIn(b'id="settingsBtn"', client.get("/").data)

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

    def test_settings_are_saved_and_checked(self):
        path = os.path.join(self.temp, "settings.json")
        with mock.patch.object(webapp.settings, "PATH", path), mock.patch.object(webapp, "apply_comfyui_priority"):
            self.assertFalse(self.client.get("/api/settings").get_json()["settings"]["low_priority"])
            saved = self.client.post("/api/settings", json={"low_priority": True, "default_workflow": {"image": "image_x", "bogus": "y"}}).get_json()
            self.assertTrue(saved["settings"]["low_priority"])
            self.assertEqual(saved["settings"]["default_workflow"], {"shorts": "", "video": "", "image": "image_x"})
            self.assertTrue(webapp.settings.load()["low_priority"], "kept in the file")
            self.assertEqual(webapp.priority_flags(), webapp.BELOW_NORMAL)
            saved = self.client.post("/api/settings", json={"reserve_ram_gb": "8"}).get_json()["settings"]
            self.assertEqual(saved["reserve_ram_gb"], 8)
            self.assertEqual(self.client.post("/api/settings", json={"reserve_ram_gb": 999}).get_json()["settings"]["reserve_ram_gb"], webapp.settings.MAX_RESERVE_GB)
            self.assertEqual(self.client.post("/api/settings", json={"reserve_ram_gb": "x"}).get_json()["settings"]["reserve_ram_gb"], 0)
            bat = os.path.join(self.temp, "run.bat")
            with open(bat, "w") as f:
                f.write("python main.py --enable-manager\n")
            self.client.post("/api/settings", json={"reserve_ram_gb": 8})
            self.assertTrue(webapp.comfyui_command(bat).endswith("--disable-auto-launch --cache-ram 8"))
            self.client.post("/api/settings", json={"reserve_ram_gb": 0})
            self.assertNotIn("--cache-ram", webapp.comfyui_command(bat))
            self.assertEqual(self.client.post("/api/settings", data="no").status_code, 400)
            self.assertIn("disk", self.client.get("/api/settings").get_json())

    def test_film_upscale_is_queued_and_checked(self):
        added = []
        film = os.path.join(webapp.VIDEO_OUTPUT, "zz_upscale_test.mp4")
        os.makedirs(webapp.VIDEO_OUTPUT, exist_ok=True)
        open(film, "wb").close()
        self.addCleanup(lambda: os.path.exists(film) and os.remove(film))
        with mock.patch.object(webapp.run_queue, "add", lambda item: added.append(item) or 1):
            self.assertEqual(self.client.post("/films/zz_upscale_test.mp4/upscale?scale=9").status_code, 400)
            self.assertEqual(self.client.post("/films/zz_upscale_test.mp4/upscale?quality=NOPE").status_code, 400)
            self.assertEqual(self.client.post("/films/missing.mp4/upscale").status_code, 404)
            body = self.client.post("/films/zz_upscale_test.mp4/upscale?scale=3&quality=ULTRA").get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(added[0]["argv"][-6:], ["--film", "zz_upscale_test.mp4", "--scale", "3", "--quality", "ULTRA"])

    def test_poweroff_is_per_session_and_cancellable(self):
        calls = []
        with mock.patch.object(webapp, "_run_shutdown", lambda cancel=False: calls.append(cancel)), \
                mock.patch.object(webapp, "POWEROFF_IDLE", 0), mock.patch.object(webapp, "work_left", lambda: False), \
                mock.patch.object(webapp, "POWEROFF_POLL", 0.01):
            self.assertFalse(self.client.get("/api/poweroff").get_json()["armed"])
            self.assertTrue(self.client.post("/api/poweroff", json={"armed": True}).get_json()["armed"])
            self.wait_until(lambda: webapp.poweroff_state()["seconds_left"] is not None, "the countdown")
            self.assertEqual(calls, [False])
            self.assertIn("poweroff", self.client.get("/api/status").get_json())
            self.assertFalse(self.client.post("/api/poweroff", json={"armed": False}).get_json()["armed"])
            self.assertEqual(calls, [False, True], "turned off, so the shutdown is cancelled")
            self.assertIsNone(webapp.poweroff_state()["seconds_left"])

    def test_restart_waits_for_work_to_end(self):
        self.comfy.script = ["hang"]
        self.assertTrue(self.generate().get_json()["ok"])
        self.wait_until(lambda: self.comfy.hanging, "the clip to start")
        self.assertEqual(self.client.post("/restart/panel").status_code, 409)
        self.assertEqual(self.client.post("/restart/comfyui").status_code, 409)

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
            self.assertEqual([item["prompt"] for item in json.load(f)], ["still queued"])
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

    def test_upscale_dialog_picks_only_method_buttons(self):
        # The dialog itself carries data-method, so a bare [data-method] lookup swallowed the Add to queue click.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("closest('button[data-method]')", html)
        self.assertNotIn("closest('[data-method]')", html)

    def test_stale_server_only_restarts_when_idle(self):
        with mock.patch.object(webapp.threading, "Timer") as timer:
            old = self.client.post("/launcher/refresh", data={"code_changed_at": "1"})
            self.assertEqual(old.get_json(), {"restarting": False})
            timer.assert_not_called()
            newer = {"code_changed_at": str(time.time() + 60)}
            # A queue about to start something isn't cut short; one on hold is saved and comes back on hold.
            with mock.patch.object(webapp.run_queue, "pending", return_value=[{"id": 1}]):
                with mock.patch.object(webapp.run_queue, "held", False):
                    self.assertEqual(self.client.post("/launcher/refresh", data=newer).get_json(), {"restarting": False})
                    timer.assert_not_called()
                with mock.patch.object(webapp.run_queue, "held", True):
                    self.assertEqual(self.client.post("/launcher/refresh", data=newer).get_json(), {"restarting": True})
            timer.assert_called_once()


if __name__ == "__main__":
    unittest.main()

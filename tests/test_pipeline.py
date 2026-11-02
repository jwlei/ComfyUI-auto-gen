"""
The pipeline against a fake ComfyUI on localhost: real HTTP, real files and real
ffmpeg on tiny synthetic clips, with no GPU and no Gemini. Run from the project root:

    python -m unittest discover tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import close, fake_claude, fake_comfy, make_clip, plan_reply

import ffmpeg
from PIL import Image

import main
from modules import comfy, media, paths, planner, publisher, run_state

SHOTS = [("Opening", 2), ("Middle", 2), ("Ending", 2)]


def pixel(path):
    with Image.open(path) as image:
        return image.convert("RGB").getpixel((5, 5))


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.image = os.path.join(self.temp, "input.jpg")
        Image.new("RGB", (640, 368), (10, 200, 30)).save(self.image)

        self.server = fake_comfy(self.temp)
        self.addCleanup(close, self.server)

        title_shots = ("A Test Film!", [{"title": t, "seconds": s, "prompt": f"{t} prompt"} for t, s in SHOTS])
        for patcher in (
            mock.patch.dict(os.environ, {"COMFYUI_URL": self.server.url, "GEMINI_API_KEY": "test"}),
            mock.patch.object(paths, "RUNS_DIR", os.path.join(self.temp, "runs")),
            mock.patch.object(paths, "OUTPUT_DIR", os.path.join(self.temp, "output")),
            mock.patch.object(paths, "UPLOADED_DIR", os.path.join(self.temp, "uploaded")),
            mock.patch.object(comfy, "POLL_INTERVAL", 0.01),
            mock.patch.object(main, "load_dotenv"),
            mock.patch.object(main.planner, "plan_film", return_value=title_shots),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def film(self, *extra):
        return main.main([self.image, "a short prompt", "--size", "64x64", *extra])

    def only_run(self):
        runs = run_state.list_runs()
        self.assertEqual(len(runs), 1)
        return runs[0]

    def test_chained_film(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()

        # The first clip starts on the input image, the rest on what came before it.
        self.assertEqual(len(self.server.prompts), 3)
        self.assertEqual(len(set(self.server.uploads)), 3)

        first = self.server.prompts[0]
        self.assertEqual(first["client_id"], "autocomfyui")
        workflow = first["prompt"]
        self.assertEqual(workflow["121"]["inputs"]["text"], "Opening prompt")
        self.assertEqual(workflow["167"]["inputs"]["image"], "uploaded.png")
        self.assertIs(workflow["290"]["inputs"]["value"], False)
        self.assertEqual(workflow["291"]["inputs"]["value"], 2)
        self.assertEqual((workflow["292"]["inputs"]["value"], workflow["293"]["inputs"]["value"]), (64, 64))
        self.assertIs(workflow["140"]["inputs"]["save_output"], False, "must stay out of ComfyUI's output folder")
        seeds = {p["prompt"]["115"]["inputs"]["noise_seed"] for p in self.server.prompts}
        self.assertEqual(len(seeds), 3)

        self.assertTrue(os.path.basename(state.final).startswith("A_Test_Film"))
        probe = ffmpeg.probe(state.final)
        kinds = [s["codec_type"] for s in probe["streams"]]
        self.assertIn("video", kinds)
        self.assertIn("audio", kinds)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        self.assertEqual((video["width"], video["height"]), (64, 64))
        # Two repeated start frames dropped: 6s less 2/24, give or take encoder padding.
        self.assertAlmostEqual(float(probe["format"]["duration"]), 6 - 2 / 24, delta=0.1)
        self.assertFalse(os.path.exists(state.parts_dir), "parts are only a step on the way")
        self.assertFalse(os.path.exists(state.final + ".part"))

    def test_failed_clip_continues_where_it_stopped(self):
        self.server.script = ["ok", "error", "error"]
        self.assertEqual(self.film(), 1)
        state = self.only_run()
        self.assertEqual(sorted(state.clips), [0])
        self.assertEqual(len(self.server.prompts), 3)
        self.assertIsNone(state.final)

        self.assertEqual(main.main(["--resume", state.id]), 0)
        # One clip was kept: only shots 2 and 3 render, and the first new one chains from it.
        self.assertEqual(len(self.server.prompts), 5)
        self.assertEqual(self.server.prompts[3]["prompt"]["121"]["inputs"]["text"], "Middle prompt")
        self.assertTrue(os.path.exists(self.only_run().final))

    def test_cut_off_clip_is_rendered_again(self):
        self.server.script = ["short"]
        self.assertEqual(self.film(), 0)
        self.assertEqual(len(self.server.prompts), 4)
        self.assertEqual(self.server.prompts[1]["prompt"]["121"]["inputs"]["text"], "Opening prompt")

    def test_clip_without_sound_keeps_the_rest_of_the_sound(self):
        self.server.script = ["ok", "silent", "ok"]
        self.assertEqual(self.film(), 0)
        audio = ffmpeg.probe(self.only_run().final, select_streams="a")["streams"][0]
        self.assertAlmostEqual(float(audio["duration"]), 6 - 2 / 24, delta=0.15)

    def test_stop_cancels_only_this_runs_clip(self):
        real_wait = comfy.ComfyClient._wait_for
        calls = []

        def stop_on_second(client, prompt_id):
            calls.append(prompt_id)
            if len(calls) == 2:
                self.server.queue = {
                    "queue_running": [[0, prompt_id, {}, {"client_id": client.client_id}],
                                      [1, "theirs", {}, {"client_id": "someone_else"}]],
                    "queue_pending": [[2, "mine", {}, {"client_id": client.client_id}],
                                      [3, "their_next", {}, {"client_id": "someone_else"}]]}
                raise KeyboardInterrupt
            return real_wait(client, prompt_id)

        with mock.patch.object(comfy.ComfyClient, "_wait_for", stop_on_second):
            self.assertEqual(self.film(), 130)

        self.assertEqual(self.server.cancels, [("/queue", {"delete": ["mine"]}), ("/interrupt", {"prompt_id": "p2"})])
        state = self.only_run()
        self.assertEqual(sorted(state.clips), [0])

        self.server.queue = {"queue_pending": [], "queue_running": []}
        self.assertEqual(main.main(["--resume", state.id]), 0)
        self.assertTrue(os.path.exists(self.only_run().final))

    def test_redo_in_a_chain_renders_what_follows(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()
        self.assertEqual(main.main(["--resume", state.id, "--redo", "2"]), 0)
        texts = [p["prompt"]["121"]["inputs"]["text"] for p in self.server.prompts[3:]]
        self.assertEqual(texts, ["Middle prompt", "Ending prompt"])

    def test_redo_with_image_start_renders_only_that_shot(self):
        self.assertEqual(self.film("--start", "image"), 0)
        state = self.only_run()
        self.assertEqual(len(set(self.server.uploads)), 1, "every clip starts on the input image")
        self.assertEqual(main.main(["--resume", state.id, "--redo", "2"]), 0)
        texts = [p["prompt"]["121"]["inputs"]["text"] for p in self.server.prompts[3:]]
        self.assertEqual(texts, ["Middle prompt"])
        # No frames dropped when nothing repeats.
        self.assertAlmostEqual(media.duration(self.only_run().final), 6, delta=0.1)

    def test_deleted_clip_counts_as_not_done(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()
        os.remove(state.clip_path(1))
        self.assertEqual(main.main(["--resume", state.id]), 0)
        self.assertEqual(len(self.server.prompts), 5)

    def test_only_stitching_left_needs_no_comfyui(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()
        os.remove(state.final)
        with mock.patch.dict(os.environ, {"COMFYUI_URL": "http://127.0.0.1:9"}):
            self.assertEqual(main.main(["--resume", state.id]), 0)
        self.assertEqual(len(self.server.prompts), 3)
        self.assertTrue(os.path.exists(self.only_run().final))

    def test_comfyui_down_doesnt_cost_a_redos_clips(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()
        with mock.patch.dict(os.environ, {"COMFYUI_URL": "http://127.0.0.1:9"}):
            self.assertEqual(main.main(["--resume", state.id, "--redo", "1"]), 1)
        self.assertEqual(sorted(self.only_run().clips), [0, 1, 2])

    def test_finished_run_renders_nothing_more(self):
        self.assertEqual(self.film(), 0)
        self.assertEqual(main.main(["--resume", self.only_run().id]), 0)
        self.assertEqual(len(self.server.prompts), 3)

    def test_plan_only_renders_nothing_and_edited_prompts_are_used(self):
        self.assertEqual(self.film("--plan-only"), 0)
        self.assertEqual(self.server.prompts, [])
        state = self.only_run()
        with open(state.path, encoding="utf-8") as f:
            saved = json.load(f)
        saved["shots"][0]["prompt"] = "My own prompt"
        with open(state.path, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        self.assertEqual(main.main(["--resume", state.id]), 0)
        self.assertEqual(self.server.prompts[0]["prompt"]["121"]["inputs"]["text"], "My own prompt")

    def test_comfyui_down_spends_nothing(self):
        with mock.patch.dict(os.environ, {"COMFYUI_URL": "http://127.0.0.1:9"}):
            self.assertEqual(self.film(), 1)
        main.planner.plan_film.assert_not_called()
        self.assertEqual(run_state.list_runs(), [])

    def test_no_key_for_the_chosen_llm_spends_nothing(self):
        with mock.patch.object(planner, "llm_configured", return_value=False):
            self.assertEqual(self.film("--llm", "claude"), 1)
        main.planner.plan_film.assert_not_called()

    def test_upload_when_done_moves_the_film_and_keeps_the_run(self):
        uploader = mock.Mock()
        uploader.return_value.upload.return_value = "https://www.youtube.com/watch?v=abc"
        with mock.patch.object(publisher, "YouTubeUploader", uploader), \
             mock.patch.object(planner, "write_metadata", return_value={"title": "T", "description": "D", "tags": ["x"]}):
            self.assertEqual(self.film("--upload", "--privacy", "unlisted"), 0)
        args = uploader.return_value.upload.call_args.args
        self.assertEqual(args[1:], ("T", "D", ["x"], "unlisted"))
        state = self.only_run()
        self.assertEqual(os.path.dirname(state.final), paths.UPLOADED_DIR, "a moved film still counts as finished")
        self.assertTrue(os.path.exists(state.final))

    def test_failed_upload_leaves_the_film_ready(self):
        uploader = mock.Mock()
        uploader.return_value.upload.return_value = None
        with mock.patch.object(publisher, "YouTubeUploader", uploader), \
             mock.patch.object(planner, "write_metadata", return_value={"title": "T", "description": "D", "tags": []}):
            self.assertEqual(self.film("--upload"), 1)
        state = self.only_run()
        self.assertEqual(os.path.dirname(state.final), paths.OUTPUT_DIR)


class PartsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)

    def test_last_frame_is_the_very_end(self):
        red, blue = os.path.join(self.temp, "r.mp4"), os.path.join(self.temp, "b.mp4")
        make_clip(red, 1.5, "red", audio=False)
        make_clip(blue, 0.5, "blue", audio=False)
        joined = os.path.join(self.temp, "joined.mp4")
        ffmpeg.concat(ffmpeg.input(red), ffmpeg.input(blue), v=1, a=0).output(joined, vcodec="libx264", pix_fmt="yuv420p") \
            .run(overwrite_output=True, quiet=True)
        r, g, b = pixel(media.last_frame(joined, os.path.join(self.temp, "last.png")))
        self.assertGreater(b, r)

    def test_sideways_phone_photo_is_stood_up(self):
        path = os.path.join(self.temp, "phone.jpg")
        exif = Image.Exif()
        exif[0x0112] = 6  # Rotated 90 degrees
        Image.new("RGB", (400, 200)).save(path, exif=exif)
        self.assertEqual(media.load_image(path).size, (200, 400))

    def test_not_an_image(self):
        path = os.path.join(self.temp, "notes.txt")
        with open(path, "w") as f:
            f.write("hello")
        with self.assertRaises(RuntimeError):
            media.load_image(path)

    def test_render_size_keeps_the_shape_on_the_grid(self):
        self.assertEqual(media.render_size(1280, 736), (1280, 736))
        self.assertEqual(media.render_size(736, 1280), (736, 1280))
        width, height = media.render_size(4032, 3024)
        self.assertEqual((width % 32, height % 32), (0, 0))
        self.assertAlmostEqual(width / height, 4 / 3, delta=0.03)
        self.assertAlmostEqual(width * height / (1280 * 736), 1, delta=0.05)


class PlanTest(unittest.TestCase):
    def test_durations(self):
        self.assertEqual(planner.fit_durations([8, 8, 8], 24), [8, 8, 8])
        self.assertEqual(sum(planner.fit_durations([1, 30, 8], 24)), 24)
        self.assertEqual(planner.fit_durations([10, 10, 10], 15), [5, 5, 5])
        self.assertTrue(all(5 <= s <= 10 for s in planner.fit_durations([9, 9, 9, 9], 30)))
        self.assertEqual(sum(planner.fit_durations([9, 9, 9, 9], 30)), 30)

    def test_shot_counts_for_the_film_limits(self):
        for seconds in range(planner.MIN_FILM_SECONDS, planner.MAX_FILM_SECONDS + 1):
            count = planner.shot_count(seconds)
            self.assertTrue(count * planner.MIN_SHOT_SECONDS <= seconds <= count * planner.MAX_SHOT_SECONDS, seconds)
        self.assertEqual(planner.shot_count(5), 1)
        self.assertEqual(planner.shot_count(600), 75)

    def test_a_ten_minute_plan_fits(self):
        shots = [{"title": f"S{i}", "seconds": 8, "prompt": f"p{i}"} for i in range(75)]
        with mock.patch.object(planner, "_ask", return_value=self.reply(shots)):
            _, planned = planner.plan_film(b"jpeg", "brief", 600, "chain")
        self.assertEqual((len(planned), sum(s["seconds"] for s in planned)), (75, 600))

    def test_very_short_films_arent_asked_for_a_story(self):
        one = planner._instructions("a storm", 5, "chain")
        self.assertIn("made of one clip", one)
        self.assertIn("exactly one shot.", one)
        self.assertIn("one clear moment", one)
        self.assertNotIn("last frame of the clip before", one, "there is no clip before")
        self.assertNotIn("a real ending", one)
        self.assertIn("a real ending", planner._instructions("a storm", 60, "chain"))

    def reply(self, shots, title="Film"):
        return json.dumps({"title": title, "shots": shots})

    def test_plan_is_fitted_to_the_length(self):
        shots = [{"title": f"S{i}", "seconds": 9, "prompt": f"p{i}"} for i in range(8)]
        with mock.patch.object(planner, "_ask", return_value=self.reply(shots)):
            title, planned = planner.plan_film(b"jpeg", "brief", 60, "chain")
        self.assertEqual(title, "Film")
        self.assertEqual(sum(s["seconds"] for s in planned), 60)

    def test_bad_replies_are_asked_again_then_given_up_on(self):
        too_few = self.reply([{"title": "S", "seconds": 8, "prompt": "p"}])
        good = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": f"p{i}"} for i in range(8)])
        with mock.patch.object(planner, "_ask", side_effect=["not json", too_few, good]) as call:
            planner.plan_film(b"jpeg", "brief", 60, "chain")
        self.assertEqual(call.call_count, 3)
        with mock.patch.object(planner, "_ask", return_value=too_few):
            with self.assertRaises(RuntimeError):
                planner.plan_film(b"jpeg", "brief", 60, "chain")

    def test_instructions_carry_the_brief_and_the_start_mode(self):
        chain = planner._instructions("a storm rolls in", 120, "chain")
        self.assertIn("a storm rolls in", chain)
        self.assertIn("15 consecutive clips", chain)
        self.assertIn("last frame", chain)
        self.assertIn("starts again from the attached image", planner._instructions("x", 120, "image"))

    def test_one_guide_for_every_video_model(self):
        guide = planner._instructions("a storm rolls in", 60, "chain")
        self.assertIn("One flowing paragraph in the present tense, 60-120 words", guide)
        self.assertIn("Name one camera move, and describe the light and the sound", guide)
        self.assertNotRegex(guide, "LTX|Wan|H3|MiniMax", "it would be wrong about two of the three")

    def test_references_are_named_in_the_same_guide(self):
        two = planner._instructions("x", 60, "image", 2)
        self.assertIn("<Picture 1> and <Picture 2>", two)
        self.assertIn("One flowing paragraph", two)
        self.assertNotRegex(two, "subject_definitions|retention_analysis|<Subject")
        self.assertIn("Every clip is made on its own", two)

    def test_without_an_image_everything_is_described(self):
        none = planner._instructions("x", 60, "image", 0)
        self.assertIn("There is no image", none)
        self.assertIn("describe the subject, place, light", none)
        self.assertNotIn("attached image", none)

    def test_a_plain_paragraph_is_accepted_with_references(self):
        reply = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": "<Picture 1> walks into the rain."} for i in range(8)])
        with mock.patch.object(planner, "_ask", return_value=reply) as call:
            _, planned = planner.plan_film([b"one", b"two"], "brief", 60, "image")
        self.assertEqual(call.call_count, 1, "asked once, and not sent back for another shape")
        self.assertEqual(planned[0]["prompt"], "<Picture 1> walks into the rain.")

    def test_metadata_failure_falls_back_to_the_title(self):
        with mock.patch.object(planner, "_ask", side_effect=RuntimeError("no quota")):
            metadata = planner.write_metadata("My Film", "brief", [{"title": "S"}])
        self.assertEqual(metadata, {"title": "My Film", "description": "brief", "tags": []})


class ClaudeTest(unittest.TestCase):
    """The real SDK, speaking its real streaming protocol to a fake server."""

    def setUp(self):
        self.server = fake_claude()
        self.addCleanup(close, self.server)
        patcher = mock.patch.dict(os.environ, {"ANTHROPIC_BASE_URL": self.server.url, "ANTHROPIC_API_KEY": "test",
                                               "ANTHROPIC_AUTH_TOKEN": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def plan(self):
        return planner.plan_film(b"jpeg-bytes", "a storm", 60, "chain", "claude")

    def test_request_and_reply(self):
        title, shots = self.plan()
        self.assertEqual(title, "The Harbour Storm")
        self.assertEqual(sum(shot["seconds"] for shot in shots), 60)

        sent = self.server.requests[0]
        body = sent["body"]
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertIs(body["stream"], True)
        self.assertEqual(body["max_tokens"], planner.CLAUDE_MAX_TOKENS)
        self.assertEqual(body["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", sent["beta"])
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertEqual(body["output_config"]["format"]["schema"], planner.PLAN_SCHEMA)
        self.assertNotIn("thinking", body)
        self.assertNotIn("temperature", body)
        image, text = body["messages"][0]["content"]
        self.assertEqual(image["type"], "image")
        self.assertEqual(image["source"]["media_type"], "image/jpeg")
        self.assertIn("a storm", text["text"])

    def test_model_can_be_changed(self):
        with mock.patch.dict(os.environ, {"CLAUDE_MODEL": "claude-sonnet-5-5"}):
            self.plan()
        self.assertEqual(self.server.requests[0]["body"]["model"], "claude-sonnet-5-5")

    def test_after_a_fallback_the_last_text_is_the_answer(self):
        self.server.reply = ['{"title": "half a repl'] + plan_reply()
        self.assertEqual(self.plan()[0], "The Harbour Storm")

    def test_refusal_says_so_without_retrying(self):
        self.server.stop_reason = "refusal"
        with self.assertRaisesRegex(RuntimeError, "declined"):
            self.plan()
        self.assertEqual(len(self.server.requests), 1)

    def test_bad_key_says_so(self):
        self.server.status = 401
        with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
            self.plan()

    def test_truncated_reply_is_asked_again(self):
        self.server.stop_reason = "max_tokens"
        self.server.reply = ['{"title": "cut of']
        with self.assertRaisesRegex(RuntimeError, "usable plan"):
            self.plan()
        self.assertEqual(len(self.server.requests), planner.PLAN_ATTEMPTS)

    def test_metadata(self):
        self.server.reply = [json.dumps({"title": "Upload title", "description": "About it.", "tags": ["a", " "]})]
        metadata = planner.write_metadata("Film", "brief", [{"title": "S"}], "claude")
        self.assertEqual(metadata, {"title": "Upload title", "description": "About it.", "tags": ["a"]})
        self.assertEqual(self.server.requests[0]["body"]["output_config"]["format"]["schema"], planner.METADATA_SCHEMA)

    def test_credentials_are_noticed(self):
        self.assertTrue(planner.llm_configured("claude"))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "ANTHROPIC_PROFILE": ""}), \
             mock.patch.object(os.path, "isdir", return_value=False):
            self.assertFalse(planner.llm_configured("claude"))


class ArgsTest(unittest.TestCase):
    def test_durations(self):
        for text, seconds in (("5", 5), ("5s", 5), ("90", 90), ("3m", 180), ("2m30", 150), ("10m", 600)):
            self.assertEqual(main.parse_duration(text), seconds)
        for text in ("4", "10m1", "11m", "soon", "", "m"):
            with self.assertRaises(Exception):
                main.parse_duration(text)

    def test_default_duration_is_parsed(self):
        self.assertEqual(main.parse_args(["i.png", "p"]).duration, 120)

    def test_a_prompt_can_start_with_a_dash(self):
        self.assertEqual(main.parse_args(["--start", "image", "--", "i.png", "-a dash"]).prompt, "-a dash")

    def test_plan_only_cant_resume(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main.parse_args(["--resume", "run_1", "--plan-only"])


if __name__ == "__main__":
    unittest.main()

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

from pipelines.video import comfy, h3_prompt, main, media, paths, planner, roles, run_state, wan_prompt
from pipelines.video.run_eta import RunEta

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

    def fake_voice(self, seconds=20):
        def voice_text(text, path):
            self.voiced.append(text)
            ffmpeg.input(f"sine=f=440:d={seconds}", f="lavfi").output(path).run(overwrite_output=True, quiet=True)
            return path
        self.voiced = []
        return mock.patch.object(main, "AudioEngine", return_value=mock.Mock(voice_text=voice_text))

    def test_narration_sets_the_length_and_is_on_the_film(self):
        with self.fake_voice(), mock.patch.object(main, "LOGS_DIR", self.temp):
            self.assertEqual(self.film("--narration", "Hello there"), 0)
        state = self.only_run()
        self.assertTrue(state.has_audio)
        self.assertEqual(state.seconds, 20)
        self.assertEqual(self.voiced, ["Hello there"])
        kinds = sorted(stream["codec_type"] for stream in ffmpeg.probe(state.final)["streams"])
        self.assertEqual(kinds, ["audio", "video"])
        self.assertIn("Hello there", main.planner.plan_film.call_args.args[1], "the shots follow what is said")

    def test_narration_is_refused_past_the_longest_film(self):
        with self.fake_voice(601), mock.patch.object(main, "LOGS_DIR", self.temp):
            self.assertEqual(self.film("--narration", "Too long"), 1)
        main.planner.plan_film.assert_not_called()

    def test_audio_and_narration_are_one_or_the_other(self):
        with self.assertRaises(SystemExit):
            main.parse_args([self.image, "a prompt", "--audio", "a.wav", "--narration", "Hello"])

    def test_chained_film(self):
        self.assertEqual(self.film(), 0)
        state = self.only_run()

        # The first clip starts on the input image, the rest on what came before it.
        self.assertEqual(len(self.server.prompts), 3)
        self.assertEqual(len(set(self.server.uploads)), 3)

        first = self.server.prompts[0]
        self.assertEqual(first["client_id"], "comfyui-auto-gen")
        workflow = first["prompt"]
        self.assertEqual(workflow["121"]["inputs"]["text"], "Opening prompt")
        self.assertEqual(workflow["167"]["inputs"]["image"], "uploaded.png")
        self.assertIs(workflow["290"]["inputs"]["value"], False)
        self.assertEqual(workflow["291"]["inputs"]["value"], 2)
        self.assertEqual((workflow["292"]["inputs"]["value"], workflow["293"]["inputs"]["value"]), (64, 64))
        self.assertIs(workflow["140"]["inputs"]["save_output"], False, "must stay out of ComfyUI's output folder")
        used = [p["prompt"]["115"]["inputs"]["noise_seed"] for p in self.server.prompts]
        self.assertEqual(len(set(used)), 3)
        self.assertEqual([shot["used_seed"] for shot in state.shots], used, "each clip keeps the seed it was made with")
        self.assertEqual([shot["seed"] for shot in state.shots], [None] * 3, "and none is set for next time")

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


    def test_a_chosen_seed_counts_up_clip_by_clip(self):
        self.assertEqual(self.film("--seed", "4294967294"), 0)
        state = self.only_run()
        expected = [4294967294, 4294967295, 0]
        self.assertEqual([p["prompt"]["115"]["inputs"]["noise_seed"] for p in self.server.prompts], expected)
        self.assertEqual([shot["seed"] for shot in state.shots], expected)
        self.assertEqual([shot["used_seed"] for shot in state.shots], expected)
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

class SeedTest(unittest.TestCase):
    def test_seeds(self):
        from core import seeds
        self.assertIsNone(seeds.parse(""))
        self.assertIsNone(seeds.parse(" Random "))
        self.assertEqual(seeds.parse(" 42 "), 42)
        for wrong in ("-1", "4294967296", "1.5", "abc"):
            with self.assertRaises(ValueError):
                seeds.parse(wrong)
        self.assertEqual(seeds.nth(10, 2), 12)
        self.assertIsNone(seeds.nth(None, 2))
        self.assertEqual(seeds.pick(7), 7)
        self.assertNotEqual({seeds.pick(7, 2) for _ in range(5)}, {7}, "a rejected render tries a new seed")
        workflow = {"1": {"inputs": {"seed": 0}}, "2": {"inputs": {"noise_seed": 1}}, "3": {"inputs": {"seed": ["4", 0]}}}
        self.assertTrue(seeds.apply(workflow, 99))
        self.assertEqual((workflow["1"]["inputs"]["seed"], workflow["2"]["inputs"]["noise_seed"]), (99, 99))
        self.assertEqual(workflow["3"]["inputs"]["seed"], ["4", 0], "a seed wired from another node is left alone")
        self.assertFalse(seeds.apply({"1": {"inputs": {}}}, 5))
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main.parse_args(["a.png", "--seed", "abc", "a prompt"])


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

    def test_references_are_cut_to_what_h3_takes(self):
        clip, quiet, short = (os.path.join(self.temp, name) for name in ("clip.mp4", "quiet.mp4", "short.mp4"))
        make_clip(clip, 17)
        make_clip(quiet, 3, audio=False)
        make_clip(short, 1)
        self.assertEqual(media.media_kind(clip), "video")
        video = os.path.join(self.temp, "ref", "ref_video_1.mp4")
        self.assertTrue(media.reference_video(clip, video), "it has sound of its own")
        self.assertAlmostEqual(media.duration(video), 15, delta=0.2)
        self.assertEqual(ffmpeg.probe(video)["streams"][0]["r_frame_rate"], "24/1")
        self.assertFalse(media.reference_video(quiet, os.path.join(self.temp, "ref", "quiet.mp4")))
        audio = media.reference_audio(clip, os.path.join(self.temp, "ref", "ref_audio_1.wav"))
        self.assertEqual(media.media_kind(audio), "audio")
        self.assertAlmostEqual(media.duration(audio), 15, delta=0.2)
        with self.assertRaises(RuntimeError):
            media.reference_video(short, os.path.join(self.temp, "ref", "short.mp4"))
        self.assertIsNone(media.media_kind(os.path.join(self.temp, "nothing.mp4")))

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

    def test_size_presets_are_on_the_grid_and_end_at_the_full_size(self):
        for _, width, height in media.SIZE_PRESETS:
            self.assertEqual((media.on_grid(width), media.on_grid(height)), (width, height))
            self.assertAlmostEqual(width / height, 16 / 9, delta=0.05)
        self.assertEqual(media.SIZE_PRESETS[-1][1:], media.DEFAULT_SHAPE)
        self.assertEqual(media.SIZE_PRESETS[-1][1] * media.SIZE_PRESETS[-1][2], media.RENDER_AREA)

    def test_a_low_resolution_film_teaches_the_full_size_speed(self):
        def learned(*size, seconds_taken):
            with mock.patch.object(RunEta, "_save"):
                eta = RunEta(["--duration", "60", *size], "no_such_file.json")
                eta.clip(1, 8, 10)
                eta.attempt_started -= seconds_taken
                eta.clip(2, 8, 10)
            return eta.timings.render(eta.render_keys, None)
        share = 640 * 352 / media.RENDER_AREA
        self.assertAlmostEqual(learned("--size", "640x352", seconds_taken=100 * share), learned(seconds_taken=100), delta=0.1)

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

    def test_each_video_model_has_its_own_clip_lengths(self):
        self.assertEqual(planner.clip_seconds(None)[:2], (5, 10), "LTX")
        self.assertEqual(planner.clip_seconds(h3_prompt.MODEL)[:2], (5, 15), "H3 was trained on 124-362 frames at 24 fps")
        self.assertEqual(planner.clip_seconds(wan_prompt.MODEL)[:2], (3, 5), "Wan 2.2 was trained on 81 frames at 16 fps")
        for model in (None, h3_prompt.MODEL, wan_prompt.MODEL):
            shortest, longest, _ = planner.clip_seconds(model)
            for seconds in range(planner.MIN_FILM_SECONDS, planner.MAX_FILM_SECONDS + 1):
                count = planner.shot_count(seconds, model)
                self.assertTrue(count * shortest <= seconds <= count * longest, (model, seconds))
        self.assertEqual(planner.shot_count(15, h3_prompt.MODEL), 1, "one H3 clip holds 15 seconds")
        self.assertEqual(planner.fit_durations([15, 15], 30, h3_prompt.MODEL), [15, 15])
        self.assertEqual(planner.fit_durations([15, 15], 30), [10, 10], "past what LTX makes, then short of the film")

    def test_a_clip_length_chosen_within_the_models(self):
        self.assertEqual(planner.clip_seconds(h3_prompt.MODEL, (6, 9)), (6, 9, 8))
        self.assertEqual(planner.clip_seconds(None, (2, 30)), (5, 10, 8), "held to what LTX makes")
        self.assertEqual(planner.clip_seconds(wan_prompt.MODEL, (8, 9)), planner.clip_seconds(wan_prompt.MODEL), "nothing of it fits: the model's own")
        self.assertEqual(planner.shot_count(60, h3_prompt.MODEL, (5, 6)), 10, "short clips: more of them (60 at 6 each)")
        self.assertEqual(planner.shot_count(60, h3_prompt.MODEL), 5, "against 5 at H3's own 12")
        self.assertTrue(all(5 <= s <= 6 for s in planner.fit_durations([8] * 10, 60, h3_prompt.MODEL, (5, 6))))
        self.assertIn("a whole number from 5 to 15", planner._instructions("x", 60, "image", 1, h3_prompt.MODEL))
        self.assertIn("a whole number from 3 to 5", planner._instructions("x", 60, "chain", 1, wan_prompt.MODEL))

    def test_an_h3_plan_can_have_fifteen_second_clips(self):
        shots = [{"title": f"S{i}", "seconds": 15, "prompt": self.h3_prompt_text()} for i in range(4)]
        with mock.patch.object(planner, "_ask", return_value=self.reply(shots)):
            _, planned = planner.plan_film(b"jpeg", "brief", 60, "image", "gemini", h3_prompt.MODEL)
        self.assertEqual([shot["seconds"] for shot in planned], [15] * 4)

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

    # --- MiniMax H3 ---

    def h3_prompt_text(self, pictures="<Picture 1>"):
        return (f"subject_definitions:\n<Subject 1> is the woman from {pictures}.\n\nsummary:\n[reference generation] She walks.\n\n"
                "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - her hair.\n\n"
                f"detailed_description:\nLive-action. [Shot 1] {'She walks. ' * 100}\n\noverall_soundscape:\nRain.\n\nnon_diegetic_music: N/A")

    def test_h3_gets_its_own_instructions(self):
        two = planner._instructions("a storm rolls in", 60, "image", 2, h3_prompt.MODEL)
        self.assertIn("a storm rolls in", two)
        for needed in ("subject_definitions:", "retention_analysis:", "<Picture 1> and <Picture 2>", "exactly 5 shots in order",
                       "6 = 6.58 s", "350-500 words"):
            self.assertIn(needed, two)
        self.assertNotIn("LTX", two)
        self.assertNotRegex(two, "(?i)freckles across|hazel|harbour", "an example's looks end up in the film")

        one = planner._instructions("x", 60, "image", 1, h3_prompt.MODEL)
        self.assertIn("The attached image is <Picture 1>", one)
        self.assertIn("It is not a frame", one, "a reference isn't an opening image")
        self.assertNotIn("<Picture 2>", one)

        none = planner._instructions("x", 60, "image", 0, h3_prompt.MODEL)
        self.assertIn("integrated_multimodal_description:", none)
        self.assertNotIn("<Picture", none)
        self.assertNotIn("retention_analysis:", none)

    def test_a_long_h3_film_still_fits_in_one_reply(self):
        self.assertIn("350-500 words", planner._instructions("x", 60, "image", 1, h3_prompt.MODEL))
        self.assertIn("280-400 words", planner._instructions("x", 600, "image", 1, h3_prompt.MODEL), "50 clips of 12 s")
        low, high = h3_prompt._detail_words(75)
        self.assertLessEqual(75 * (high + h3_prompt.SECTION_WORDS), h3_prompt.FILM_WORDS)

    def test_other_workflows_keep_their_instructions(self):
        self.assertEqual(planner._instructions("x", 60, "chain"), planner._instructions("x", 60, "chain", 1, None))
        self.assertIn("LTX-2", planner._instructions("x", 60, "chain"))

    def test_clip_lengths_match_the_workflows_own_rounding(self):
        self.assertEqual([h3_prompt.clip_frames(s) for s in range(5, 11)], [124, 158, 175, 192, 226, 243])
        self.assertTrue(all(h3_prompt.clip_frames(s) % 17 == 5 for s in range(1, 16)))

    def test_a_prompt_h3_cant_use_is_asked_for_again(self):
        self.assertIsNone(h3_prompt.problem(self.h3_prompt_text(), 1))
        self.assertIn("<Picture 2>", h3_prompt.problem(self.h3_prompt_text(), 2))
        self.assertIsNone(h3_prompt.problem(self.h3_prompt_text("<Picture 1> and <Picture 2>"), 2))
        self.assertIn("subject_definitions", h3_prompt.problem("A woman walks in the rain.", 1))
        self.assertIn("more than one", h3_prompt.problem(self.h3_prompt_text().replace("[Shot 1] She", "[Shot 1] She stops. [Shot 2] She"), 1))
        sketch = self.h3_prompt_text().replace("She walks. " * 100, "She walks.")
        self.assertIn("words", h3_prompt.too_thin([sketch] * 8, 1))
        self.assertIsNone(h3_prompt.too_thin([sketch] + [self.h3_prompt_text()] * 7, 1), "one short clip isn't worth a new film")
        short = self.h3_prompt_text().replace("She walks. " * 100, "She walks. " * 40)
        self.assertIsNotNone(h3_prompt.too_thin([short] * 8, 1))
        self.assertIsNone(h3_prompt.too_thin([short] * 75, 1), "a long film has less room per clip")
        self.assertIn("<Subject 1>", h3_prompt.problem(self.h3_prompt_text().replace("<Subject 1>", "Woman"), 1))
        text_only = "integrated_multimodal_description: [Shot 1] " + "Rain falls. " * 100 + "\n\noverall_soundscape: Rain.\n\nnon_diegetic_music: N/A"
        self.assertIsNone(h3_prompt.problem(text_only, 0))

        paragraph = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": "A woman walks."} for i in range(8)])
        good = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": self.h3_prompt_text()} for i in range(8)])
        with mock.patch.object(planner, "_ask", side_effect=[paragraph, good]) as call:
            _, planned = planner.plan_film(b"jpeg", "brief", 60, "image", "gemini", h3_prompt.MODEL)
        self.assertEqual(call.call_count, 2)
        self.assertNotIn("Your last reply", call.call_args_list[0].args[1])
        self.assertIn("Your last reply came back with a prompt without subject_definitions", call.call_args_list[1].args[1])
        self.assertEqual(planned[0]["prompt"], self.h3_prompt_text())

        thin = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": sketch} for i in range(8)])
        with mock.patch.object(planner, "_ask", return_value=thin) as call:
            _, planned = planner.plan_film(b"jpeg", "brief", 60, "image", "gemini", h3_prompt.MODEL)
        self.assertEqual(call.call_count, 2, "asked once more, then a film of short prompts beats no film")
        self.assertEqual(len(planned), 8)
        with mock.patch.object(planner, "_ask", return_value=paragraph) as call:
            planner.plan_film(b"jpeg", "brief", 60, "image", "gemini", None)
        self.assertEqual(call.call_count, 1, "only H3 needs this shape")

    def test_every_clip_gets_the_same_subject_definitions_naming_every_picture(self):
        full = self.h3_prompt_text("<Picture 1>, with a scar on the left brow")
        drifted = self.h3_prompt_text("<Picture 1>").replace("She walks. ", "She stops. ")
        forgot = drifted.replace(" from <Picture 1>", "")
        tidied = h3_prompt.tidy([forgot, full, drifted], 1)
        self.assertEqual(len({prompt.split("summary:")[0] for prompt in tidied}), 1)
        self.assertIn("a scar on the left brow", tidied[0])
        self.assertIn("She stops.", tidied[0], "the rest of each clip stays its own")
        self.assertEqual(tidied[1], full)

        one_picture = h3_prompt.tidy([self.h3_prompt_text()], 2)[0]
        self.assertIn("<Subject 1> also draws on <Picture 2>.\n\nsummary:", one_picture)
        self.assertIsNone(h3_prompt.problem(one_picture, 2))

        bare = self.h3_prompt_text().replace("<Subject 1>", "Subject 1").replace("<Picture 1>", "Picture 1")
        self.assertEqual(h3_prompt.tidy([bare], 1), [self.h3_prompt_text()], "dropped brackets are put back")
        self.assertIn("[Shot 1]", h3_prompt.tidy([bare], 1)[0], "a shot isn't a label")
        self.assertEqual(h3_prompt.tidy(["A woman walks."], 1), ["A woman walks."], "asked for again, not patched")
        self.assertEqual(h3_prompt.tidy(["Text only."], 0), ["Text only."])

    # --- H3's reference videos and audio clips ---

    def test_references_are_labelled_as_h3_numbers_them(self):
        found = roles.references([(roles.MOTION, True), (roles.CHARACTER, False)], [roles.VOICE])
        self.assertEqual([(r.label, r.sound) for r in found], [("<Video 1>", "<Audio 1>"), ("<Video 2>", None), ("<Audio 2>", None)],
                         "a video's own sound is labelled before any audio clip")
        self.assertEqual(roles.labels(found), ["<Video 1>", "<Audio 1>", "<Video 2>", "<Audio 2>"])

    def test_h3_is_told_what_each_reference_is_for(self):
        found = roles.references([(roles.MOTION, False)], [roles.MUSIC])
        guide = planner._instructions("x", 60, "image", 1, h3_prompt.MODEL, references=found)
        self.assertIn("<Video 1> shows the movement and camera to follow", guide)
        self.assertIn("<Audio 1> is the film's music", guide)
        self.assertIn("can't watch or hear", guide)
        self.assertNotIn("<Video 1>", planner._instructions("x", 60, "image", 1, h3_prompt.MODEL))

    def test_an_h3_prompt_has_to_name_every_reference(self):
        found = roles.references([(roles.CHARACTER, True)], [roles.VOICE])
        text = self.h3_prompt_text()
        self.assertIn("<Video 1>", h3_prompt.problem(text, 1, found))
        tidied = h3_prompt.tidy([text], 1, (), found)[0]
        self.assertIsNone(h3_prompt.problem(tidied, 1, found))
        self.assertIn("Another character comes from <Video 1>.", tidied.split("summary:")[0], "where H3 reads who is in it")
        self.assertIn("<Subject 1> speaks with the voice in <Audio 2>.", tidied)
        self.assertEqual(h3_prompt.tidy(["Video 1 and Audio 1."], 0, (), found)[0].split(" ")[:3], ["<Video", "1>", "and"],
                         "dropped brackets are put back")

    def test_empty_reference_slots_come_out_of_the_workflow(self):
        workflow = {"1": {"inputs": {"prompt": "x", "ref_videos.ref_video_0": ["3", 0], "ref_video_audios.ref_video_audio_0": ["3", 1],
                                     "ref_videos.ref_video_1": ["5", 0], "ref_audios.ref_audio_0": ["6", 0]},
                          "class_type": h3_prompt.REFERENCE_NODE},
                    "2": {"inputs": {"file": ""}, "class_type": "LoadVideo"},
                    "3": {"inputs": {"video": ["2", 0]}, "class_type": "GetVideoComponents"},
                    "4": {"inputs": {"file": ""}, "class_type": "LoadVideo"},
                    "5": {"inputs": {"video": ["4", 0]}, "class_type": "GetVideoComponents"},
                    "6": {"inputs": {"audio": ""}, "class_type": "LoadAudio"}}
        fields = {"prompt": "1.prompt", "_most_references": 12,
                  "ref_video": "2.file", "ref_video_link": "1.ref_videos.ref_video_0", "ref_video_split": "3.video",
                  "ref_video_sound": "1.ref_video_audios.ref_video_audio_0",
                  "ref_video_2": "4.file", "ref_video_2_link": "1.ref_videos.ref_video_1", "ref_video_2_split": "5.video",
                  "ref_audio": "6.audio", "ref_audio_link": "1.ref_audios.ref_audio_0"}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "workflow.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(workflow, f)
            with open(os.path.join(folder, "workflow.map.json"), "w", encoding="utf-8") as f:
                json.dump(fields, f)
            with mock.patch.dict(os.environ, {"COMFYUI_WORKFLOW": path}):
                client = comfy.ComfyClient()
            self.assertEqual(comfy.workflow_model(path), h3_prompt.MODEL)
        self.assertEqual(client.takes(), {"min_images": 0, "max_images": 0, "max_videos": 2, "max_audios": 1, "max_references": 3})
        filled = json.loads(json.dumps(workflow))
        client._fill_media(filled, "video", ["mine.mp4"])
        client._fill_media(filled, "audio", [])
        self.assertEqual(filled["2"]["inputs"]["file"], "mine.mp4")
        self.assertEqual(sorted(filled), ["1", "2", "3"], "the second video's and the audio's nodes are gone")
        self.assertEqual(sorted(filled["1"]["inputs"]), ["prompt", "ref_video_audios.ref_video_audio_0", "ref_videos.ref_video_0"])

    def test_a_workflows_nodes_say_whose_prompts_it_gets(self):
        self.assertIsNone(comfy.ComfyClient().video_model(), "the built-in workflow")

        def model_of(node):
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "workflow.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"1": {"inputs": {"prompt": "x"}, "class_type": "Other"}, "2": node}, f)
                with open(os.path.join(folder, "workflow.map.json"), "w", encoding="utf-8") as f:
                    json.dump({"prompt": "1.prompt"}, f)
                with mock.patch.dict(os.environ, {"COMFYUI_WORKFLOW": path}):
                    return comfy.ComfyClient().video_model()

        self.assertEqual(model_of({"class_type": h3_prompt.REFERENCE_NODE, "inputs": {}}), h3_prompt.MODEL)
        self.assertEqual(model_of({"class_type": "Wan22ImageToVideoLatent", "inputs": {}}), wan_prompt.MODEL)
        self.assertEqual(model_of({"class_type": "CLIPLoader", "inputs": {"type": "wan"}}), wan_prompt.MODEL)
        self.assertIsNone(model_of({"class_type": "CLIPLoader", "inputs": {"type": "ltxv"}}))

    # --- Wan ---

    def test_wan_gets_plain_prose_without_sound(self):
        one = planner._instructions("a storm rolls in", 60, "chain", 1, wan_prompt.MODEL)
        self.assertIn("a storm rolls in", one)
        self.assertIn("from Wan", one)
        self.assertIn("exactly 12 shots in order", one, "Wan makes 5 s at most")
        self.assertIn("how things move", one)
        self.assertIn("exact last frame", one, "a continuous take works here")
        self.assertIn("Wan makes no sound", one)
        self.assertNotRegex(one, "subject_definitions|<Picture|<Subject|LTX")

        none = planner._instructions("x", 60, "image", 0, wan_prompt.MODEL)
        self.assertIn("There is no image", none)
        self.assertIn("Nothing shows Wan how anything looks", none)
        self.assertNotIn("attached image", none)
        self.assertNotIn("exact last frame", planner._instructions("x", 5, "chain", 1, wan_prompt.MODEL), "there is no clip before")

    # --- Image 2 ---

    def test_image_2_as_extra_context_is_the_subject_or_the_place(self):
        guide = planner._instructions("x", 60, "image", 2)
        self.assertIn("<Picture 1> shows the film's subject, and <Picture 2> adds to it", guide)
        self.assertIn("It shows the same subject", guide)
        self.assertIn("It shows a place", guide)
        self.assertIn("If the brief says how to use it", guide)
        self.assertNotIn("for you alone", guide)

    def test_image_2_only_the_llm_sees_is_put_into_words(self):
        guide = planner._instructions("x", 60, "chain", 1, None, planner.EXTRA_CONTEXT, True)
        self.assertIn("The first attached image is the film's opening frame", guide)
        self.assertIn("for you alone", guide)
        self.assertIn("early in the film", guide)
        self.assertIn("one prompt and one start frame", guide, "the video model still gets only the one")
        self.assertNotIn("<Picture", guide, "no label for an image the video model doesn't have")
        separate = planner._instructions("x", 60, "image", 1, None, planner.EXTRA_CONTEXT, True)
        self.assertIn("starts again from the first attached image", separate)
        self.assertIn("early in each clip", separate)

    def test_image_2_as_another_character(self):
        words = planner._instructions("x", 60, "chain", 1, None, planner.ANOTHER_CHARACTER, True)
        self.assertIn("the second image shows another character", words)
        self.assertIn("as asked for", words, "not ruled out by 'no new main characters'")
        self.assertIn("aren't in the opening frame", words)
        self.assertNotIn("It shows a place", words)
        references = planner._instructions("x", 60, "image", 2, None, planner.ANOTHER_CHARACTER)
        self.assertIn("<Picture 2> shows another character", references)
        self.assertNotIn("for you alone", references)

    def test_without_image_2_no_guide_mentions_one(self):
        for model in (None, h3_prompt.MODEL, wan_prompt.MODEL):
            for images in (0, 1):
                guide = planner._instructions("x", 60, "chain", images, model)
                self.assertNotRegex(guide, "second image|another character|It shows a place|<Picture 2>", model)

    def test_h3_defines_image_2_by_what_it_is(self):
        context = planner._instructions("x", 60, "image", 2, h3_prompt.MODEL)
        self.assertIn("<Picture 1> and <Picture 2>", context)
        self.assertIn("place is <Subject 2>", context)
        self.assertNotIn("comes from <Picture 1> and <Picture 2>:", context, "a place isn't part of what the subject looks like")
        character = planner._instructions("x", 60, "image", 2, h3_prompt.MODEL, planner.ANOTHER_CHARACTER)
        self.assertIn("two different subjects", character)
        self.assertNotIn("It shows a place", character)
        self.assertNotIn("Call them <Picture 2>", character, "H3 calls a subject by its own label")

    def test_h3_never_says_a_reference_is_a_frame(self):
        for role in planner.IMAGE_2_ROLES:
            guide = planner._instructions("x", 60, "image", 1, h3_prompt.MODEL, role, True)
            self.assertIn("for you alone", guide)
            self.assertNotIn("opening frame", guide, "no clip starts on it")

    def test_h3_tidying_keeps_another_character_apart(self):
        text = self.h3_prompt_text()
        merged = h3_prompt.tidy([text], 2)[0]
        self.assertIn("<Subject 1> also draws on <Picture 2>", merged)
        apart = h3_prompt.tidy([text], 2, True)[0]
        self.assertIn("<Subject 2> comes from <Picture 2>", apart)
        self.assertNotIn("<Subject 1> also draws on", apart)

    def test_wan_has_no_labels_for_image_2(self):
        for images, in_words in ((1, True), (2, False)):
            for role in planner.IMAGE_2_ROLES:
                guide = planner._instructions("x", 60, "chain", images, wan_prompt.MODEL, role, in_words)
                self.assertIn("the second image", guide)
                self.assertNotIn("<Picture", guide, "it takes plain prose")

    def test_each_image_after_the_first_has_its_own_role(self):
        roles = [planner.SAME_SUBJECT, planner.ANOTHER_CHARACTER, planner.PLACE, planner.PLACE]
        guide = planner._instructions("x", 60, "image", 5, None, roles)
        for needed in ("<Picture 2> shows the same subject as <Picture 1>", "<Picture 3> shows another character",
                       "Call them <Picture 3>", "<Picture 4> shows a place", "<Picture 5> shows a place", "moves between them"):
            self.assertIn(needed, guide)
        wan = planner._instructions("x", 60, "image", 3, wan_prompt.MODEL, [planner.PLACE, planner.ANOTHER_CHARACTER])
        self.assertIn("the second image shows a place", wan)
        self.assertIn("the third image shows another character", wan)
        self.assertNotIn("<Picture", wan)
        self.assertIn("the second image shows a place", planner._instructions("x", 60, "chain", 1, None, planner.PLACE, True))
        self.assertIn("shows the same subject again", planner._instructions("x", 60, "image", 2, None, planner.SAME_SUBJECT))

    def test_h3_numbers_subjects_by_role(self):
        roles = [planner.SAME_SUBJECT, planner.ANOTHER_CHARACTER, planner.PLACE, planner.EXTRA_CONTEXT]
        guide = planner._instructions("x", 60, "image", 5, h3_prompt.MODEL, roles)
        for needed in ("<Subject 1> also comes from <Picture 2>", "<Picture 3> is another character, <Subject 2>",
                       "<Picture 4> is a place, <Subject 3>", "If <Picture 5> shows the same subject"):
            self.assertIn(needed, guide)
        merged = h3_prompt.tidy([self.h3_prompt_text()], 4, roles[:3])[0]
        self.assertIn("<Subject 2> comes from <Picture 3>", merged)
        self.assertIn("<Subject 3> comes from <Picture 4>", merged)
        self.assertIn("<Subject 1> also draws on <Picture 2>", merged)

    def test_one_role_fingerprints_as_it_always_did(self):
        images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4), "red")]
        self.assertEqual(media.inputs_fingerprint(images, "context"), media.inputs_fingerprint(images, []))
        self.assertEqual(media.inputs_fingerprint(images, "character"), media.inputs_fingerprint(images, ["character"]))
        self.assertNotEqual(media.inputs_fingerprint(images, ["view"]), media.inputs_fingerprint(images, ["place"]))

    def test_roles_come_one_per_image(self):
        args = main.parse_args(["a.png", "--reference", "b.png", "--reference", "c.png", "--image-role", "view",
                                "--image-role", "place", "a prompt"])
        self.assertEqual(args.image_roles, ["view", "place"])
        self.assertEqual(main.parse_args(["a.png", "--image-2-role", "character", "a prompt"]).image_roles, ["character"],
                         "a queue saved before still runs")

    def test_plan_film_attaches_image_2_after_the_others(self):
        reply = self.reply([{"title": f"S{i}", "seconds": 8, "prompt": "A walk."} for i in range(8)])
        with mock.patch.object(planner, "_ask", return_value=reply) as call:
            planner.plan_film([b"one"], "brief", 60, "chain", "gemini", None, planner.EXTRA_CONTEXT, b"two")
        self.assertEqual(call.call_args.args[4], [b"one", b"two"])
        self.assertIn("for you alone", call.call_args.args[1])

    def test_reference_videos_and_audio_clips(self):
        args = main.parse_args(["a.png", "--ref-video", "walk.mp4", "--ref-video", "room.mp4", "--ref-video-role", "character",
                                "--ref-audio", "voice.wav", "--ref-audio-role", "voice", "a prompt"])
        videos, audios = main.reference_roles(args)
        self.assertEqual(videos, [("walk.mp4", "character"), ("room.mp4", "motion")], "one without a role is for movement")
        self.assertEqual(audios, [("voice.wav", "voice")])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main.parse_args(["a.png", "--ref-video-role", "place", "a prompt"])

        h3 = {"min_images": 0, "max_images": 9, "max_videos": 3, "max_audios": 3, "max_references": 12}
        main.check_references(h3, 9, [("v", "motion")] * 2, [("a", "sound")])
        for images, videos, audios, says in ((9, 3, 1, "12 references"), (1, 4, 0, "3 reference videos"), (1, 0, 4, "3 reference audio")):
            with self.assertRaises(RuntimeError) as caught:
                main.check_references(h3, images, [("v", "motion")] * videos, [("a", "sound")] * audios)
            self.assertIn(says, str(caught.exception))
        with self.assertRaises(RuntimeError) as caught:
            main.check_references({"min_images": 1, "max_images": 1}, 1, [("v", "motion")], [])
        self.assertIn("no reference videos", str(caught.exception))

    def test_an_image_2_needs_an_image_1(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            main.parse_args(["--no-image", "--reference", "b.png", "a prompt"])

    def test_wan_prompts_arent_held_to_h3s_shape(self):
        reply = self.reply([{"title": f"S{i}", "seconds": 5, "prompt": "A woman walks into the rain."} for i in range(12)])
        with mock.patch.object(planner, "_ask", return_value=reply) as call:
            _, planned = planner.plan_film(b"jpeg", "brief", 60, "chain", "gemini", wan_prompt.MODEL)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(planned[0]["prompt"], "A woman walks into the rain.")

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


class BusyComfyTest(unittest.TestCase):
    """ComfyUI too busy to answer for a while is waited out; silent for too long, it's given up."""

    def client(self):
        client = object.__new__(comfy.ComfyClient)
        client.url, client.timeout = "http://comfy", 3600
        return client

    def test_a_slow_reply_is_waited_out(self):
        done = mock.Mock(json=lambda: {"p1": {"status": {"completed": True}, "outputs": {}}})
        replies = [comfy.requests.ReadTimeout("read timeout=30"), comfy.requests.ReadTimeout("read timeout=30"), done]
        with mock.patch.object(comfy.requests, "get", side_effect=replies) as get, mock.patch.object(comfy.time, "sleep"):
            entry = self.client()._wait_for("p1")
        self.assertEqual(entry["status"], {"completed": True}, "the image it was making is still collected")
        self.assertEqual(get.call_count, 3)

    def test_silent_for_too_long_gives_up(self):
        now = [0.0]
        with mock.patch.object(comfy.requests, "get", side_effect=comfy.requests.ConnectionError("refused")),                 mock.patch.object(comfy.time, "time", lambda: now[0]),                 mock.patch.object(comfy.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)):
            with self.assertRaisesRegex(RuntimeError, "has not answered for 30 minutes"):
                self.client()._wait_for("p1")


if __name__ == "__main__":
    unittest.main()

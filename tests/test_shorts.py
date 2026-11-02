"""
Shorts' side of the panel and the shared quota rule. Nothing reaches Google and no job
starts: the queue and the voice request are stand-ins, and every file goes to a throwaway folder.
"""
import asyncio
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import DATA  # noqa: F401, sets the throwaway folder before anything else loads

from core import api_log, audio, gemini
from panel import app as _both_pipelines  # noqa: F401
from panel import server as webapp
from panel import shorts as shorts_panel
from pipelines.shorts import main as pipeline
from pipelines.shorts import run_state
from pipelines.shorts.asset_manager import AssetManager
from pipelines.shorts.brain import ContentBrain
from pipelines.shorts.run_state import RunState, list_runs
from pipelines.shorts.stock_check import is_text_card

BASE = {"runs": "1", "lang": "en-GB", "avatar_clips": "auto", "avatar_source": "comfyui", "avatar_style": "lipsync",
        "avatar_clips_min": "3", "avatar_clips_max": "5", "avatar_seconds_min": "3", "avatar_seconds_max": "6"}
TEXT_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. limit: 20, model: gemini-3.8-flash "
            "Please retry in 28m56.79798639s.', 'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}}")
IMAGE_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. limit: 0, model: gemini-3.1-flash-image', "
             "'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}}")
TTS_429 = "429: Rate limit exceeded for model gemini-3.8-flash-tts (limit: 10 requests per day on Free Tier). Please retry in 13h28m56s or upgrade your tier."
MINUTE_429 = "429 RESOURCE_EXHAUSTED. {'error': {'message': 'limit: 5', 'details': [{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}, {'retryDelay': '45s'}]}}"


class Refused(Exception):
    code = 429


class QuotaTest(unittest.TestCase):
    def setUp(self):
        self.clear()
        self.addCleanup(self.clear)
        self.asked, self.refuse = [], {}
        test = self

        class Models:
            def generate_content(self, model, contents, config=None):
                test.asked.append(model)
                if model in test.refuse:
                    raise Refused(test.refuse[model])
                return mock.Mock(text="fine")

        patches = [mock.patch.object(gemini, "_get_client", lambda: mock.Mock(models=Models())),
                   mock.patch.object(gemini.time, "sleep", lambda seconds: None)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def clear(self):
        os.makedirs(os.path.dirname(api_log.EXHAUSTED_PATH), exist_ok=True)
        if os.path.exists(api_log.EXHAUSTED_PATH):
            os.remove(api_log.EXHAUSTED_PATH)
        gemini._skipped_models.clear()

    def saved(self):
        with open(api_log.EXHAUSTED_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_the_wait_comes_from_the_response(self):
        self.assertAlmostEqual(api_log._wait_seconds(TEXT_429), 28 * 60 + 56.798, delta=1)
        self.assertEqual(api_log._wait_seconds(TTS_429), 13 * 3600 + 28 * 60 + 56)
        self.assertEqual(api_log._wait_seconds(MINUTE_429), 45)
        self.assertIsNone(api_log._wait_seconds(IMAGE_429))
        self.assertGreater(api_log._wait_seconds("retry in 28m56.79798639"), 28 * 60, "a message the log cut off")
        self.assertEqual(api_log._short_wait(TTS_429), "13h 28m")
        self.assertEqual(api_log._short_wait(MINUTE_429), "45s")
        self.assertEqual(api_log._short_wait("retry in 90s"), "1m")

    def test_with_no_wait_given_it_lasts_until_midnight_pacific(self):
        api_log.mark_exhausted("image", IMAGE_429)
        back = datetime.fromtimestamp(self.saved()["image"], ZoneInfo("America/Los_Angeles"))
        self.assertEqual((back.hour, back.minute), (0, 0))
        self.assertTrue(0 < back.timestamp() - time.time() <= 25 * 3600)

    def test_a_passed_reset_brings_the_model_back_by_itself(self):
        with open(api_log.EXHAUSTED_PATH, "w", encoding="utf-8") as f:
            json.dump({"old": time.time() - 1, "new": time.time() + 100, "junk": "x"}, f)
        self.assertFalse(api_log.is_exhausted("old"))
        self.assertTrue(api_log.is_exhausted("new"))
        self.assertFalse(api_log.is_exhausted("junk"))
        api_log.mark_exhausted("third", TTS_429)
        self.assertEqual(sorted(self.saved()), ["new", "third"], "keeps the others, drops what has passed")

    def test_a_broken_file_costs_one_request_not_a_run(self):
        with open(api_log.EXHAUSTED_PATH, "w", encoding="utf-8") as f:
            f.write("not json")
        self.assertFalse(api_log.is_exhausted("a"))
        api_log.mark_exhausted("a", TTS_429)
        self.assertTrue(api_log.is_exhausted("a"))

    def test_later_calls_go_straight_to_the_next_model_also_in_a_new_process(self):
        self.refuse["m1"] = TEXT_429
        chain = ["m1", "m2", "m3"]
        gemini._generate("x", models=chain)
        self.assertEqual(self.asked, ["m1", "m2"])
        self.asked.clear()
        gemini._generate("x", models=chain)
        self.assertEqual(self.asked, ["m2"])
        self.asked.clear()
        gemini._skipped_models.clear()  # As a new process starts: only the file carries it over
        gemini._generate("x", models=chain)
        self.assertEqual(self.asked, ["m2"])

    def test_everything_used_up_asks_nothing_and_says_why(self):
        for model in ("m1", "m2"):
            api_log.mark_exhausted(model, TEXT_429)
        with self.assertRaisesRegex(RuntimeError, r"m1 \(daily quota used up\)"):
            gemini._generate("x", models=["m1", "m2"])
        self.assertEqual(self.asked, [])

    def test_a_per_minute_limit_is_waited_out_and_never_marked(self):
        calls = []

        def once(model):
            calls.append(model)
            return Refused(MINUTE_429) if len(calls) == 1 else None

        class Models:
            def generate_content(self, model, contents, config=None):
                problem = once(model)
                if problem:
                    raise problem
                return mock.Mock(text="fine")

        with mock.patch.object(gemini, "_get_client", lambda: mock.Mock(models=Models())):
            gemini._generate("x", models=["m1", "m2"])
        self.assertEqual(calls, ["m1", "m1"])
        self.assertFalse(api_log.is_exhausted("m1"))

    def test_a_used_up_voice_model_is_skipped_by_later_runs(self):
        engine = audio.AudioEngine.__new__(audio.AudioEngine)
        engine.gemini_models, engine.gemini_voice, engine.style = ["v1", "v2"], "x", "calm"
        engine.timeout, engine.min_interval, engine.last_request = 1, 0, {}
        engine.output_dir = os.path.join(DATA, "voice_quota")
        os.makedirs(engine.output_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, engine.output_dir, ignore_errors=True)
        posted = []

        def fake_post(url, headers=None, json=None, timeout=None):
            posted.append(json["model"])
            message = TTS_429 if json["model"] == "v1" else "stop here"
            return mock.Mock(status_code=429 if json["model"] == "v1" else 400, text=message,
                             json=lambda: {"error": {"message": message}})

        script = [{"id": 1, "text": "Hello", "mood": "calm"}]
        with mock.patch.object(audio.requests, "post", fake_post), mock.patch.object(audio.time, "sleep", lambda seconds: None), \
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with self.assertRaises(RuntimeError):
                engine._narrate_gemini(script)
            self.assertEqual(posted[0], "v1")
            self.assertEqual(posted.count("v1"), 1)
            left = self.saved()["v1"] - time.time()
            self.assertAlmostEqual(left, 13 * 3600 + 28 * 60 + 56, delta=5)
            posted.clear()
            with self.assertRaisesRegex(RuntimeError, "v1: daily quota used up"):
                engine._narrate_gemini(script)
        self.assertNotIn("v1", posted)


class GenerateContinuesTest(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(run_state.RUNS_DIR, ignore_errors=True)
        os.makedirs(run_state.RUNS_DIR)
        self.addCleanup(shutil.rmtree, run_state.RUNS_DIR, ignore_errors=True)
        webapp._runs_cache["at"] = 0.0
        self.added = []
        patcher = mock.patch.object(webapp.run_queue, "add", lambda item: self.added.append(item) or False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = webapp.app.test_client()

    def stopped(self, run_id, topic, **fields):
        argv, env, error = shorts_panel.build_run(BASE)
        self.assertIsNone(error)
        state = RunState(id=run_id, argv=argv[len(shorts_panel.COMMAND):], env={k: v for k, v in env.items() if k in run_state.RUN_ENV},
                         number=1, videos_after=0, topic=topic, script=[{"id": 1, "text": "Hi"}], **fields)
        state.save()
        return state

    def generate(self, **form):
        return self.client.post("/generate/shorts", data={**BASE, **form}).get_json()

    def test_an_empty_topic_takes_the_oldest(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        self.stopped("run_20261102_100000_1", "New topic")
        self.assertEqual(self.generate()["continued"], "Old topic")
        self.assertEqual(self.added[0]["resume"], "run_20261101_100000_1")
        self.assertEqual(self.added[0]["argv"][-2:], ["--resume", "run_20261101_100000_1"])
        self.assertEqual(self.added[0]["pipeline"], "shorts")

    def test_a_typed_topic_takes_its_own(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        self.stopped("run_20261102_100000_1", "New topic")
        self.assertEqual(self.generate(topic="  new TOPIC ")["continued"], "New topic")

    def test_another_topic_or_language_is_a_new_video(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        self.assertNotIn("continued", self.generate(topic="Something else"))
        self.assertNotIn("resume", self.added[-1])
        self.assertNotIn("continued", self.generate(lang="en-US"))

    def test_nothing_stopped_is_a_new_video(self):
        self.assertNotIn("continued", self.generate())

    def test_a_run_in_hand_is_not_taken_twice(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        pending = [{"resume": "run_20261101_100000_1", "topic": "Old topic", "output": ""}]
        with mock.patch.object(webapp.run_queue, "pending", lambda: pending):
            self.assertIsNone(shorts_panel.run_to_continue("", shorts_panel.build_run(BASE)[0]))

    def after(self, **form):
        shutil.rmtree(run_state.RUNS_DIR, ignore_errors=True)
        state = self.stopped("run_20261101_100000_1", "T", voiced=True, video_pairs=[], avatar_indices=[0], reference_image="r.jpg",
                             avatar_clips={0: "c.mp4"}, avatar_prompts={0: "p"}, avatar_done=True, final="f.mp4")
        os.makedirs(state.audio_dir)
        take = os.path.join(state.audio_dir, "x.wav")
        open(take, "w").close()
        argv, env, error = shorts_panel.build_run({**BASE, **form})
        self.assertIsNone(error)
        state.adopt(argv[len(shorts_panel.COMMAND):], env, int(form.get("runs", 1)) - 1)
        with open(state.path, encoding="utf-8") as f:
            return json.load(f), os.path.exists(take)

    def test_the_same_options_keep_everything(self):
        saved, audio_kept = self.after()
        self.assertEqual(saved["avatar_clips"], {"0": "c.mp4"})
        self.assertTrue(saved["voiced"] and audio_kept)
        self.assertEqual(saved["final"], "f.mp4")

    def test_more_videos_cost_this_one_nothing(self):
        saved, _ = self.after(runs="3")
        self.assertEqual((saved["number"], saved["videos_after"]), (1, 2))
        self.assertTrue(saved["avatar_clips"])

    def test_another_workflow_renders_her_again_and_keeps_what_cost_tokens(self):
        saved, audio_kept = self.after(avatar_style="reaction")
        self.assertEqual(saved["avatar_clips"], {})
        self.assertIsNone(saved["final"])
        self.assertTrue(saved["voiced"] and audio_kept)
        self.assertEqual(saved["avatar_prompts"], {"0": "p"})
        self.assertEqual(saved["avatar_indices"], [0])

    def test_another_voice_is_voiced_again_and_keeps_the_script(self):
        voice = shorts_panel.VOICES[1][0]
        saved, audio_kept = self.after(voice=voice)
        self.assertFalse(saved["voiced"] or audio_kept)
        self.assertEqual(saved["avatar_clips"], {})
        self.assertTrue(saved["script"])
        self.assertEqual(saved["avatar_prompts"], {"0": "p"})

    def test_another_clip_length_plans_her_scenes_again(self):
        saved, _ = self.after(avatar_seconds_max="9")
        self.assertIsNone(saved["avatar_indices"])
        self.assertEqual(saved["avatar_prompts"], {})

    def test_the_estimate_says_so_before_you_press(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        self.assertEqual(self.client.post("/api/estimate/shorts", data=BASE).get_json()["continues"], "Old topic")
        self.assertNotIn("continues", self.client.post("/api/estimate/shorts", data={**BASE, "topic": "Another"}).get_json())

    def test_continue_and_discard_work_as_they_did(self):
        self.stopped("run_20261101_100000_1", "Old topic")
        self.assertEqual(self.client.post("/runs/run_20261101_100000_1/continue").get_json(), {"ok": True, "queued": False})
        self.added.clear()
        self.assertEqual(self.client.post("/runs/run_20261101_100000_1/discard").get_json(), {"ok": True})
        self.assertEqual(list_runs(), [])
        self.assertEqual(self.client.post("/runs/run_20261101_100000_1/continue").status_code, 404)

    def test_a_picked_topic_is_kept_and_not_paid_for_again(self):
        state = RunState.new(["--runs", "1"], 1, 0)
        state.save()
        self.assertEqual(list_runs(), [], "no topic yet: nothing to continue")
        state.update(topic="Picked")
        state.release_batch()
        self.assertEqual(list_runs()[0].topic, "Picked", "a failed script keeps the topic")

        asked = []

        class Brain:
            def get_trending_topic(self, *a):
                raise AssertionError("the topic was paid for already")

            def find_repeat_topic(self, *a):
                raise AssertionError("the topic was checked already")

            def generate_script(self, topic, lang=None):
                asked.append(topic)
                raise RuntimeError("stop here")

        with mock.patch.object(pipeline, "ContentBrain", Brain):
            args = pipeline.parse_args(["--resume", state.id])
            self.assertIsNone(asyncio.run(pipeline.generate_one(args, RunState.load(state.id))))
        self.assertEqual(asked, ["Picked"])

    def test_clip_prompts_survive_being_saved(self):
        state = RunState.new(["--runs", "1"], 1, 0)
        state.update(topic="T", script=[{"id": 1, "text": "a"}])
        state.keep_prompt(3, "kept")
        self.assertEqual(RunState.load(state.id).avatar_prompts, {3: "kept"})


class VoiceKeptTest(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.join(DATA, "voice_kept")
        shutil.rmtree(self.folder, ignore_errors=True)
        os.makedirs(self.folder)
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.asked = []
        self.scenes = lambda: [{"id": 1, "text": "x " * 20, "mood": ""}, {"id": 2, "text": "y " * 20, "mood": ""}]

    def wav(self, seconds_per_scene, gaps=True):
        rate = 8000
        samples = []
        for i, seconds in enumerate(seconds_per_scene):
            samples += [12000 if n % 8 < 4 else -12000 for n in range(int(rate * seconds))]
            if gaps and i < len(seconds_per_scene) - 1:
                samples += [0] * rate
        path = os.path.join(self.folder, "made.wav")
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data

    def engine(self):
        engine = audio.AudioEngine.__new__(audio.AudioEngine)
        engine.gemini_models, engine.gemini_voice, engine.style, engine.output_dir = ["v1"], "x", "calm", self.folder
        return engine

    def requests(self, reply):
        def request(engine, text, style, model):
            self.asked.append(text[:1])
            return reply(text)
        return mock.patch.object(audio.AudioEngine, "_gemini_request", request)

    def test_a_take_is_paid_for_once(self):
        take = self.wav([3, 3])
        with self.requests(lambda text: take):
            self.assertEqual(len(self.engine()._narrate_gemini(self.scenes())), 2)
            self.assertEqual(self.asked, ["x"])
            self.asked.clear()
            self.assertEqual(len(self.engine()._narrate_gemini(self.scenes())), 2)
        self.assertEqual(self.asked, [], "the take is there")

    def test_a_stop_part_way_through_the_scenes_only_asks_for_what_is_missing(self):
        take, one = self.wav([6], gaps=False), self.wav([2])

        def stops_at_y(text):
            if text.startswith("y"):
                raise KeyboardInterrupt
            return one if len(self.asked) > 1 else take

        with self.requests(stops_at_y), self.assertRaises(KeyboardInterrupt):
            self.engine()._narrate_gemini(self.scenes())
        self.assertEqual(self.asked, ["x", "x", "y"])
        self.asked.clear()
        with self.requests(lambda text: one):
            self.assertEqual(len(self.engine()._narrate_gemini(self.scenes())), 2)
        self.assertEqual(self.asked, ["y"])
        self.assertEqual([f for f in os.listdir(self.folder) if f.endswith(".tmp")], [])

    def test_a_doubled_take_is_voiced_once_more_and_not_used(self):
        doubled, good = self.wav([40], gaps=False), self.wav([6, 6])
        replies = iter([doubled, good])
        with self.requests(lambda text: next(replies)):
            narrated = self.engine()._narrate_gemini(self.scenes())
        self.assertEqual(self.asked, ["x", "x"], "one more take, not one per scene")
        self.assertEqual(len(narrated), 2)
        self.assertLess(sum(scene["duration"] for scene in narrated), 14)

    def test_a_kept_take_that_was_doubled_is_not_trusted(self):
        with open(os.path.join(self.folder, "narration_full_v1.wav"), "wb") as f:
            f.write(self.wav([40], gaps=False))
        good = self.wav([6, 6])
        with self.requests(lambda text: good):
            self.assertEqual(len(self.engine()._narrate_gemini(self.scenes())), 2)
        self.assertEqual(self.asked, ["x"], "a run continued later gets a clean take")

    def test_a_model_that_keeps_doubling_hands_over_to_the_next(self):
        doubled, good = self.wav([40], gaps=False), self.wav([6, 6])
        engine = self.engine()
        engine.gemini_models = ["v1", "v2"]

        def request(engine, text, style, model):
            self.asked.append(model)
            return doubled if model == "v1" else good

        with mock.patch.object(audio.AudioEngine, "_gemini_request", request):
            self.assertEqual(len(engine._narrate_gemini(self.scenes())), 2)
        self.assertEqual(self.asked, ["v1", "v1", "v2"])
        self.assertEqual(sorted(f for f in os.listdir(self.folder) if f.startswith("narration")), ["narration_full_v2.wav"])

    def test_normal_reads_pass_including_the_slowest_seen(self):
        engine = self.engine()
        path = os.path.join(self.folder, "slow.wav")
        with open(path, "wb") as f:
            f.write(self.wav([0.51 * 20], gaps=False))
        with self.requests(lambda text: self.fail("it was fine")):
            self.assertGreater(engine._read_once(path, "word " * 20, "calm", "v1"), 10)

    def test_a_cut_off_file_is_not_taken_for_a_good_one(self):
        take, one = self.wav([6], gaps=False), self.wav([2])
        with open(os.path.join(self.folder, "narration_full_v1.wav"), "wb") as f:
            f.write(take)
        with open(os.path.join(self.folder, "scene_1_v1.wav"), "wb") as f:
            f.write(one)
        with open(os.path.join(self.folder, "scene_2_v1.wav"), "wb") as f:
            f.write(b"RIFF")
        with self.requests(lambda text: one):
            self.engine()._narrate_gemini(self.scenes())
        self.assertEqual(self.asked, ["y"])


class TextCardTest(unittest.TestCase):
    """Stock clips that are just text on a colour. Their tags don't say so, so the frames decide."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp(dir=DATA)
        cls.flat = os.path.join(cls.folder, "flat.mp4")
        cls.busy = os.path.join(cls.folder, "busy.mp4")
        for path, source in ((cls.flat, "color=c=orange:s=360x640:r=24:d=2"),
                             (cls.busy, "color=c=gray:s=360x640:r=24:d=2,noise=alls=80:allf=t")):
            ffmpeg.input(source, f="lavfi").output(path, vcodec="libx264", pix_fmt="yuv420p", preset="ultrafast") \
                .run(overwrite_output=True, quiet=True)

    def manager(self):
        with mock.patch.dict(os.environ, {"VIDEO_PROVIDER": "pixabay", "PIXABAY_API_KEY": "x"}):
            manager = AssetManager(clips_dir=tempfile.mkdtemp(dir=self.folder))
        manager.downloaded = []

        def download(url, filename):
            manager.downloaded.append(url)
            path = os.path.join(manager.assets_dir, f"{len(manager.downloaded)}_{filename}")
            shutil.copy(self.flat if url.startswith("card") else self.busy, path)
            return path
        manager.download_video = download
        return manager

    @staticmethod
    def clip(url):
        return {"url": url, "tags": f"tags of {url}", "duration": 5, "width": 360, "height": 640}

    def test_a_flat_clip_is_asked_about_and_the_answer_decides(self):
        sheets = []
        self.assertTrue(is_text_card(self.flat, lambda sheet: sheets.append(sheet) or True))
        self.assertTrue(sheets[0].startswith(b"\xff\xd8"), "the model gets one JPEG")
        self.assertFalse(is_text_card(self.flat, lambda sheet: False), "a flat night sky stays")

    def test_a_detailed_clip_costs_no_question(self):
        self.assertFalse(is_text_card(self.busy, lambda sheet: self.fail("asked about a detailed clip")))

    def test_an_unreadable_clip_is_not_called_one(self):
        broken = os.path.join(self.folder, "broken.mp4")
        with open(broken, "wb") as f:
            f.write(b"not a video")
        self.assertFalse(is_text_card(broken, lambda sheet: self.fail("nothing to ask about")))

    def test_the_answer_is_read_from_the_last_line(self):
        for reply, card in (("Orange with the word skeleton\nB", True), ("A drawn stomach on orange\n**C**", True),
                            ("Dashcam on a night road\nA", False), ("Something odd\nD.", False)):
            with mock.patch("pipelines.shorts.brain._generate", return_value=mock.Mock(text=reply)):
                self.assertEqual(ContentBrain().shows_only_text(b"jpeg"), card, reply)

    def test_no_answer_means_it_is_not_used(self):
        with mock.patch("pipelines.shorts.brain._generate", side_effect=RuntimeError("quota")):
            self.assertTrue(ContentBrain().shows_only_text(b"jpeg"))
        with mock.patch("pipelines.shorts.brain._generate", return_value=mock.Mock(text="")):
            self.assertTrue(ContentBrain().shows_only_text(b"jpeg"))

    def test_a_picked_text_card_is_traded_for_the_next_clip(self):
        manager = self.manager()
        results = {"one": [self.clip("card_1"), self.clip("real_1")], "two": [self.clip("real_2")]}
        manager.find_candidates = lambda query: results[query]
        scene = {"id": 1, "text": "A line", "visual_1": "one", "visual_2": "two"}

        pair, = manager.get_videos([scene], lambda searches: {0: 0, 1: 0}, lambda sheet: True)

        self.assertEqual(manager.downloaded, ["card_1", "real_1", "real_2"])
        self.assertEqual([part["found"] for part in scene["footage"]], ["tags of real_1", "tags of real_2"], "the fit is rated on what's shown")
        self.assertFalse(os.path.exists(os.path.join(manager.assets_dir, "1_scene_1_a.mp4")), "the card is deleted")
        self.assertTrue(all(os.path.exists(path) for path in pair))

    def test_a_text_card_never_comes_back_for_a_later_scene(self):
        manager = self.manager()
        manager.find_candidates = lambda query: [self.clip("card_1"), self.clip(f"real_{query}")]
        scenes = [{"id": n, "text": "A line", "visual_1": f"{n}a", "visual_2": f"{n}b"} for n in (1, 2)]

        manager.get_videos(scenes, lambda searches: {i: 0 for i in range(len(searches))}, lambda sheet: True)

        self.assertEqual(manager.downloaded.count("card_1"), 1)
        self.assertEqual(manager.rejected, {"card_1"})

    def test_one_clip_carries_the_scene_when_the_other_search_has_only_cards(self):
        manager = self.manager()
        results = {"one": [self.clip("card_1")], "two": [self.clip("real_2")]}
        manager.find_candidates = lambda query: results[query]
        scene = {"id": 1, "text": "A line", "visual_1": "one", "visual_2": "two"}

        pair, = manager.get_videos([scene], lambda searches: {0: 0, 1: 0}, lambda sheet: True)

        self.assertEqual(pair[0], pair[1], "a pair of the same clip plays as one shot")
        self.assertEqual([part["found"] for part in scene["footage"]], [None, "tags of real_2"])

    def test_a_scene_with_only_cards_has_no_footage_rather_than_a_card(self):
        manager = self.manager()
        manager.find_candidates = lambda query: [self.clip(f"card_{query}")]
        scene = {"id": 1, "text": "A line", "visual_1": "one", "visual_2": "two"}

        self.assertEqual(manager.get_videos([scene], None, lambda sheet: True), [None])
        self.assertEqual([part["found"] for part in scene["footage"]], [None, None], "so the avatar goes here first")

    def test_the_pipeline_passes_the_check_along(self):
        with open(os.path.join(ROOT, "pipelines", "shorts", "main.py"), encoding="utf-8") as f:
            self.assertIn("get_videos(script, brain.pick_stock_clips, brain.shows_only_text)", f.read())


class SwitchTest(unittest.TestCase):
    def setUp(self):
        self.client = webapp.app.test_client()

    def test_a_short_has_a_poster_as_it_loads_nothing_until_played(self):
        folder = webapp.FOLDERS["output"]
        os.makedirs(folder, exist_ok=True)
        files = [os.path.join(folder, name) for name in ("a.mp4", "a.still.jpg", "b.mp4")]
        for path in files:
            open(path, "wb").close()
            self.addCleanup(os.remove, path)
        self.assertEqual(self.client.get("/posters/output/a.mp4").status_code, 200)
        self.assertEqual(self.client.get("/posters/output/b.mp4").status_code, 404)
        self.assertEqual(self.client.get("/posters/output/nope.mp4").status_code, 404)
        self.assertEqual(self.client.get("/posters/output/..%2F..%2Fsecret.mp4").status_code, 404)

    def test_the_status_is_never_cached_and_names_its_server(self):
        reply = self.client.get("/api/status")
        self.assertEqual(reply.headers["Cache-Control"], "no-store")
        self.assertEqual(reply.get_json()["server_started"], webapp.STARTED_AT)

    def test_both_addresses_open_the_same_page_on_their_side(self):
        for address, side in (("/", ""), ("/shorts", "shorts"), ("/video", "video")):
            page = self.client.get(address).get_data(as_text=True)
            self.assertIn('id="switchShorts"', page)
            self.assertIn('id="switchVideo"', page)
            self.assertIn(f'let side = "{side}";', page)

    def test_upload_is_only_on_shorts(self):
        page = self.client.get("/").get_data(as_text=True)
        video_form = page[page.index('id="formVideo"'):page.index('<div class="side">')]
        self.assertNotIn('name="upload"', video_form)
        self.assertNotIn('name="privacy"', video_form)
        shorts_form = page[page.index('id="formShorts"'):page.index('id="formVideo"')]
        self.assertIn('name="upload"', shorts_form)

    def test_workflows_are_kept_apart_by_their_name(self):
        saved = [{"name": "a_ytautoshort.json", "label": "a_ytautoshort", "takes": {}}, {"name": "b.json", "label": "b", "takes": {}},
                 {"name": "c.json", "label": "c", "problem": "No."}, {"name": "d_ytautoshort.json", "label": "d_ytautoshort", "problem": "No."}]
        with mock.patch.object(webapp, "saved_workflows", lambda: saved):
            names = lambda side: [w["name"] for w in self.client.get(f"/api/workflows?pipeline={side}").get_json()["workflows"]]
            self.assertEqual(names("shorts"), ["a_ytautoshort.json"])
            self.assertEqual(names("video"), ["b.json"])


if __name__ == "__main__":
    unittest.main()

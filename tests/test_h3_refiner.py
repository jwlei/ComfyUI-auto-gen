import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import DATA  # noqa: F401, isolate application data before importing the panel

from core import director, h3_refiner
from panel import video
from panel import shorts
from pipelines.shorts.avatar_generator import AvatarGenerator
from pipelines.video import main
from pipelines.video.run_state import RunState


class H3RefinerTest(unittest.TestCase):
    def test_settings_and_video_form(self):
        self.assertEqual(h3_refiner.settings(None, None), ("off", "balanced"))
        for creativity in h3_refiner.CREATIVITY:
            flags, error = video.build_options({"h3_enhance": "review", "h3_creativity": creativity})
            self.assertIsNone(error)
            self.assertEqual(video.option(flags, "--h3-creativity"), creativity)
            self.assertEqual(video.option(flags, "--h3-enhance"), "review")
        flags, error = video.build_options({"h3_enhance": "render"})
        self.assertIsNone(error)
        self.assertEqual(video.option(flags, "--h3-creativity"), "balanced")
        self.assertIsNotNone(video.build_options({"h3_enhance": "render", "h3_creativity": "invented"})[1])
        self.assertNotIn("--h3-enhance", video.build_options({})[0])

    def test_director_anchor_depends_on_actual_image_mode(self):
        graph = {"1": {"class_type": director.DIRECTOR, "inputs": {"director_inputs": [director.INPUTS_ID, 0]}},
                 director.INPUTS_ID: {"class_type": director.INPUTS, "_meta": {},
                                      "inputs": {"ref_assets_1": [director.ASSETS_ID, 0]}},
                 director.ASSETS_ID: {"class_type": director.ASSETS}}
        self.assertTrue(h3_refiner.supports(graph))
        self.assertTrue(h3_refiner.first_frame(graph, ["one"]))
        self.assertFalse(h3_refiner.first_frame(graph, ["one", "two"]))
        graph[director.INPUTS_ID]["_meta"][director.IMAGES_AS] = "reference"
        self.assertFalse(h3_refiner.first_frame(graph, ["one"]))
        self.assertFalse(h3_refiner.supports({"1": {"class_type": director.DIRECTOR, "inputs": {}}}))

    def test_compiler_passes_creativity_and_checks_wiring(self):
        with tempfile.TemporaryDirectory() as temp:
            image = os.path.join(temp, "image.png")
            with open(image, "wb") as file:
                file.write(b"image")
            refiner = object.__new__(h3_refiner.Refiner)
            refiner.root, refiner.python, refiner.url = temp, "python", "http://127.0.0.1:42/v1"
            captured = []

            def invoke(args, **kwargs):
                request = json.loads(kwargs["input"])
                captured.append(request)
                return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                    "prompt": "enhanced", "mode": "ref2va", "frames": 124,
                    "wiring": [{"wiring": "ref_image_1", "sha256": h3_refiner.digest(image)}],
                }))

            with mock.patch.object(h3_refiner.subprocess, "run", side_effect=invoke):
                for creativity in h3_refiner.CREATIVITY:
                    self.assertEqual(refiner.compile("a scene", 5, 1280, 736, creativity, [image]), "enhanced")
                self.assertEqual([x["brief"]["creativity"] for x in captured], list(h3_refiner.CREATIVITY))
                self.assertEqual(captured[0]["brief"]["assets"][0]["role"], "subject")

            def wrong(*args, **kwargs):
                return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                    "prompt": "enhanced", "mode": "ref2va", "frames": 124,
                    "wiring": [{"wiring": "ref_image_2", "sha256": h3_refiner.digest(image)}],
                }))
            with mock.patch.object(h3_refiner.subprocess, "run", side_effect=wrong):
                with self.assertRaisesRegex(RuntimeError, "bindings"):
                    refiner.compile("a scene", 5, 1280, 736, "balanced", [image])

    def test_video_audio_and_paired_soundtrack_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [os.path.join(temp, name) for name in ("image", "video", "audio")]
            for path in paths:
                with open(path, "wb") as file:
                    file.write(path.encode())
            refiner = object.__new__(h3_refiner.Refiner)
            refiner.root, refiner.python, refiner.url = temp, "python", "http://127.0.0.1:42/v1"

            def invoke(args, **kwargs):
                request = json.loads(kwargs["input"])
                assets = request["brief"]["assets"]
                self.assertEqual([a["kind"] for a in assets], ["image", "video", "audio", "audio"])
                self.assertEqual(assets[2]["paired_video_path"], paths[1])
                self.assertEqual(assets[3]["role"], "voice_timbre")
                wiring = [dict(wiring=name, sha256=h3_refiner.digest(asset["path"])) for name, asset in
                          zip(("ref_image_1", "ref_video_1", "ref_video_audio_1", "ref_audio_1"), assets)]
                return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                    {"prompt": "enhanced", "mode": "ref2va", "frames": 124, "wiring": wiring}))

            def extract(path, **kwargs):
                class Output:
                    def run(self, **kwargs):
                        with open(path, "wb") as file:
                            file.write(b"soundtrack")
                return Output()

            stream = mock.Mock()
            stream.output.side_effect = extract
            with mock.patch.object(h3_refiner.ffmpeg, "input", return_value=stream), \
                 mock.patch.object(h3_refiner.subprocess, "run", side_effect=invoke):
                result = refiner.compile("scene", 5, 1280, 736, "balanced", [paths[0]],
                                         videos=[{"path": paths[1], "role": "motion", "seconds": 2, "sound": True}],
                                         audios=[{"path": paths[2], "role": "voice", "seconds": 2}])
            self.assertEqual(result, "enhanced")

    def test_before_review_saves_each_prompt_and_skips_completed(self):
        state = RunState.new("brief", 10, "image", 1280, 736, image_count=0)
        state.h3_mode, state.h3_creativity = "review", "extreme"
        state.shots = [{"prompt": "first", "seconds": 5}, {"prompt": "second", "seconds": 5}]
        refiner = mock.MagicMock()
        refiner.__enter__.return_value = refiner
        refiner.compile.side_effect = ["first enhanced", "second enhanced"]
        with mock.patch.object(main.requests, "get", side_effect=main.requests.ConnectionError), \
             mock.patch.object(main.h3_refiner, "Refiner", return_value=refiner), \
             mock.patch.object(main, "ComfyClient", return_value=SimpleNamespace(workflow={})), \
             mock.patch.object(state, "save") as save:
            main.refine_review(state)
            main.refine_review(state)
        self.assertEqual([s["prompt"] for s in state.shots], ["first enhanced", "second enhanced"])
        self.assertEqual(refiner.compile.call_count, 2)
        self.assertEqual(save.call_count, 2)
        self.assertEqual(refiner.compile.call_args.args[4], "extreme")

    def test_refined_director_frames_use_h3_grid(self):
        from pipelines.video import comfy
        client = object.__new__(comfy.ComfyClient)
        client.workflow = {"1": {"class_type": "value", "inputs": {"fps": 24, "frames": 0}}}
        client.fields = {"fps": ("1", "fps"), "frames": ("1", "frames")}
        with mock.patch.object(client, "_fill_images"), mock.patch.object(client, "_fill_media"), \
             mock.patch.object(client, "_run", return_value={}) as run, mock.patch.object(client, "_download"):
            client.render("x", [], 5, 640, 384, "out", h3_refined=True)
        self.assertEqual(run.call_args.args[0]["1"]["inputs"]["frames"], 124)

    def test_refinement_cache_tracks_prompt_settings_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "plate.png")
            with open(path, "wb") as file:
                file.write(b"one")
            state = RunState.new("brief", 5, "image", 640, 384, image_count=0)
            shot = {"prompt": "hello", "seconds": 5, "seed": 1}
            old = main.refinement_key(state, shot, [path])
            state.h3_creativity = "bold"
            self.assertNotEqual(old, main.refinement_key(state, shot, [path]))
            state.h3_creativity = "balanced"
            with open(path, "wb") as file:
                file.write(b"two")
            self.assertNotEqual(old, main.refinement_key(state, shot, [path]))

    def test_shorts_option_is_saved_and_limited_to_h3(self):
        form = {"voice": "", "avatar_clips": "auto", "workflow": "h3.json",
                "h3_enhance": "render", "h3_creativity": "extreme"}
        entry = {"name": "h3.json", "path": "h3.json", "takes": {}}
        with mock.patch.object(shorts, "saved_workflows", return_value=[entry]), \
             mock.patch.object(shorts, "workflow_model", return_value="minimax_h3"):
            _, env, error = shorts.build_run(form)
        self.assertIsNone(error)
        self.assertEqual(env["H3IR_ENHANCE"], "render")
        self.assertEqual(env["H3IR_CREATIVITY"], "extreme")
        with mock.patch.object(shorts, "saved_workflows", return_value=[entry]), \
             mock.patch.object(shorts, "workflow_model", return_value=None):
            self.assertIn("MiniMax H3", shorts.build_run(form)[2])

    def test_busy_comfy_is_not_unloaded(self):
        refiner = object.__new__(h3_refiner.Refiner)
        refiner.comfy_url = "http://127.0.0.1:8188"
        refiner.process = None
        response = mock.Mock()
        response.json.return_value = {"queue_running": [[1]], "queue_pending": []}
        with mock.patch.object(h3_refiner.requests, "get", return_value=response), \
             mock.patch.object(h3_refiner.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "running or queued"):
                refiner.__enter__()
        post.assert_not_called()

    def test_shorts_enhanced_render_removes_all_extra_reference_links(self):
        avatar = object.__new__(AvatarGenerator)
        avatar.h3_mode, avatar.url, avatar.client_id = "render", "http://comfy", "test"
        avatar.width, avatar.height = 640, 384
        avatar.workflow = {
            "1": {"class_type": "value", "inputs": {"prompt": "", "fps": 24, "frames": 0}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "one"}},
            "3": {"class_type": "LoadImage", "inputs": {"image": "two"}},
            "4": {"class_type": "refs", "inputs": {"first": ["2", 0], "extra": ["3", 0]}},
            "5": {"class_type": "refs", "inputs": {"extra": ["3", 0]}},
        }
        avatar.fields = {"prompt": ("1", "prompt"), "fps": ("1", "fps"), "frames": ("1", "frames"),
                         "image": ("2", "image"), "image_link": ("4", "first"),
                         "image_2": ("3", "image"), "image_2_link": ("4", "extra"),
                         "image_2_also": [("5", "extra")]}
        response = mock.Mock(status_code=200)
        response.json.return_value = {"prompt_id": "id"}
        with mock.patch("pipelines.shorts.avatar_generator.requests.post", return_value=response) as post, \
             mock.patch.object(avatar, "_wait_for", return_value=None):
            avatar._render("brief", "uploaded.png", None, 5, "clip")
        graph = post.call_args.kwargs["json"]["prompt"]
        self.assertEqual(graph["1"]["inputs"]["frames"], 124)
        self.assertEqual(graph["2"]["inputs"]["image"], "uploaded.png")
        self.assertNotIn("3", graph)
        self.assertNotIn("extra", graph["4"]["inputs"])
        self.assertNotIn("extra", graph["5"]["inputs"])


if __name__ == "__main__":
    unittest.main()

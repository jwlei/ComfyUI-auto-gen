"""
The Image side: the panel in-process, starting the real pipelines.image.main as a real process,
which talks to a fake ComfyUI and a fake Claude. Nothing costs anything.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import OBJECT_INFO, close, edit_image_workflow, fake_claude, fake_comfy, text_to_image_workflow, versions_reply

from PIL import Image

from panel import app as _all_pipelines  # noqa: F401
from panel import server as webapp
from pipelines.image import media, paths, planner
from pipelines.image.run_eta import RunEta
from pipelines.image.run_state import RunState, list_runs
from pipelines.video import paths as video_paths


def png_bytes(size=(800, 1200), color=(10, 200, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class SizeTest(unittest.TestCase):
    def test_an_exact_size_wins(self):
        self.assertEqual(media.pick_size("1280x736", "1:1", "1"), (1280, 736))
        self.assertEqual(media.pick_size("1920x1080"), (1920, 1080), "what you asked for, on the model's grid or not")

    def test_a_size_off_the_grid_renders_just_larger(self):
        self.assertEqual(media.render_size(1920, 1080), (1920, 1088))
        self.assertEqual(media.render_size(2560, 1440), (2560, 1440), "already on it")
        self.assertEqual(media.render_size(3840, 2160), (3840, 2160))
        self.assertEqual(media.render_size(1272, 2772), (1280, 2784), "the OnePlus 15")

    def test_an_edit_never_gets_a_size_its_encoder_would_wrap_around(self):
        self.assertEqual(media._reference_latent(1376, 768), (171, 96), "what put a band down the right edge of a real image")
        sizes = [(1376, 768), (800, 1200), (832, 1248)] + [(width, height) for _, width, height in media.DESKTOP_SIZES + media.PHONE_SIZES]
        for width, height in sizes:
            made = media.edit_render_size(width, height)
            self.assertEqual([side % 2 for side in media._reference_latent(*made)], [0, 0], (width, height))
            self.assertGreaterEqual(made, (width, 0), "never narrower, as it's cropped down to the size asked for")
            self.assertGreaterEqual(made[1], height)
            self.assertEqual([side % media.SIZE_STEP for side in made], [0, 0], "and still on the model's grid")
            self.assertLess(made[0] * made[1], width * height * 1.12, "with little to crop")

    def test_a_size_with_nothing_wrong_is_left_as_it_is(self):
        self.assertEqual(media.edit_render_size(1024, 1024), (1024, 1024))
        self.assertEqual(media.edit_render_size(1392, 752), (1392, 752), "what ComfyUI's own template uses")

    def test_the_screens_on_offer(self):
        offered = {(width, height) for _, width, height in media.DESKTOP_SIZES + media.PHONE_SIZES}
        self.assertLessEqual({(1920, 1080), (2560, 1440), (3840, 2160), (1272, 2772)}, offered)
        self.assertTrue(all(height > width for _, width, height in media.PHONE_SIZES), "phones stand upright")

    def test_aspect_and_megapixels_of_your_own(self):
        self.assertEqual(media.pick_size(None, "1:1", "1"), (1024, 1024))
        self.assertEqual(media.pick_size(None, "16:9", "1"), (1360, 768))
        self.assertEqual(media.pick_size(None, "9:16", "2"), (1088, 1936))

    def test_matching_the_image(self):
        source = (800, 1200)
        self.assertEqual(media.pick_size(None, media.FROM_IMAGE, media.FROM_IMAGE, source), (800, 1200), "its own resolution")
        self.assertEqual(media.pick_size(None, media.FROM_IMAGE, "1", source), (832, 1248), "its shape")
        self.assertEqual(media.pick_size(None, "1:1", media.FROM_IMAGE, source), (976, 976), "its pixel count")

    def test_nothing_to_match_means_a_square_megapixel(self):
        self.assertEqual(media.pick_size(None, media.FROM_IMAGE, media.FROM_IMAGE), (1024, 1024))

    def test_a_huge_photo_is_capped(self):
        width, height = media.pick_size(None, media.FROM_IMAGE, media.FROM_IMAGE, (12000, 9000))
        self.assertLessEqual(width * height, media.MAX_MEGAPIXELS * media.PIXELS_PER_MEGAPIXEL * 1.01)

    def test_what_isnt_a_size(self):
        self.assertIsNone(media.parse_size("big"))
        self.assertIsNone(media.parse_size("10x10"), "too small to make")
        self.assertIsNone(media.parse_size("99999x1024"), "too big to make")
        self.assertIsNone(media.parse_aspect("0:9"))
        self.assertIsNone(media.parse_megapixels("99"))


class FitTest(unittest.TestCase):
    def test_the_image_comes_out_exactly_that_size(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "source.png")
            Image.new("RGB", (800, 1200)).save(source)
            with Image.open(media.fit(source, 1360, 768, os.path.join(folder, "fitted.png"))) as fitted:
                self.assertEqual(fitted.size, (1360, 768))


class TrimTest(unittest.TestCase):
    def trimmed(self, made, rendered, wanted):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "image.png")
            image = Image.new("RGB", made, (200, 0, 0))
            # The centre is blue: a crop that's off centre would take some of the red.
            image.paste((0, 0, 200), (0, 4, made[0], made[1] - 4))
            image.save(path)
            media.trim(path, rendered, wanted)
            with Image.open(path) as result:
                self.assertEqual(os.listdir(folder), ["image.png"], "no part left behind")
                return result.size, result.convert("RGB").getpixel((0, 0))

    def test_it_crops_from_the_middle_to_the_exact_size(self):
        self.assertEqual(self.trimmed((320, 288), (320, 288), (320, 280)), ((320, 280), (0, 0, 200)))

    def test_an_image_already_that_size_is_left_alone(self):
        self.assertEqual(self.trimmed((320, 288), (320, 288), (320, 288))[0], (320, 288))

    def test_a_workflow_that_set_its_own_size_is_left_alone(self):
        self.assertEqual(self.trimmed((512, 512), (320, 288), (320, 280))[0], (512, 512))


class PlannerTest(unittest.TestCase):
    def reply(self, count, title="Set", same=False):
        return json.dumps({"title": title, "versions": [{"title": f"V{i}", "prompt": "the same" if same else f"Prompt {i}"} for i in range(count)]})

    def write(self, *replies, count=3):
        with mock.patch.object(planner.llm, "_ask", side_effect=list(replies)) as call:
            return planner.write_prompts([], "a brief", count, "gemini"), call

    def test_one_request_for_the_whole_set(self):
        (title, versions), call = self.write(self.reply(15), count=15)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(title, "Set")
        self.assertEqual([v["prompt"] for v in versions], [f"Prompt {i}" for i in range(15)])

    def test_too_many_are_left_out_rather_than_asked_for_again(self):
        (_, versions), call = self.write(self.reply(5))
        self.assertEqual((len(versions), call.call_count), (3, 1))

    def test_too_few_or_the_same_are_asked_for_again_saying_what_went_wrong(self):
        (_, versions), call = self.write("not json", self.reply(3, same=True), self.reply(3))
        self.assertEqual((len(versions), call.call_count), (3, 3))
        self.assertIn("1 different prompts instead of 3", call.call_args_list[2].args[1])

    def test_what_it_did_write_beats_paying_for_nothing(self):
        (_, versions), call = self.write(self.reply(2), self.reply(1), "not json")
        self.assertEqual((len(versions), call.call_count), (2, 3))

    def test_nothing_usable_says_so(self):
        with self.assertRaisesRegex(RuntimeError, "usable prompts"):
            self.write("no", "no", "no")

    def test_editing_and_making_get_different_instructions(self):
        edit = planner._instructions("different outfits", 15, 1)
        self.assertIn("edit that image", edit)
        self.assertIn("same person", edit)
        self.assertIn("15 different versions", edit)
        self.assertIn("clearly different", edit)
        make = planner._instructions("a lighthouse", 1, 0)
        self.assertIn("each has to say everything", make)
        self.assertNotIn("edit that image", make)
        self.assertNotIn("clearly different", make, "one image has nothing to differ from")
        self.assertIn("Picture 2", planner._instructions("x", 2, 2))


class EtaTest(unittest.TestCase):
    def test_each_workflow_learns_its_own_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "timings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"image:fast.json": 10, "image:slow.json": 200, "Prompts": 30}, f)
            count = ["--count", "3"]
            self.assertEqual(RunEta(count, {"COMFYUI_WORKFLOW": "x/fast.json"}, path).plan()["total"], 30 + 30)
            self.assertEqual(RunEta(count, {"COMFYUI_WORKFLOW": "x/slow.json"}, path).plan()["total"], 30 + 600)
            self.assertEqual(RunEta([*count, "--direct"], {"COMFYUI_WORKFLOW": "x/fast.json"}, path).plan()["total"], 30, "no LLM, no prompts step")
            self.assertEqual(RunEta([*count, "--direct", "--megapixels", "2"], {"COMFYUI_WORKFLOW": "x/fast.json"}, path).plan()["total"], 60, "twice the pixels")
            self.assertEqual(RunEta([*count, "--direct", "--size", "2048x1024"], {"COMFYUI_WORKFLOW": "x/fast.json"}, path).plan()["total"], 60)


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.comfy = fake_comfy(self.temp)
        self.addCleanup(close, self.comfy)
        self.claude = fake_claude()
        self.claude.reply = versions_reply()
        self.addCleanup(close, self.claude)
        patcher = mock.patch.dict(os.environ, {
            "COMFYUI_URL": self.comfy.url, "ANTHROPIC_BASE_URL": self.claude.url, "ANTHROPIC_API_KEY": "test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.comfy.saved = {"image_text.json": text_to_image_workflow(), "image_edit.json": edit_image_workflow()}
        self.comfy.object_info = OBJECT_INFO
        webapp._comfyui_cache["at"] = 0.0
        webapp._runs_cache["at"] = 0.0
        webapp._saved_workflows_cache["at"] = 0.0
        if os.path.exists(webapp.IMAGE_TIMINGS):
            os.remove(webapp.IMAGE_TIMINGS)
        self.client = webapp.app.test_client()
        self.addCleanup(self.clear_everything)

    def clear_everything(self):
        webapp.stop_run()
        self.wait_idle()
        for queue in webapp.QUEUES.values():
            queue.items, queue.held, queue.results = [], False, []
            queue._save()
        for folder in (paths.RUNS_DIR, paths.OUTPUT_DIR, video_paths.INPUTS_DIR):
            shutil.rmtree(folder, ignore_errors=True)

    def wait_idle(self):
        deadline = time.time() + 90
        while time.time() < deadline:
            webapp.run_queue.pending()
            if webapp.job.idle and not webapp.run_queue.current:
                return
            time.sleep(0.05)
        self.fail(f"Timed out. Log: {[l['text'] for l in webapp.job.lines][-15:]}")

    def generate(self, image=False, **fields):
        form = {"prompt": "different outfits", "count": "3", "llm": "claude", "aspect": "1:1", "megapixels": "1", **fields}
        if image:
            form["image"] = (io.BytesIO(png_bytes()), "woman.png")
        return self.client.post("/generate/image", data={k: v for k, v in form.items() if v is not None}, content_type="multipart/form-data")

    def make(self, image=False, **fields):
        reply = self.generate(image, **fields)
        self.assertTrue(reply.get_json().get("ok"), reply.get_json())
        self.wait_idle()
        return reply.get_json()

    def status(self):
        webapp._runs_cache["at"] = 0.0
        return self.client.get("/api/status").get_json()

    def only_run(self):
        runs = list_runs()
        self.assertEqual(len(runs), 1, [l["text"] for l in webapp.job.lines][-15:])
        return runs[0]

    def inputs(self, prompt, kind):
        return next(node["inputs"] for node in prompt["prompt"].values() if node["class_type"] == kind)

    def sent_prompt(self, prompt):
        return self.inputs(prompt, "PrimitiveStringMultiline")["value"] if any(
            node["class_type"] == "PrimitiveStringMultiline" for node in prompt["prompt"].values()) else next(
            node["inputs"]["prompt"] for node in prompt["prompt"].values() if node["class_type"] == "TextEncodeQwenImageEditPlus" and node["inputs"]["prompt"])

    # --- Which kind of run ---

    def test_text_alone_makes_images_from_text(self):
        self.make()
        run = self.only_run()
        self.assertTrue(run.finished, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(os.path.basename(run.workflow), "image_text.json")
        self.assertEqual(run.title, "Three Looks")
        self.assertEqual(self.comfy.uploads, [], "no image to give it")
        self.assertEqual([self.sent_prompt(p) for p in self.comfy.prompts], [f"Change to look number {n}." for n in (1, 2, 3)])
        self.assertTrue(all(block["type"] == "text" for block in self.claude.requests[0]["body"]["messages"][0]["content"]), "the LLM has no image either")
        self.assertIn("each has to say everything", self.claude.requests[0]["body"]["messages"][0]["content"][0]["text"])
        for shot in run.shots:
            self.assertTrue(os.path.exists(run.image_path(shot)))
        self.assertTrue(run.output_dir.startswith(paths.OUTPUT_DIR))
        self.assertEqual(os.path.basename(run.output_dir), f"Three_Looks_{run.id}")

    def test_an_image_edits_it(self):
        self.make(image=True, aspect="image", megapixels="1")
        run = self.only_run()
        self.assertEqual(os.path.basename(run.workflow), "image_edit.json")
        self.assertEqual(self.claude.requests[0]["body"]["messages"][0]["content"][0]["type"], "image", "the LLM sees what it's editing")
        self.assertIn("edit that image", self.claude.requests[0]["body"]["messages"][0]["content"][1]["text"])
        self.assertEqual(len(self.comfy.uploads), 1, "uploaded once, however many are made from it")
        self.assertEqual(self.inputs(self.comfy.prompts[0], "LoadImage")["image"], "uploaded.png")
        self.assertEqual([self.sent_prompt(p) for p in self.comfy.prompts], [f"Change to look number {n}." for n in (1, 2, 3)])
        self.assertEqual(len(self.comfy.prompts), 3)

    def test_nothing_reaches_comfyuis_output_folder(self):
        self.make()
        kinds = [node["class_type"] for node in self.comfy.prompts[0]["prompt"].values()]
        self.assertIn("PreviewImage", kinds)
        self.assertFalse({"SaveImage", "SaveImageAdvanced"} & set(kinds))

    def test_each_image_gets_a_new_seed(self):
        self.make()
        seeds = [self.inputs(p, "KSampler")["seed"] for p in self.comfy.prompts]
        self.assertEqual(len(set(seeds)), 3)

    def test_a_chosen_seed_counts_up_and_shows_in_the_set(self):
        self.assertEqual(self.generate(seed="12x").status_code, 400)
        self.make(seed="100")
        self.assertEqual([self.inputs(p, "KSampler")["seed"] for p in self.comfy.prompts], [100, 101, 102])
        run = self.only_run()
        shots = self.client.get(f"/image-runs/{run.id}").get_json()["shots"]
        self.assertEqual([(shot["seed"], shot["used_seed"]) for shot in shots], [(100, 100), (101, 101), (102, 102)])
        prompts = [shot["prompt"] for shot in shots]
        reply = self.client.post(f"/image-runs/{run.id}/shots", json={"prompts": prompts, "seeds": ["7", None, "x"]})
        self.assertEqual(reply.status_code, 400)
        self.assertTrue(self.client.post(f"/image-runs/{run.id}/shots", json={"prompts": prompts, "seeds": ["7", None, "102"]}).get_json()["ok"])
        self.assertEqual([shot["seed"] for shot in self.client.get(f"/image-runs/{run.id}").get_json()["shots"]], [7, None, 102])

    def test_a_workflow_can_be_chosen(self):
        self.assertEqual(self.generate(workflow="image_edit.json").status_code, 400, "it can't run without an image")
        self.assertEqual(self.generate(image=True, workflow="image_text.json").status_code, 400, "and this one has nowhere to put one")
        self.assertEqual(self.generate(workflow="nowhere.json").status_code, 400)

    # --- Whose prompts ---

    def test_a_direct_prompt_goes_in_word_for_word_with_no_llm(self):
        self.make(prompt="Exactly these words", prompt_mode="direct", count="4", llm=None)
        run = self.only_run()
        self.assertEqual(self.claude.requests, [])
        self.assertEqual([self.sent_prompt(p) for p in self.comfy.prompts], ["Exactly these words"] * 4)
        self.assertEqual(len({self.inputs(p, "KSampler")["seed"] for p in self.comfy.prompts}), 4)
        self.assertTrue(run.direct)
        self.assertEqual(run.title, "Exactly these words")

    def test_review_change_a_prompt_then_make_them(self):
        self.make(review="1")
        status = self.status()
        self.assertEqual(status["returncode"], 0, status["lines"])
        self.assertEqual(self.comfy.prompts, [], "reviewing first makes nothing")
        run = self.only_run()
        self.assertEqual([r["id"] for r in status["stopped_runs"]], [run.id])
        self.assertIn("Prompts ready", status["stopped_runs"][0]["detail"])
        self.assertTrue(status["stopped_runs"][0]["when"], "its date is its own, for the box to give it a line")
        self.assertNotIn(status["stopped_runs"][0]["when"], status["stopped_runs"][0]["detail"])
        detail = self.client.get(f"/image-runs/{run.id}").get_json()
        self.assertTrue(detail["is_image"])
        prompts = [shot["prompt"] for shot in detail["shots"]]
        prompts[1] = "My own wording."
        self.assertTrue(self.client.post(f"/image-runs/{run.id}/shots", json={"prompts": prompts}).get_json()["ok"])
        self.assertEqual(self.client.post(f"/image-runs/{run.id}/shots", json={"prompts": ["too few"]}).status_code, 400)
        self.assertTrue(self.client.post(f"/runs/{run.id}/continue").get_json()["ok"])
        self.wait_idle()
        self.assertEqual(self.sent_prompt(self.comfy.prompts[1]), "My own wording.")
        self.assertTrue(self.only_run().finished)
        self.assertEqual(len(self.claude.requests), 1)

    # --- Size ---

    def test_the_size_asked_for_reaches_the_latent(self):
        self.make(aspect="16:9", megapixels="1", count="1")
        latent = self.inputs(self.comfy.prompts[0], "EmptyLatentImage")
        self.assertEqual((latent["width"], latent["height"]), (1360, 768))

    def test_an_exact_size(self):
        self.make(aspect="custom", size="512x768", count="1")
        latent = self.inputs(self.comfy.prompts[0], "EmptyLatentImage")
        self.assertEqual((latent["width"], latent["height"]), (512, 768))

    def test_a_screen_comes_out_exactly_that_size(self):
        for preset, latent_size in (("1920x1080", (1920, 1088)), ("1272x2772", (1280, 2784))):
            self.comfy.prompts.clear()
            self.make(aspect=preset, count="1", prompt=f"a wallpaper for {preset}")
            latent = self.inputs(self.comfy.prompts[0], "EmptyLatentImage")
            self.assertEqual((latent["width"], latent["height"]), latent_size, "the model gets a size on its grid")
            run = next(run for run in list_runs() if preset in run.brief)
            with Image.open(run.image_path(run.shots[0])) as image:
                self.assertEqual(image.size, tuple(int(side) for side in preset.split("x")), "and what you get is the screen's")

    def test_an_edit_for_a_phone_comes_out_exactly_that_size(self):
        self.make(image=True, aspect="1272x2772", count="1")
        self.assertEqual(self.uploaded_size(), media.edit_render_size(1272, 2772))
        self.assertEqual(self.finished_size(), (1272, 2772))

    def test_only_the_edit_encoder_gets_the_special_size(self):
        self.make(aspect="1376x768", count="1")
        latent = self.inputs(self.comfy.prompts[0], "EmptyLatentImage")
        self.assertEqual((latent["width"], latent["height"]), (1376, 768), "Krea has no such encoder")

    def uploaded_size(self):
        return Image.open(io.BytesIO(self.comfy.uploads[0])).size

    def finished_size(self):
        run = self.only_run()
        with Image.open(run.image_path(run.shots[0])) as image:
            return image.size

    def test_an_edit_can_match_the_images_resolution(self):
        self.make(image=True, aspect="image", megapixels="image", count="1")
        self.assertEqual(self.uploaded_size(), media.edit_render_size(800, 1200), "a workflow that follows its source comes out this size")
        self.assertEqual(self.finished_size(), (800, 1200), "and what's over is cropped off")

    def test_an_edit_can_match_its_shape_at_a_pixel_count(self):
        self.make(image=True, aspect="image", megapixels="1", count="1")
        self.assertEqual(self.finished_size(), (832, 1248))

    def test_an_edit_can_be_given_a_size_of_its_own(self):
        self.make(image=True, aspect="16:9", megapixels="1", count="1")
        self.assertEqual(self.uploaded_size(), (1360, 768), "nothing wrong with this one")
        self.assertEqual(self.finished_size(), (1360, 768))
        self.assertEqual(self.only_run().image_count, 1)
        with Image.open(self.only_run().input_paths[0]) as source:
            self.assertEqual(source.size, (800, 1200), "the original is kept whole")

    def test_bad_options_are_refused(self):
        for fields in ({"count": "0"}, {"count": "101"}, {"count": "many"}, {"aspect": "wide"}, {"megapixels": "99"}, {"aspect": "custom", "size": "big"},
                       {"prompt_mode": "other"}, {"llm": "nobody"}, {"prompt": " "}):
            self.assertEqual(self.generate(**fields).status_code, 400, fields)
        self.assertEqual(webapp.run_queue.pending(), [])

    # --- Never making anything twice ---

    def test_generate_continues_a_stopped_set_without_new_prompts(self):
        self.comfy.script = ["error", "error"]
        self.make()
        run = self.only_run()
        self.assertEqual(self.status()["returncode"], 1)
        self.assertEqual(len(run.done), 2, "one that won't come out doesn't cost the rest")
        self.assertFalse(run.finished)
        self.assertEqual(len(self.comfy.prompts), 4)

        reply = self.make()
        self.assertEqual(reply["continued"], "Three Looks")
        run = self.only_run()
        self.assertTrue(run.finished)
        self.assertEqual(len(self.claude.requests), 1, "the prompts weren't paid for again")
        self.assertEqual(len(self.comfy.prompts), 5, "and only the missing one was made")

    def test_a_different_set_gets_its_own_prompts(self):
        self.make(review="1")
        self.make(prompt="something else", review="1")
        self.assertEqual(len(list_runs()), 2)
        self.assertEqual(len(self.claude.requests), 2)

    def test_a_bad_file_is_made_again(self):
        self.comfy.script = ["corrupt"]
        self.make(count="1")
        self.assertTrue(self.only_run().finished)
        self.assertEqual(len(self.comfy.prompts), 2)

    # --- The gallery ---

    def test_images_can_be_seen_opened_and_downloaded(self):
        self.make()
        status = self.status()
        run = self.only_run()
        self.assertEqual(status["stopped_runs"], [], "a finished set isn't waiting for anything")
        self.assertEqual([s["id"] for s in status["image_sets"]], [run.id])
        self.assertEqual(status["image_sets"][0]["size"], "1024x1024")
        self.assertEqual([image["position"] for image in status["image_sets"][0]["images"]], [1, 2, 3])
        self.assertEqual(self.client.get(f"/image-runs/{run.id}/files/1").status_code, 200)
        self.assertIn("attachment", self.client.get(f"/image-runs/{run.id}/files/1?download=1").headers["Content-Disposition"])
        thumbnail = self.client.get(f"/image-runs/{run.id}/thumbnails/1")
        self.assertEqual(thumbnail.mimetype, "image/jpeg")
        self.assertEqual(self.client.get(f"/image-runs/{run.id}/files/9").status_code, 404)
        self.assertEqual(self.client.get("/image-runs/image_19990101_000000/files/1").status_code, 404)
        self.assertEqual(self.client.get("/image-runs/..%2f..%2fetc/files/1").status_code, 404)

    def test_one_image_made_again_leaves_the_others(self):
        self.make()
        run = self.only_run()
        before = {shot["file"]: os.path.getmtime(run.image_path(shot)) for shot in run.shots[1:]}
        self.assertEqual(self.client.post(f"/image-runs/{run.id}/redo/9").status_code, 400)
        self.assertTrue(self.client.post(f"/image-runs/{run.id}/redo/1").get_json()["ok"])
        self.wait_idle()
        run = self.only_run()
        self.assertTrue(run.finished)
        self.assertEqual(len(self.comfy.prompts), 4)
        self.assertEqual({shot["file"]: os.path.getmtime(run.image_path(shot)) for shot in run.shots[1:]}, before)

    def test_deleting_an_image_keeps_it_from_coming_back(self):
        self.make()
        run = self.only_run()
        gone = run.image_path(run.shots[1])
        self.assertTrue(self.client.post(f"/image-runs/{run.id}/delete/2").get_json()["ok"])
        self.assertFalse(os.path.exists(gone))
        run = self.only_run()
        self.assertEqual((len(run.shots), run.finished), (2, True), "it isn't owed to the set any more")
        self.assertEqual(self.status()["stopped_runs"], [])
        self.assertEqual(self.client.get(f"/image-runs/{run.id}/files/2").status_code, 200, "the next one moved up")

    def test_deleting_the_last_image_takes_the_set(self):
        self.make(count="1")
        run = self.only_run()
        self.client.post(f"/image-runs/{run.id}/delete/1")
        self.assertEqual(list_runs(), [])
        self.assertFalse(os.path.exists(run.output_dir))

    def test_deleting_a_set_takes_its_images(self):
        self.make()
        run = self.only_run()
        self.assertTrue(self.client.post(f"/runs/{run.id}/discard").get_json()["ok"])
        self.assertEqual(list_runs(), [])
        self.assertFalse(os.path.exists(run.output_dir))
        self.assertEqual(self.status()["image_sets"], [])

    def test_a_missing_file_is_owed_again(self):
        self.make(count="2")
        run = self.only_run()
        os.remove(run.image_path(run.shots[0]))
        self.assertFalse(RunState.load(run.id).finished)

    # --- The form's side of it ---

    def test_the_list_has_only_images_and_says_why_one_cant_run(self):
        broken = edit_image_workflow()
        broken["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"] = ["gone.safetensors"]
        self.comfy.saved = {"image_text.json": text_to_image_workflow(), "image_edit.json": broken, "film.json": {"nodes": []}}
        listed = self.client.get("/api/workflows?pipeline=image").get_json()["workflows"]
        self.assertEqual(listed, [
            {"name": "image_edit.json", "label": "image_edit", "problem": "Needs gone.safetensors, which isn't installed."},
            {"name": "image_text.json", "label": "image_text", "takes": {"min_images": 0, "max_images": 0, "size": True}}])
        self.assertEqual(self.client.get("/api/workflows?pipeline=video").get_json()["workflows"], [], "they don't turn up among the films")
        reply = self.generate(image=True)
        self.assertEqual(reply.status_code, 400)
        self.assertIn("gone.safetensors", reply.get_json()["error"], "Automatic says why it has nothing to edit with")
        self.assertEqual(self.generate(workflow="image_edit.json", image=True).status_code, 400)

    def test_the_estimate_starts_nothing_and_follows_the_options(self):
        one = self.client.post("/api/estimate/image", data={"count": "1", "image_count": "0"}).get_json()
        many = self.client.post("/api/estimate/image", data={"count": "15", "image_count": "1"}).get_json()
        self.assertEqual(one["images"], 1)
        self.assertEqual(many["images"], 15)
        self.assertGreater(many["total"], one["total"])
        self.assertFalse(self.client.post("/api/estimate/image", data={"count": "0"}).get_json()["ok"])
        self.assertEqual(self.comfy.prompts, [])

    def test_a_setup_check_makes_nothing(self):
        self.assertTrue(self.client.post("/check/image", data={"prompt_mode": "direct", "workflow": "image_text.json"}).get_json()["ok"])
        self.wait_idle()
        self.assertEqual(webapp.job.returncode, 0, [l["text"] for l in webapp.job.lines][-15:])
        self.assertEqual(self.comfy.prompts, [])

    def test_progress_is_read_from_the_log(self):
        self.make(count="2")
        status = self.status()
        self.assertEqual(status["pipeline"], "image")
        self.assertEqual(status["progress"]["run_id"], self.only_run().id)
        self.assertEqual(status["progress"]["stage"], [2, 2, "Images"])
        self.assertEqual(status["progress"]["detail"], "Image 2 of 2: Look 2")

    def test_the_image_page_opens(self):
        page = self.client.get("/image")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="formImage"', page.data)
        self.assertIn(b'"image"', page.data)


if __name__ == "__main__":
    unittest.main()

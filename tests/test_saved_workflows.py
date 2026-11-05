"""Turning what ComfyUI saves into what its API takes, and finding the inputs to fill."""
import os
import tempfile
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fakes import (OBJECT_INFO, edit_image_workflow, fibo2_workflow, h3_workflow, reference_workflow, saved_workflow, switchable_workflow,
                   text_only_workflow, text_to_image_workflow)

from core import saved_workflows, workflow_options
from fakes import _ui_node
from core import seeds
from core.saved_workflows import _installed_as
from core.saved_workflows import (Unusable, _from_comfyui, _grow_references, _from_earlier, _prepare, find_fields, find_image_fields, frames_for,
                                  image_model, image_takes, is_for_image, is_for_shorts, pipeline_of, takes, to_api)


def node_of(ui, node_id):
    return next(node for node in ui["nodes"] if node["id"] == node_id)


class WhichPipelineTest(unittest.TestCase):
    def test_the_suffix_decides(self):
        self.assertTrue(is_for_shorts("x_ytautoshort.json"))
        self.assertFalse(is_for_shorts("x.json"))
        self.assertFalse(is_for_shorts("ytautoshort_x.json"))

    def test_the_prefix_makes_it_an_image_workflow(self):
        self.assertEqual(pipeline_of("image_krea2_turbo_t2i_int8.json"), "image")
        self.assertEqual(pipeline_of("mine/image_edit.json"), "image", "in a folder too")
        self.assertEqual(pipeline_of("video_image_to_video.json"), "video")
        self.assertEqual(pipeline_of("image_x_ytautoshort.json"), "shorts", "tuned for Shorts is tuned for Shorts")
        self.assertFalse(is_for_image("my_image_edit.json"))

    def test_one_that_saves_images_is_for_image_whatever_its_name(self):
        self.assertEqual(pipeline_of("Fibo2_turbo_edit_test.json", fibo2_workflow()), "image")
        self.assertEqual(pipeline_of("Fibo2_turbo_edit_test.json"), "video", "by name alone it would be a film")
        self.assertEqual(pipeline_of("film.json", saved_workflow()), "video", "it saves a video")
        self.assertEqual(pipeline_of("x_ytautoshort.json", fibo2_workflow()), "shorts")

    def test_shorts_needs_load_audio_and_video_does_not(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            for stem, fields in (("film", {"prompt": "1.text"}), ("silent_ytautoshort", {"prompt": "1.text"}),
                                 ("talks_ytautoshort", {"prompt": "1.text", "audio": "2.audio"})):
                with open(os.path.join(folder, stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump({}, f)
                with open(os.path.join(folder, stem + ".map.json"), "w", encoding="utf-8") as f:
                    json.dump({"_saved_as": stem + ".json", **fields}, f)
            self.assertEqual(sorted(entry["label"] for entry in _from_earlier(folder)), ["film", "talks_ytautoshort"])


class ConvertTest(unittest.TestCase):
    def test_only_what_the_video_needs_reaches_the_api(self):
        api = to_api(saved_workflow(), OBJECT_INFO)
        self.assertEqual(sorted(api, key=int), ["1", "2", "3", "4", "5", "6", "7", "11"])
        self.assertNotIn("8", api, "a preview would save files of its own")

    def test_the_editors_parts_are_resolved(self):
        sampler = to_api(saved_workflow(), OBJECT_INFO)["6"]["inputs"]
        self.assertEqual(sampler["positive"], ["2", 0], "through Set and Get")
        self.assertEqual(sampler["model"], ["1", 0], "through the bypassed LoRA")
        self.assertEqual((sampler["seed"], sampler["steps"]), (123, 20), "the seed's dropdown isn't an input")

    def test_a_bypassed_image_is_switched_on(self):
        api = to_api(saved_workflow(), OBJECT_INFO)
        self.assertEqual(api["5"]["inputs"]["start_image"], ["4", 0])
        self.assertEqual(api["4"]["inputs"]["image"], "example.png")

    def test_savevideo_becomes_one_that_stays_out_of_the_output_folder(self):
        save = to_api(saved_workflow(), OBJECT_INFO)["11"]
        self.assertEqual(save["class_type"], "VHS_VideoCombine")
        self.assertIs(save["inputs"]["save_output"], False)
        self.assertEqual(save["inputs"]["images"], ["7", 0], "CreateVideo's own input")
        self.assertEqual(save["inputs"]["frame_rate"], 24)

    def test_what_wont_run_says_why(self):
        missing_model = saved_workflow()
        node_of(missing_model, 1)["widgets_values"] = ["gone.safetensors"]
        with self.assertRaisesRegex(Unusable, "gone.safetensors"):
            to_api(missing_model, OBJECT_INFO)

        missing_node = saved_workflow()
        node_of(missing_node, 7)["type"] = "SomeoneElsesNode"
        with self.assertRaisesRegex(Unusable, "SomeoneElsesNode"):
            to_api(missing_node, OBJECT_INFO)

        looping = {"nodes": [{"id": 1, "type": "loop", "inputs": []}], "links": [], "definitions": {"subgraphs": [
            {"id": "loop", "inputs": [], "outputs": [], "links": [], "nodes": [{"id": 1, "type": "loop", "inputs": []}]}]}}
        with self.assertRaisesRegex(Unusable, "contain each other"):
            to_api(looping, OBJECT_INFO)

        no_combine = {kind: spec for kind, spec in OBJECT_INFO.items() if kind != "VHS_VideoCombine"}
        with self.assertRaisesRegex(Unusable, "VideoHelperSuite"):
            to_api(saved_workflow(), no_combine)


class SubgraphTest(unittest.TestCase):
    def test_a_subgraph_opens_as_if_unpacked(self):
        api = to_api(text_to_image_workflow(), OBJECT_INFO, images=True)
        by_class = {node["class_type"]: node for node in api.values()}
        self.assertEqual(by_class["KSampler"]["inputs"]["seed"], 99, "what's set on its face wins over what's inside")
        self.assertEqual(by_class["KSampler"]["inputs"]["steps"], 8, "what isn't on its face stays as saved inside")
        self.assertEqual(by_class["PrimitiveStringMultiline"]["inputs"]["value"], "saved prompt")
        self.assertEqual(by_class["EmptyLatentImage"]["inputs"]["width"], ["49", 0], "wired to a node outside it")
        self.assertEqual(by_class["PreviewImage"]["inputs"]["images"][0], next(k for k, n in api.items() if n["class_type"] == "VAEDecode"), "its output goes on to the node outside")

    def test_a_video_workflow_in_a_subgraph_opens_too(self):
        ui = saved_workflow()
        inner = [node for node in ui["nodes"] if node["id"] in (1, 2, 3)]
        ui["nodes"] = [node for node in ui["nodes"] if node["id"] not in (1, 2, 3)] + [
            {"id": 99, "type": "sub", "inputs": [], "outputs": [{"type": "MODEL"}, {"type": "CONDITIONING"}, {"type": "CONDITIONING"}], "widgets_values": []}]
        # The clip links now run inside it. What leaves it comes from its outputs.
        ui["links"] = [link for link in ui["links"] if link[0] not in (2, 7)]
        for link in ui["links"]:
            if link[1] in (1, 2, 3):
                link[1], link[2] = 99, {1: 0, 2: 1, 3: 2}[link[1]]
        ui["definitions"] = {"subgraphs": [{"id": "sub", "inputs": [], "outputs": [{"name": "m"}, {"name": "p"}, {"name": "n"}], "nodes": inner, "links": [
            {"id": 2, "origin_id": 1, "origin_slot": 1, "target_id": 2, "target_slot": 0, "type": "CLIP"},
            {"id": 7, "origin_id": 1, "origin_slot": 1, "target_id": 3, "target_slot": 0, "type": "CLIP"},
            {"id": 50, "origin_id": 1, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "MODEL"},
            {"id": 51, "origin_id": 2, "origin_slot": 0, "target_id": -20, "target_slot": 1, "type": "CONDITIONING"},
            {"id": 52, "origin_id": 3, "origin_slot": 0, "target_id": -20, "target_slot": 2, "type": "CONDITIONING"}]}]}
        api = to_api(ui, OBJECT_INFO)
        self.assertEqual(api[find_fields(api, OBJECT_INFO)["prompt"][0]]["inputs"]["text"], "saved positive")
        self.assertEqual(api["6"]["inputs"]["negative"][0], next(k for k, n in api.items() if n["inputs"].get("text") == "saved negative"))


class ImageWorkflowTest(unittest.TestCase):
    def test_text_to_image(self):
        api = to_api(text_to_image_workflow(), OBJECT_INFO, images=True)
        kinds = [node["class_type"] for node in api.values()]
        fields = find_image_fields(api, OBJECT_INFO)
        self.assertEqual(api[fields["prompt"][0]]["inputs"][fields["prompt"][1]], "saved prompt")
        self.assertEqual(api[fields["width"][0]]["class_type"], "EmptyLatentImage")
        self.assertEqual(image_takes(fields), {"min_images": 0, "max_images": 0, "size": True, "model": None})
        self.assertNotIn("SaveImage", kinds, "it would write to ComfyUI's output folder")
        self.assertIn("PreviewImage", kinds)

    def test_a_switch_goes_the_way_its_set(self):
        kinds = [node["class_type"] for node in to_api(text_to_image_workflow(), OBJECT_INFO, images=True).values()]
        self.assertNotIn("ComfySwitchNode", kinds)
        self.assertNotIn("LoraLoaderModelOnly", kinds, "a style LoRA that's off needs no file")
        self.assertNotIn("TextGenerate", kinds, "and nothing rewrites the prompt, though refining was on")

    def test_image_edit(self):
        api = to_api(edit_image_workflow(), OBJECT_INFO, images=True)
        fields = find_image_fields(api, OBJECT_INFO)
        self.assertEqual(api[fields["prompt"][0]]["inputs"]["prompt"], "saved instruction", "the positive one, not the negative")
        self.assertEqual(image_takes(fields, image_model(api)), {"min_images": 1, "max_images": 1, "size": True, "model": "qwen_image_edit"},
                         "its size follows the image, which it can't run without")
        self.assertNotIn("FluxKontextImageScale", [node["class_type"] for node in api.values()], "or the size asked for would be snapped away")
        encoders = [node for node in api.values() if node["class_type"] == "TextEncodeQwenImageEditPlus"]
        self.assertTrue(all(node["inputs"]["image1"] == [fields["image"][0], 0] for node in encoders))
        self.assertTrue(all("image2" not in node["inputs"] for node in encoders), "the input nothing is plugged into is left out")
        self.assertIsInstance(next(node for node in api.values() if node["class_type"] == "KSampler")["inputs"]["seed"], int)

    def test_fibo2_edits_up_to_sixteen_or_none(self):
        api = to_api(fibo2_workflow(), OBJECT_INFO, images=True)
        _grow_references(api, OBJECT_INFO)
        fields = find_image_fields(api, OBJECT_INFO)
        self.assertEqual(image_takes(fields, image_model(api)), {"min_images": 0, "max_images": 16, "size": True, "model": "fibo2"},
                         "a required list that can hold none, and the node's own 16 names, not twenty")
        encoder = next(node for node in api.values() if node["class_type"] == "TextEncodeFibo2")
        self.assertEqual(sorted(name for name in encoder["inputs"] if name.startswith("images.")), sorted(f"images.image_{n}" for n in range(1, 17)))
        self.assertEqual((fields["prompt"], fields["width"]), (["4", "prompt"], ["4", "width"]), "its prompt and size are on the encoder")
        self.assertEqual(fields["image_link"], ["4", "images.image_1"], "the bypassed image comes out when it's left empty")

    def test_the_text_encoder_says_which_model(self):
        api = {"1": {"class_type": "CLIPLoader", "inputs": {"type": "krea2"}, "_meta": {"title": "x"}}}
        self.assertEqual(image_model(api), "krea2")
        self.assertIsNone(image_model({"1": {"class_type": "CLIPLoader", "inputs": {"type": "sd3"}, "_meta": {"title": "x"}}}))

    def test_what_wont_run_says_why(self):
        with self.assertRaisesRegex(Unusable, "Save Image"):
            to_api(reference_workflow(), OBJECT_INFO, images=True)
        missing = edit_image_workflow()
        missing["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"] = ["gone.safetensors"]
        with self.assertRaisesRegex(Unusable, "gone.safetensors"):
            to_api(missing, OBJECT_INFO, images=True)


class EditorOnlyTest(unittest.TestCase):
    def api(self, chosen=None):
        return workflow_options.convert(switchable_workflow(), OBJECT_INFO, chosen)

    def test_what_only_the_editor_has_is_left_out(self):
        api = self.api()[0]
        kinds = {node["class_type"] for node in api.values()}
        self.assertFalse(kinds & {"Label (rgthree)", "Bookmark (rgthree)", "Fast Groups Bypasser (rgthree)"})
        self.assertNotIn("Image Saver", kinds, "it writes to ComfyUI's output folder, and Civitai")
        self.assertIn("PreviewImage", kinds)

    def test_anything_everywheres_links_are_made(self):
        api = self.api()[0]
        encoder = next(node for node in api.values() if node["class_type"] == "CLIPTextEncode")
        self.assertEqual(encoder["inputs"]["clip"], ["1", 1])

    def test_a_node_saved_without_its_widgets_names_gets_them_from_comfyui(self):
        enhancer = self.api()[0]["2"]["inputs"]
        self.assertEqual((enhancer["enabled"], enhancer["strength"], enhancer["debug"]), (True, 1.5, False))

    def test_a_workflow_that_only_previews_is_for_images(self):
        ui = text_to_image_workflow()
        ui["nodes"][0] = _ui_node(29, "PreviewImage", links={"images": 44})
        self.assertEqual(pipeline_of("simple.json", ui), "image")
        self.assertIn("PreviewImage", [node["class_type"] for node in to_api(ui, OBJECT_INFO, images=True).values()])


class OptionsTest(unittest.TestCase):
    def options(self, ui=None):
        return {option["id"]: option for option in workflow_options.offered(ui or switchable_workflow(), OBJECT_INFO)}

    def test_each_switch_and_setting_is_an_option(self):
        options = self.options()
        self.assertEqual(sorted(options), ["group_color_correction", "group_sharpen", "group_upscaler", "group_upscaler.choice",
                                           "lora_knpv3_1", "prompt_boost", "prompt_enhancer", "steps"])
        self.assertEqual(options["prompt_enhancer"]["role"], "llm", "chosen with the LLM, not as a switch")
        self.assertNotIn("group_prompt_enhancer", options, "the pipeline writes the prompt: its own enhancer stays off")
        self.assertEqual((options["group_sharpen"]["default"], options["steps"]["default"]), ("0", "8"))
        self.assertEqual(options["group_upscaler.choice"]["choices"], ["720p", "1080p", "4K"])
        self.assertEqual((options["lora_knpv3_1"]["default"], options["prompt_boost"]["default"]), ("0", "1.5"))

    def test_one_that_cant_run_says_why(self):
        self.assertIn("LayerColor", self.options()["group_color_correction"]["problem"])
        self.assertNotIn("problem", self.options()["group_sharpen"])

    def test_a_run_makes_the_workflow_again_as_chosen(self):
        api = workflow_options.convert(switchable_workflow(), OBJECT_INFO, {
            "group_sharpen": "1", "group_upscaler": "1", "group_upscaler.choice": "1080p", "lora_knpv3_1": "0.7", "prompt_boost": "0", "steps": "12"})[0]
        kinds = [node["class_type"] for node in api.values()]
        self.assertEqual(kinds.count("Image Lucy Sharpen"), 2, "the sharpen group, and the one inside the upscaler")
        self.assertEqual(next(node for node in api.values() if node["class_type"] == "CustomCombo")["inputs"]["choice"], "1080p")
        lora = next(node for node in api.values() if node["class_type"] == "Power Lora Loader (rgthree)")["inputs"]["lora_1"]
        self.assertEqual((lora["on"], lora["strength"], lora["lora"]), (True, 0.7, "KNPV3_1.safetensors"), "the file where it's installed")
        self.assertIs(api["2"]["inputs"]["enabled"], False)
        self.assertEqual(next(node for node in api.values() if node["class_type"] == "KSampler")["inputs"]["steps"], 12)

    def test_as_saved_the_switched_off_parts_are_left_out(self):
        kinds = [node["class_type"] for node in workflow_options.convert(switchable_workflow(), OBJECT_INFO)[0].values()]
        self.assertNotIn("Image Lucy Sharpen", kinds)
        self.assertNotIn("CustomCombo", kinds)

    def test_choices_are_checked(self):
        options = list(self.options().values())
        self.assertIsNone(workflow_options.check(options, {"group_sharpen": "1", "steps": "20", "lora_knpv3_1": "0"}))
        self.assertIn("can't be changed", workflow_options.check(options, {"group_color_correction": "1"}))
        self.assertIn("one of", workflow_options.check(options, {"group_upscaler.choice": "8K"}))
        self.assertIn("from 1 to 150", workflow_options.check(options, {"steps": "0"}))
        self.assertIn("no option", workflow_options.check(options, {"made_up": "1"}))

    def test_a_templates_subgraph_switches_are_options(self):
        options = self.options(text_to_image_workflow())
        self.assertEqual(options["face_enable_lora"]["default"], "0")
        self.assertIn("isn't installed", options["face_enable_lora"]["problem"], "its LoRA file isn't there to switch on")
        self.assertNotIn("face_prompt_enhance", options, "kept off")
        ui = text_to_image_workflow()
        ui["nodes"][1]["widgets_values"][4] = "lora.safetensors"
        self.assertNotIn("problem", self.options(ui)["face_enable_lora"])
        api = workflow_options.convert(ui, OBJECT_INFO, {"face_enable_lora": "1"})[0]
        self.assertIn("LoraLoaderModelOnly", [node["class_type"] for node in api.values()], "its switch went the way chosen")


class ShorthandTest(unittest.TestCase):
    def test_a_workflow_by_its_name_without_its_side(self):
        from core.saved_workflows import shorthand
        self.assertEqual(shorthand("logs/workflows/image_krea2_SFW_NSFW.json", "image"), "krea2_SFW_NSFW")
        self.assertEqual(shorthand("video_wan22.json", "ltx"), "wan22")
        self.assertEqual(shorthand("minimax_h3_ytautoshort.json", "comfyui"), "minimax_h3")
        self.assertEqual(shorthand("shorts_ltx_ytautoshort", "comfyui"), "ltx")
        self.assertEqual(shorthand("Fibo2 turbo edit.json", "image"), "Fibo2_turbo_edit")
        self.assertEqual(shorthand(None, "ltx"), "ltx", "the built-in one")

    def test_a_second_with_the_same_name_keeps_the_first(self):
        from core.saved_workflows import title_words, unique_name
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(unique_name(folder, "A_Harbour_ltx", ".mp4"), "A_Harbour_ltx")
            open(os.path.join(folder, "A_Harbour_ltx.mp4"), "w").close()
            self.assertEqual(unique_name(folder, "A_Harbour_ltx", ".mp4"), "A_Harbour_ltx_2")
            os.makedirs(os.path.join(folder, "Looks_krea2"))
            self.assertEqual(unique_name(folder, "Looks_krea2"), "Looks_krea2_2")
        self.assertEqual(title_words("A harbour: at dawn!", "film"), "A_harbour_at_dawn")
        self.assertEqual(title_words("", "film"), "film")


class LorasTest(unittest.TestCase):
    def test_only_the_loras_an_output_reads_from(self):
        api = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors"}},
            "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "styles\Film.safetensors", "strength_model": 0.8}},
            "3": {"class_type": "Power Lora Loader (rgthree)", "inputs": {"model": ["2", 0],
                  "lora_1": {"on": True, "lora": "KNPV3_1.safetensors", "strength": 1},
                  "lora_2": {"on": False, "lora": "Off.safetensors", "strength": 1}}},
            "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
            "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
            # Switched out of the way: still in the workflow, but nothing reads it.
            "9": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Unused.safetensors", "strength_model": 1}},
        }
        self.assertEqual(workflow_options.loras_in(api), ["Film (0.8)", "KNPV3_1"])
        self.assertEqual(workflow_options.loras_used(None), [])


class InstalledAsTest(unittest.TestCase):
    def test_another_precision_of_the_same_model(self):
        installed = ["krea2_turbo_fp8_scaled.safetensors", "krea2_turbo_int8_convrot.safetensors"]
        self.assertEqual(_installed_as("krea2_turbo_fp8.safetensors", installed), "krea2_turbo_fp8_scaled.safetensors")
        self.assertEqual(_installed_as("krea2_turbo.safetensors", ["krea2_turbo_fp8_scaled.safetensors"]), "krea2_turbo_fp8_scaled.safetensors")

    def test_another_folder(self):
        self.assertEqual(_installed_as("Krea2\\KNPV3_1.safetensors", ["KNPV3_1.safetensors"]), "KNPV3_1.safetensors")

    def test_never_a_guess(self):
        self.assertIsNone(_installed_as("krea2_turbo.safetensors", installed := ["krea2_turbo_fp8_scaled.safetensors", "krea2_turbo_bf16.safetensors"]),
                          "two would fit: which is meant can't be told")
        self.assertIsNone(_installed_as("krea2.safetensors", ["krea2_turbo_fp8.safetensors"]), "turbo is another model, not a format")
        self.assertIsNone(_installed_as("model.gguf", ["model.safetensors"]))

    def test_a_workflow_runs_on_the_one_installed(self):
        ui = saved_workflow()
        next(node for node in ui["nodes"] if node["id"] == 1)["widgets_values"] = ["model_fp16.safetensors"]
        info = {**OBJECT_INFO, "CheckpointLoader": {"display_name": "x", "input": {"required": {"ckpt_name": [["model_fp16_scaled.safetensors"]]}}}}
        self.assertEqual(to_api(ui, info)["1"]["inputs"]["ckpt_name"], "model_fp16_scaled.safetensors")


class PromptWriterTest(unittest.TestCase):
    def test_a_templates_refine_switch_is_its_prompt_enhancer(self):
        options = {option["id"]: option for option in workflow_options.offered(text_to_image_workflow(), OBJECT_INFO)}
        self.assertEqual((options["prompt_enhancer"]["role"], options["prompt_enhancer"]["default"]), ("llm", "0"))
        off = workflow_options.convert(text_to_image_workflow(), OBJECT_INFO)[0]
        on = workflow_options.convert(text_to_image_workflow(), OBJECT_INFO, {"prompt_enhancer": "1"})[0]
        self.assertNotIn("TextGenerate", [node["class_type"] for node in off.values()], "off unless chosen")
        self.assertIn("TextGenerate", [node["class_type"] for node in on.values()])

    def test_prompt_groups_are_prompt_writers_on_the_image_side_only(self):
        ui = switchable_workflow()
        ui["groups"].append({"title": "Image to prompt", "color": "#3f789e", "bounding": [850, 200, 200, 150]})
        ui["nodes"].append({**_ui_node(30, "LoadImage", {"image": "a.png"}, values=["a.png", "image"], mode=4), "pos": [900, 250]})
        ui["nodes"].append({**_ui_node(31, "Fast Groups Bypasser (rgthree)"), "pos": [0, 1150], "properties": {"matchTitle": "Image to prompt"}})
        image = {option["id"]: option for option in workflow_options.describe(ui, OBJECT_INFO, "image")}
        self.assertEqual({key for key, option in image.items() if option.get("role")}, {"prompt_enhancer", "image_to_prompt"})
        self.assertEqual(image["image_to_prompt"]["label"], "Image to prompt")
        video = workflow_options.describe(ui, OBJECT_INFO, "video")
        self.assertFalse([option for option in video if option.get("role")], "the LLM always writes a film's prompts")
        turned_on = workflow_options._before(ui, list(image.values()), {"image_to_prompt": "1"})
        self.assertEqual(next(node for node in turned_on["nodes"] if node["id"] == 30)["mode"], 0)
        turned_off = workflow_options._before(ui, list(image.values()), {})
        self.assertEqual(next(node for node in turned_off["nodes"] if node["id"] == 16)["mode"], 4, "an enhancer stays off unless chosen")

    def test_each_image_gets_its_own_prompt_from_the_enhancer(self):
        workflow = {"1": {"class_type": "TextGenerate", "inputs": {"sampling_mode.seed": 0, "seed": 1}},
                    "2": {"class_type": "KSampler", "inputs": {"noise_seed": 5, "denoise": True}}}
        seeds.apply(workflow, 42)
        self.assertEqual((workflow["1"]["inputs"]["sampling_mode.seed"], workflow["1"]["inputs"]["seed"], workflow["2"]["inputs"]["noise_seed"]), (42, 42, 42))
        self.assertIs(workflow["2"]["inputs"]["denoise"], True)


class PrepareTest(unittest.TestCase):
    def test_a_reference_limit_is_written_into_the_map(self):
        """MiniMax H3's limit is a number among the map's inputs, which once made listing it fail."""
        import json
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(saved_workflows.REFERENCE_FILE_LIMITS, {"ReferenceToVideo": 12}):
            path, found = _prepare("r2v.json", reference_workflow(), OBJECT_INFO, folder, "video")
            with open(path[:-len(".json")] + ".map.json", encoding="utf-8") as f:
                written = json.load(f)
        self.assertEqual((written["_most_references"], written["_pipeline"]), (12, "video"))
        self.assertEqual(found["max_images"], 20)


class FieldsTest(unittest.TestCase):
    def fields(self, ui=None):
        return find_fields(to_api(ui or saved_workflow(), OBJECT_INFO), OBJECT_INFO)

    def test_finds_the_inputs_from_the_graph(self):
        self.assertEqual(self.fields(), {
            "prompt": ["2", "text"], "image": ["4", "image"], "image_link": ["5", "start_image"], "frames": ["5", "length"], "fps": ["11", "frame_rate"],
            "width": ["5", "width"], "height": ["5", "height"]})

    def test_the_negative_prompt_is_never_taken_for_the_prompt(self):
        api = to_api(saved_workflow(), OBJECT_INFO)
        # A node between the sampler and the text that takes both, as LTX's conditioning does.
        api["60"] = {"class_type": "Conditioning", "_meta": {"title": "Conditioning"},
                     "inputs": {"negative": ["3", 0], "positive": ["2", 0]}}
        api["6"]["inputs"]["positive"] = ["60", 0]
        self.assertEqual(find_fields(api, OBJECT_INFO)["prompt"], ["2", "text"])
        del api["60"], api["6"]["inputs"]["positive"]
        with self.assertRaisesRegex(Unusable, "prompt"):
            find_fields(api, OBJECT_INFO)

    def test_titled_nodes_win_over_the_latents_inputs(self):
        api = to_api(saved_workflow(), OBJECT_INFO)
        for node_id, title in (("40", "LENGTH (in seconds)"), ("41", "WIDTH"), ("42", "HEIGHT")):
            api[node_id] = {"class_type": "INTConstant", "inputs": {"value": 8}, "_meta": {"title": title}}
        fields = find_fields(api, OBJECT_INFO)
        self.assertEqual((fields["seconds"], fields["width"], fields["height"]),
                         (["40", "value"], ["41", "value"], ["42", "value"]))
        self.assertNotIn("frames", fields)

    def test_each_kind_of_workflow_takes_what_it_can(self):
        ti2v = self.fields()
        self.assertEqual(takes(ti2v), {"min_images": 0, "max_images": 1}, "its image can be unplugged")

        t2v = self.fields(text_only_workflow())
        self.assertEqual(takes(t2v), {"min_images": 0, "max_images": 0})

        i2v = dict(ti2v)
        del i2v["image_link"]
        self.assertEqual(takes(i2v), {"min_images": 1, "max_images": 1}, "it can't run without one")

        both = {**i2v, "text_to_video": ["9", "value"]}
        self.assertEqual(takes(both), {"min_images": 0, "max_images": 1}, "its switch makes it text to video")

    def test_a_reference_workflow(self):
        fields = self.fields(reference_workflow())
        self.assertEqual(takes(fields), {"min_images": 0, "max_images": 2})
        self.assertEqual((fields["image"], fields["image_2"]), (["2", "image"], ["1", "image"]),
                         "in the order the node numbers them, not the order they were added")
        self.assertEqual(fields["image_link"], ["6", "ref_images.ref_image_0"])
        self.assertEqual(fields["prompt"], ["3", "value"], "no sampler's positive input to follow here")
        self.assertEqual(fields["seconds"], ["4", "value"], "titled duration, and a decimal")
        self.assertEqual((fields["width"], fields["height"]), (["6", "width"], ["6", "height"]), "replaces what fed them")

    def test_a_reference_node_that_grows_gets_twenty(self):
        api = to_api(reference_workflow(), OBJECT_INFO)
        _grow_references(api, OBJECT_INFO)
        fields = find_fields(api, OBJECT_INFO)
        self.assertEqual(takes(fields), {"min_images": 0, "max_images": 20}, "two wired, eighteen added")
        self.assertEqual([fields[f"image_{n}_link"][1] for n in range(3, 21)],
                         [f"ref_images.ref_image_{n}" for n in range(2, 20)], "numbered on from the last one, 10 after 9")
        info = {**OBJECT_INFO, "ReferenceToVideo": {**OBJECT_INFO["ReferenceToVideo"], "input": {
            **OBJECT_INFO["ReferenceToVideo"]["input"], "optional": {"ref_images": ["COMFY_AUTOGROW_V3", {"template": {"max": 3}}]}}}}
        api = to_api(reference_workflow(), info)
        _grow_references(api, info)
        self.assertEqual(takes(find_fields(api, info))["max_images"], 3, "never past the node's own limit")
        api = to_api(saved_workflow(), OBJECT_INFO)
        before = dict(api)
        _grow_references(api, OBJECT_INFO)
        self.assertEqual(api, before, "a workflow without one is left alone")

    def test_minimax_h3_takes_nine_images_and_three_videos_or_audio_clips(self):
        api = to_api(h3_workflow(), OBJECT_INFO)
        _grow_references(api, OBJECT_INFO)
        fields = find_fields(api, OBJECT_INFO)
        self.assertEqual(takes(fields), {"min_images": 0, "max_images": 9, "max_videos": 3, "max_audios": 3, "max_references": 12})
        h3 = api["6"]["inputs"]
        for number in range(3):
            loader, _ = h3[f"ref_audios.ref_audio_{number}"]
            self.assertEqual(api[loader]["class_type"], "LoadAudio")
            split, _ = h3[f"ref_videos.ref_video_{number}"]
            self.assertEqual(h3[f"ref_video_audios.ref_video_audio_{number}"], [split, 1], "a video's own sound goes with it")
            self.assertEqual(api[api[split]["inputs"]["video"][0]]["class_type"], "LoadVideo")
        self.assertEqual(fields["ref_video_2_link"], ["6", "ref_videos.ref_video_1"])
        self.assertEqual(fields["ref_video_3_sound"], ["6", "ref_video_audios.ref_video_audio_2"])
        self.assertEqual(fields["ref_audio"][1], "audio")
        self.assertNotIn("audio", fields, "reference audio isn't the film's soundtrack")

    def test_reference_videos_wired_in_comfyui_are_kept(self):
        api = to_api(h3_workflow(), OBJECT_INFO)
        api["90"] = {"class_type": "LoadVideo", "inputs": {"file": "mine.mp4"}, "_meta": {"title": "Load Video"}}
        api["91"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["90", 0]}, "_meta": {"title": "Split"}}
        api["6"]["inputs"]["ref_videos.ref_video_0"] = ["91", 0]
        _grow_references(api, OBJECT_INFO)
        fields = find_fields(api, OBJECT_INFO)
        self.assertEqual(fields["ref_video"], ["90", "file"])
        self.assertEqual(takes(fields)["max_videos"], 1, "nothing added beside the one wired")

    def test_frames_stay_on_the_models_grid(self):
        self.assertEqual(frames_for(5, 24), 121)
        self.assertEqual(frames_for(8, 24), 193)
        self.assertEqual(frames_for(1, 16) % 4, 1)


class QwenImage21TemplateTest(unittest.TestCase):
    """How ComfyUI's Qwen-Image 2.1 templates are saved: by a newer editor, which lists fewer widgets than it saves."""

    ENCODER = {"display_name": "Text Encode Qwen Image 2.1", "input_order": {"required": ["clip", "prompt", "negative_prompt", "resolution", "images"]},
               "input": {"required": {"clip": ["CLIP"], "prompt": ["STRING", {}], "negative_prompt": ["STRING", {}],
                                      "resolution": ["INT", {"default": 1024}], "images": ["COMFY_AUTOGROW_V3", {}]}}}
    GENERATE = {"display_name": "Generate Text", "input_order": {"required": ["clip", "prompt", "max_length", "sampling_mode"],
                                                                 "optional": ["thinking", "mtp", "system_prompt"]},
                "input": {"required": {"clip": ["CLIP"], "prompt": ["STRING", {}], "max_length": ["INT", {}],
                                       "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                                           {"key": "on", "inputs": {"required": {"temperature": ["FLOAT", {}], "seed": ["INT", {}]},
                                                                    "optional": {"presence_penalty": ["FLOAT", {}]}}},
                                           {"key": "off", "inputs": {"required": {}}}]}]},
                          "optional": {"thinking": ["BOOLEAN", {}], "mtp": ["COMBO", {"options": ["auto", "off"]}],
                                       "system_prompt": ["STRING", {"forceInput": True}]}}}

    def test_a_widget_it_didnt_list_still_gets_its_value(self):
        node = _ui_node(1, "TextEncodeQwenImage21", {"prompt": "", "negative_prompt": ""}, values=["a cat", "", 1024])
        self.assertEqual(saved_workflows._widgets_of(node, self.ENCODER), {"prompt": "a cat", "negative_prompt": "", "resolution": 1024})

    def test_a_dropdowns_own_widgets_follow_its_choice(self):
        node = _ui_node(1, "TextGenerate", {"prompt": "", "max_length": 0, "sampling_mode.seed": 0},
                        values=["", 4096, "on", 1.0, 7, 1.5, True, "auto"])
        self.assertEqual(saved_workflows._widgets_of(node, self.GENERATE),
                         {"prompt": "", "max_length": 4096, "sampling_mode": "on", "sampling_mode.temperature": 1.0, "sampling_mode.seed": 7,
                          "sampling_mode.presence_penalty": 1.5, "thinking": True, "mtp": "auto"})
        off = _ui_node(1, "TextGenerate", {"prompt": ""}, values=["", 512, "off", False, "auto"])
        self.assertEqual(saved_workflows._widgets_of(off, self.GENERATE),
                         {"prompt": "", "max_length": 512, "sampling_mode": "off", "thinking": False, "mtp": "auto"})

    def test_a_face_saved_with_a_value_for_every_input_reads_in_the_subgraphs_order(self):
        subgraph = {"inputs": [{"name": "prompt", "type": "STRING"}, {"name": "switch", "type": "BOOLEAN"}, {"name": "width", "type": "INT"},
                               {"name": "scheduler", "type": "COMBO"}, {"name": "images.image_1", "type": "IMAGE"}]}
        face = _ui_node(9, "sg", {"prompt": "", "scheduler": ""}, values=["a cat", False, 1024, "euler"])
        self.assertEqual(saved_workflows._face_widgets(face, subgraph), {"prompt": "a cat", "switch": False, "width": 1024, "scheduler": "euler"})
        listed = _ui_node(9, "sg", {"prompt": "", "scheduler": ""}, values=["a cat", "euler"])
        self.assertEqual(saved_workflows._face_widgets(listed, subgraph), {"prompt": "a cat", "scheduler": "euler"}, "as before, when they match")

    def test_a_text_preview_on_the_way_to_the_model_is_stepped_over(self):
        api = {"1": {"class_type": "PreviewAny", "inputs": {"source": "a cat"}, "_meta": {"title": "Preview Any"}},
               "2": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": ["1", 0]}, "_meta": {"title": "Encode"}}}
        saved_workflows._skip_text_previews(api)
        self.assertEqual(api, {"2": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "a cat"}, "_meta": {"title": "Encode"}}})

    def test_which_qwen_image_21_it_is_follows_whether_it_is_given_an_image(self):
        made = {"1": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "x", "resolution": 1024}}}
        edited = {"1": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "x", "images.image_1": ["2", 0]}}}
        self.assertEqual(image_model(made), "qwen_image_21")
        self.assertEqual(image_model(edited), "qwen_image_21_edit")



class LoraFamilyTest(unittest.TestCase):
    """A LoRA made for an earlier Qwen-Image does nothing on 2.1: told apart by its layers, from its header."""

    def make(self, folder, name, keys):
        import json
        import struct
        header = json.dumps({key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}).encode()
        with open(os.path.join(folder, "models", "loras", name), "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + bytes(2))

    def test_an_earlier_qwen_image_lora_is_refused_on_2_1_only(self):
        from unittest import mock
        from core import lora_check
        folder = tempfile.mkdtemp()
        os.makedirs(os.path.join(folder, "models", "loras"))
        self.make(folder, "old.safetensors", ["transformer_blocks.0.img_mlp.net.0.proj.lora_down.weight"])
        self.make(folder, "new.safetensors", ["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"])
        self.make(folder, "other.safetensors", ["lora_unet_down_blocks_0.weight"])
        with mock.patch.dict(os.environ, {"COMFYUI_DIR": folder, "COMFYUI_LAUNCHER": ""}):
            self.assertIn("earlier Qwen-Image", lora_check.problem("old.safetensors", "qwen_image_21"))
            self.assertIn("earlier Qwen-Image", lora_check.problem("old.safetensors", "qwen_image_21_edit"))
            self.assertIsNone(lora_check.problem("new.safetensors", "qwen_image_21"))
            self.assertIsNone(lora_check.problem("other.safetensors", "qwen_image_21"), "not telling, so not refused")
            self.assertIsNone(lora_check.problem("old.safetensors", "krea2"), "only Qwen-Image 2.1 is checked")
            self.assertIsNone(lora_check.problem("gone.safetensors", "qwen_image_21"))

if __name__ == "__main__":
    unittest.main()

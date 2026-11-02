"""Turning what ComfyUI saves into what its API takes, and finding the inputs to fill."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fakes import OBJECT_INFO, reference_workflow, saved_workflow, text_only_workflow

from core.saved_workflows import Unusable, _from_comfyui, _grow_references, _from_earlier, find_fields, frames_for, is_for_shorts, takes, to_api


def node_of(ui, node_id):
    return next(node for node in ui["nodes"] if node["id"] == node_id)


class WhichPipelineTest(unittest.TestCase):
    def test_the_suffix_decides(self):
        self.assertTrue(is_for_shorts("x_ytautoshort.json"))
        self.assertFalse(is_for_shorts("x.json"))
        self.assertFalse(is_for_shorts("ytautoshort_x.json"))

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

        with self.assertRaisesRegex(Unusable, "subgraphs"):
            to_api({**saved_workflow(), "definitions": {"subgraphs": [{"id": "x"}]}}, OBJECT_INFO)

        no_combine = {kind: spec for kind, spec in OBJECT_INFO.items() if kind != "VHS_VideoCombine"}
        with self.assertRaisesRegex(Unusable, "VideoHelperSuite"):
            to_api(saved_workflow(), no_combine)


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

    def test_frames_stay_on_the_models_grid(self):
        self.assertEqual(frames_for(5, 24), 121)
        self.assertEqual(frames_for(8, 24), 193)
        self.assertEqual(frames_for(1, 16) % 4, 1)


if __name__ == "__main__":
    unittest.main()

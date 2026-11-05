import copy
import unittest

from core import director


def graph():
    api = {"17": {"class_type": director.DIRECTOR, "_meta": {"title": "Director"},
                  "inputs": {"total_frames": 49, "frame_rate": 24, "width": 384, "height": 384, "timeline_data": "{}",
                             "task_type": "t2v", "postprocess_config": "x"}}}
    director.adapt(api)
    return api


class DirectorTest(unittest.TestCase):
    def test_adapt_wires_prompt_and_nine_images(self):
        api = graph()
        self.assertEqual(api["17"]["inputs"]["director_inputs"], [director.INPUTS_ID, 0])
        self.assertEqual(sum(node["class_type"] == "LoadImage" for node in api.values()), 9)
        self.assertEqual(api[director.INPUTS_ID]["inputs"]["ref_assets_1"], [director.ASSETS_ID, 0])
        self.assertEqual(api["17"]["inputs"]["postprocess_config"], "")

    def test_adapt_leaves_one_with_an_inputs_node_alone(self):
        api = graph()
        before = copy.deepcopy(api)
        director.adapt(api)
        self.assertEqual(api, before)

    def test_no_images_is_text_to_video(self):
        api = graph()
        api[director.INPUTS_ID]["inputs"]["ref_prompt_1"] = "a fox"
        for node_id in [str(director.FIRST_ID + n) for n in range(1, 10)]:
            del api[director.ASSETS_ID]["inputs"][f"image_{int(node_id) - director.FIRST_ID}"]
            del api[node_id]
        director.shape(api)
        self.assertEqual(api[director.INPUTS_ID]["inputs"], {"text_prompt_1": "a fox"})
        self.assertNotIn(director.ASSETS_ID, api)
        self.assertTrue(api["17"]["inputs"]["task_type"].startswith("t2v"))

    def test_one_image_is_a_start_frame_and_more_are_references(self):
        api = graph()
        api[director.INPUTS_ID]["inputs"]["ref_prompt_1"] = "a fox"
        for number in range(2, 10):
            del api[director.ASSETS_ID]["inputs"][f"image_{number}"]
        one = copy.deepcopy(api)
        director.shape(one)
        self.assertEqual(one[director.INPUTS_ID]["inputs"], {"image_prompt_1": "a fox", "image_1": [str(director.FIRST_ID + 1), 0]})
        self.assertTrue(one["17"]["inputs"]["task_type"].startswith("i2v"))
        api[director.ASSETS_ID]["inputs"]["image_2"] = [str(director.FIRST_ID + 2), 0]
        director.shape(api)
        self.assertEqual(api[director.INPUTS_ID]["inputs"]["ref_prompt_1"], "a fox")
        self.assertTrue(api["17"]["inputs"]["task_type"].startswith("r2v"))


if __name__ == "__main__":
    unittest.main()

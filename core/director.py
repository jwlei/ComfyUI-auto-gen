"""
MiniMax H3 Motion Director, the one node that makes text, image and reference clips of the H3 model: what it makes
is set by its Task type and by what it is given through its Inputs node (prompts, and images through an Assets node).
A saved workflow has it set up by hand in the editor, on a timeline of its own. The panel gives it one clip's
prompt and images, so a workflow with it gets those wired in when it's converted (adapt), and the right mode chosen
for what a run gives it (shape).
"""
import json

DIRECTOR = "MiniMaxH3MotionDirector"
INPUTS = "MiniMaxH3MotionDirectorInputs"
ASSETS = "MiniMaxH3MotionDirectorAssets"
MOST_IMAGES = 9
# Where the saved timeline's reference images go in an Assets node: Picture 1 to 9.
TASK_TYPES = {
    "t2v": "t2v \u2014 \u6587\u751f\u89c6\u9891(Text to Video)",
    "i2v": "i2v \u2014 \u56fe\u751f\u89c6\u9891(Image to Video)",
    "r2v": "r2v \u2014 \u53c2\u8003\u4e3b\u4f53\u751f\u89c6\u9891(Reference to Video)",
}
FIRST_ID = 9000
ASSETS_ID, INPUTS_ID = "9010", "9011"
STAND_IN = "example.png"
IMAGES_AS = "images_as"


def director_of(api):
    """The id of the one Motion Director that has no Inputs node plugged in, or None."""
    found = [node_id for node_id, node in api.items()
             if node["class_type"] == DIRECTOR and not isinstance(node["inputs"].get("director_inputs"), list)]
    return found[0] if len(found) == 1 else None


def timeline(frames, fps, width, height):
    """A timeline of one clip with nothing in it: the prompt and images come from the Inputs node."""
    seconds = round(frames / fps, 2) if fps else 0
    return json.dumps({
        "version": 5, "editMode": "segment", "totalFrames": frames, "frameRate": fps,
        "video": {"fileName": "", "videoFile": "", "subfolder": "", "type": "input", "frames": [], "frameMap": [],
                  "sourceFrameCount": 0, "deletedSourceRanges": []},
        "videoClips": [],
        "global": {"taskType": "", "prompt": "", "refs": [], "refAudios": [], "refVideos": [], "continuousReference": False,
                   "referenceVideo": {"videoFile": "", "fileName": "", "type": "input", "subfolder": ""}, "genImage": {"imageFile": ""}},
        "output": {"mode": "fixed", "longEdge": max(width, height), "width": width, "height": height, "maxExportFrames": 0,
                   "exportMode": "all", "audioMode": "generate", "continuityEnabled": False, "continuityOverlapFrames": 9},
        "runSelectEnabled": False, "runSelection": [], "liveTaePreview": True,
        "segments": [{"id": "s0", "start": 0, "length": frames, "frameCount": frames, "durationSec": seconds, "prompt": "",
                      "negativePrompt": "", "taskType": "", "refs": [], "refAudios": [], "refVideos": [],
                      "genImage": {"imageFile": "", "fileName": ""},
                      "contextLink": {"schema": "previous_context_link_v1", "enabled": False, "visual": False, "audio": False}}],
    }, ensure_ascii=False)


def adapt(api):
    """
    Gives the director a Load Image for each of nine pictures, an Assets node holding them and an Inputs node holding the
    prompt, set up for a reference clip. A run takes out what it doesn't use, and changes the mode if the images say so.
    """
    director = director_of(api)
    if not director:
        return
    settings = api[director]["inputs"]
    assets = {}
    for number in range(1, MOST_IMAGES + 1):
        node_id = str(FIRST_ID + number)
        api[node_id] = {"class_type": "LoadImage", "inputs": {"image": STAND_IN}, "_meta": {"title": f"Load Image {number}"}}
        assets[f"image_{number}"] = [node_id, 0]
    api[ASSETS_ID] = {"class_type": ASSETS, "inputs": assets, "_meta": {"title": "Director Assets"}}
    api[INPUTS_ID] = {"class_type": INPUTS, "_meta": {"title": "Director Inputs", IMAGES_AS: "start"},
                      "inputs": {"ref_prompt_1": "", "ref_assets_1": [ASSETS_ID, 0]}}
    settings["director_inputs"] = [INPUTS_ID, 0]
    settings["task_type"] = TASK_TYPES["r2v"]
    # Its timeline and its refine pass are the saved clip's own: the first would hold its prompts and images, and the
    # second makes the clip again with models that may not be installed.
    settings["timeline_data"] = timeline(int(settings.get("total_frames") or 0), float(settings.get("frame_rate") or 24),
                                         int(settings.get("width") or 0), int(settings.get("height") or 0))
    settings["postprocess_config"] = ""


def shape(workflow):
    """
    Called once a run's images, size and length are in. No image makes it a text clip, one is a start frame (or a
    reference, if asked) and more are references. What isn't used comes out.
    """
    director = next((node for node in workflow.values() if node["class_type"] == DIRECTOR), None)
    link = director["inputs"].get("director_inputs") if director else None
    if not (isinstance(link, list) and link[0] in workflow and workflow[link[0]]["class_type"] == INPUTS):
        return
    inputs_node = workflow[link[0]]
    inputs = inputs_node["inputs"]
    prompt = inputs.pop("ref_prompt_1", "")
    assets_link = inputs.pop("ref_assets_1", None)
    assets_id = assets_link[0] if isinstance(assets_link, list) else None
    held = workflow[assets_id]["inputs"] if assets_id in workflow else {}
    images = [held[name] for name in sorted(held, key=lambda name: int(name.split("_")[1])) if isinstance(held[name], list)]
    if not images:
        mode = "t2v"
        inputs["text_prompt_1"] = prompt
    elif len(images) == 1 and inputs_node["_meta"].get(IMAGES_AS, "start") == "start":
        mode = "i2v"
        inputs["image_prompt_1"], inputs["image_1"] = prompt, images[0]
    else:
        mode = "r2v"
        inputs["ref_prompt_1"], inputs["ref_assets_1"] = prompt, assets_link
    if mode != "r2v":
        workflow.pop(assets_id, None)
    settings = director["inputs"]
    settings["task_type"] = TASK_TYPES[mode]
    settings["timeline_data"] = timeline(int(settings["total_frames"]), float(settings["frame_rate"]),
                                         int(settings["width"]), int(settings["height"]))

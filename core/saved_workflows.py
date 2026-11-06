"""
The workflows saved in ComfyUI, made into what the pipeline runs: an API-format
workflow and the '.map.json' next to it.

ComfyUI saves what its editor shows, and only its web page turns that into what
its API takes. So this does the same for the parts that exist only in the editor:
subgraphs, Set/Get nodes, reroutes, primitives, and bypassed or muted nodes. Which
inputs to fill is worked out from the graph. Anything that won't run says why, rather
than failing an hour into a queue.
"""
import copy
import itertools
import json
import os
import re
from urllib.parse import quote

import requests

from core import director

# Only in the editor: nothing of these reaches ComfyUI's API.
NOTES = ("Note", "MarkdownNote")
# rgthree's labels, bookmarks and group switches: the editor's alone. core/workflow_options.py offers what the switches do.
EDITOR_ONLY = ("Label (rgthree)", "Bookmark (rgthree)", "Fast Groups Bypasser (rgthree)", "Fast Groups Muter (rgthree)",
               "Fast Bypasser (rgthree)", "Fast Muter (rgthree)")
PASS_THROUGH = ("SetNode", "GetNode", "Reroute")
BYPASSED, MUTED = 4, 2
SEED_CONTROLS = ("fixed", "increment", "decrement", "randomize")
# Latent video comes in 4n+1 frames.
FRAME_STEP = 4
PROMPT_INPUTS = ("text", "prompt", "value")
# Tuned for Shorts, and all Shorts offers. Everything else saved in ComfyUI is for Video.
SHORTS_SUFFIX = "_shorts"
# SaveVideo always writes to ComfyUI's output folder, which the pipeline stays out of.
COMBINE = "VHS_VideoCombine"
COMBINE_DEFAULTS = {"loop_count": 0, "filename_prefix": "pipeline", "format": "video/h264-mp4", "pix_fmt": "yuv420p",
                    "crf": 19, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": False}
# Saved as the Image side's: ComfyUI's own image templates are named this way, and its image-to-video ones video_.
IMAGE_PREFIX = "image_"
# These always write to ComfyUI's output folder. Preview Image does the same job in its temp folder.
IMAGE_SAVES = ("SaveImage", "SaveImageAdvanced", "Image Saver")
PREVIEW = "PreviewImage"
SWITCHES = ("ComfySwitchNode",)
CUSTOM_COMBO = "CustomCombo"
# Show the text that passes through them, as ComfyUI's Qwen-Image 2.1 templates show the prompt on its way to the model.
TEXT_PREVIEWS = ("PreviewAny",)
# Snap the source image to a size of their own. The pipeline fits it to the size you chose instead.
IMAGE_SCALERS = ("FluxKontextImageScale",)
# What a model file's name can differ by and still be that model: a precision or format after it, such as
# krea2_turbo_fp8 installed as krea2_turbo_fp8_scaled. Its folder can differ too.
FORMAT_SUFFIX = re.compile(r"(?:[_.-](?:scaled|fp8|fp16|bf16|fp32|int8|convrot|e4m3fn|e5m2|mixed|nvfp4|awq|q\d\w*))+", re.I)
# A prompt the node feeding the sampler is told to follow, rather than the one to fill: a prompt writer's instructions.
SYSTEM_PROMPT = re.compile(r"system", re.I)
# What saves a video, so a workflow that saves images and none of these is for the Image side, whatever its name.
VIDEO_SAVES = ("SaveVideo", "VHS_VideoCombine", "CreateVideo", "SaveWEBM", "SaveAnimatedWEBP", "SaveAnimatedPNG")
# The image models whose prompts or limits are their own (pipelines/image/models.py), by the node that says which it is,
# or else by the type its text encoder is loaded as.
IMAGE_MODEL_NODES = {"TextEncodeFibo2": "fibo2", "TextEncodeQwenImageEditPlus": "qwen_image_edit"}
IMAGE_MODEL_ENCODERS = ("fibo2", "krea2")
# Qwen-Image 2.1's encoder, for making images and editing them: which it does depends on whether it's given an image.
QWEN_IMAGE_21_ENCODER = "TextEncodeQwenImage21"
# How a subgraph's links name the two ends it has in place of nodes.
SUBGRAPH_IN, SUBGRAPH_OUT = -10, -20
# Far more than any workflow has. Only subgraphs that contain each other get this far.
MOST_SUBGRAPHS = 200


class Unusable(Exception):
    """Why the pipeline can't run this workflow, in words for the panel."""


def _input_spec(spec):
    inputs = spec.get("input", {})
    return {**inputs.get("required", {}), **inputs.get("optional", {})}


def _options(config):
    """A dropdown's choices, or None if the input isn't one, or takes files the pipeline fills in."""
    extra = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
    if any(key in extra for key in ("image_upload", "audio_upload", "video_upload")):
        return None
    return config[0] if isinstance(config[0], list) else extra.get("options") if config[0] == "COMBO" else None


def _link_row(link):
    """A link as [id, origin node, origin slot, target node, target slot, type]. Subgraphs save theirs as objects."""
    if isinstance(link, dict):
        return [link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"], link["type"]]
    return list(link)


# The input types a node shows as a widget, rather than a socket to plug into.
WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "BDGROUP")  # BDGROUP: a section header some nodes save a text for
# A dropdown whose choice brings widgets of its own, saved after it as name.widget, as Generate Text's sampling mode.
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"


def _is_widget(config):
    kind = config[0]
    forced = len(config) > 1 and isinstance(config[1], dict) and config[1].get("forceInput")
    return (isinstance(kind, list) or kind in WIDGET_TYPES or kind == DYNAMIC_COMBO) and not forced


def _spec_inputs(spec):
    inputs = spec.get("input", {})
    return {**inputs.get("required", {}), **inputs.get("optional", {})}


def _chosen_widgets(config, choice):
    """The widgets a dynamic dropdown's choice brings with it, in the order they're saved."""
    for option in config[1].get("options", []) if len(config) > 1 and isinstance(config[1], dict) else []:
        if isinstance(option, dict) and option.get("key") == choice:
            inputs = option.get("inputs", {})
            return {**inputs.get("required", {}), **inputs.get("optional", {})}
    return {}


def _widget_names(node, spec):
    """
    The names of a node's widgets, in the order their values are saved. The editor lists them among the node's inputs;
    a node saved before it did has only its values, so the names come from ComfyUI's own order for its inputs. So do
    they when it lists fewer than it saved values for, as ComfyUI's Qwen-Image 2.1 template does with its encoder's
    resolution: listed are only those it can plug into.
    """
    names = [entry["widget"]["name"] for entry in node.get("inputs", []) if "widget" in entry]
    if not spec:
        return names
    every = _spec_inputs(spec)
    order = [*spec.get("input_order", {}).get("required", []), *spec.get("input_order", {}).get("optional", [])] or list(every)
    widgets = [name for name in order if name in every and _is_widget(every[name])]
    if not names:
        return widgets
    values = node.get("widgets_values")
    seeds = sum(1 for value in values if value in SEED_CONTROLS) if isinstance(values, list) else 0
    if isinstance(values, list) and len(values) - seeds > len(names) and {name.split(".")[0] for name in names} <= set(widgets):
        return widgets
    return names


def _widgets_of(node, spec=None):
    """What a node's widgets are set to, by name, from the list they're saved in. spec: what ComfyUI says about the node."""
    values = node.get("widgets_values")
    if isinstance(values, dict):
        return {name: value for name, value in values.items() if name != "videopreview"}
    values = list(values or [])
    every = _spec_inputs(spec) if spec else {}
    names = _widget_names(node, spec)
    seeds = sum(1 for value in values if value in SEED_CONTROLS)
    if None in values and len(values) - seeds > len(names):
        # The editor saves a null for each button or label a node's script adds, which ComfyUI has no input for.
        values = [value for value in values if value is not None]
    inputs = {}
    position = 0
    waiting = list(names)
    while waiting:
        name = waiting.pop(0)
        if position >= len(values):
            break
        inputs[name] = values[position]
        position += 1
        # The dropdown saved after a seed, which only the editor reads.
        if position < len(values) and values[position] in SEED_CONTROLS and isinstance(inputs[name], int):
            position += 1
        config = every.get(name)
        if config and config[0] == DYNAMIC_COMBO:
            brought = [f"{name}.{widget}" for widget, sub in _chosen_widgets(config, inputs[name]).items() if _is_widget(sub)]
            waiting = brought + [other for other in waiting if other not in brought]
    # Widgets a node's own script added, such as Power Lora Loader's list.
    loras = [value for value in values[position:] if isinstance(value, dict) and "lora" in value]
    inputs.update({f"lora_{number}": lora for number, lora in enumerate(loras, 1)})
    return inputs


def face_names(instance, subgraph):
    """
    The names of a subgraph face's saved values, in order. Its values follow the face's widgets; a newer editor saves
    one for each of the subgraph's inputs a value can be typed into, in the subgraph's order, while listing only some
    on the face, as ComfyUI's Qwen-Image 2.1 templates are saved.
    """
    values = instance.get("widgets_values")
    typed = [entry["name"] for entry in subgraph.get("inputs", []) if entry.get("type") in WIDGET_TYPES]
    listed = [entry["widget"]["name"] for entry in instance.get("inputs", []) if "widget" in entry]
    return typed if isinstance(values, list) and len(values) == len(typed) > len(listed) else listed


def _face_widgets(instance, subgraph):
    """What a subgraph's face is set to, by name."""
    values = instance.get("widgets_values")
    if isinstance(values, dict):
        return _widgets_of(instance)
    names = face_names(instance, subgraph)
    if names != [entry["widget"]["name"] for entry in instance.get("inputs", []) if "widget" in entry]:
        return dict(zip(names, values))
    return _widgets_of(instance)


def _open_subgraph(instance, subgraph, nodes, links, node_ids, link_ids):
    """Puts one subgraph's nodes where it stood, each wired to what the subgraph's own inputs and outputs were."""
    renumbered = {node["id"]: next(node_ids) for node in subgraph["nodes"]}
    inner = copy.deepcopy(subgraph["nodes"])
    entries = {entry["link"]: entry for node in inner for entry in node.get("inputs", []) if entry.get("link") is not None}
    for entry in entries.values():
        entry["link"] = None
    for node in inner:
        node["id"] = renumbered[node["id"]]
        if instance.get("mode"):
            node["mode"] = instance["mode"]
    nodes += inner

    outer_inputs = {entry["name"]: entry for entry in instance.get("inputs", [])}
    widgets = _face_widgets(instance, subgraph)
    outputs = {}
    for link_id, origin, slot, target, target_slot, kind in map(_link_row, subgraph.get("links", [])):
        if target == SUBGRAPH_OUT:
            outputs[target_slot] = (renumbered.get(origin, origin), slot)
            continue
        if origin == SUBGRAPH_IN:
            name = subgraph["inputs"][slot]["name"]
            outer = links.get(outer_inputs.get(name, {}).get("link"))
            if outer:
                origin, slot = outer[1], outer[2]
            elif name in widgets:
                # Set on the subgraph's face, which is what counts over the value saved inside it.
                origin, slot = next(node_ids), 0
                nodes.append({"id": origin, "type": "PrimitiveNode", "widgets_values": [widgets[name]]})
            else:
                continue  # Nothing plugged in: as if the input inside had nothing either
        else:
            origin = renumbered[origin]
        new_id = next(link_ids)
        links[new_id] = [new_id, origin, slot, renumbered[target], target_slot, kind]
        if link_id in entries:
            entries[link_id]["link"] = new_id

    for link_id, link in list(links.items()):
        if link[1] == instance["id"]:
            origin, slot = outputs.get(link[2], (None, None))
            if origin == SUBGRAPH_IN:
                outer = links.get(outer_inputs.get(subgraph["inputs"][slot]["name"], {}).get("link"))
                origin, slot = (outer[1], outer[2]) if outer else (None, None)
            if origin is None:
                del links[link_id]
            else:
                link[1], link[2] = origin, slot


def _with_hidden_links(ui):
    """
    Anything Everywhere's links, which the editor makes as it queues a workflow: from one node to every input of its
    type left unplugged. It saves what it made in the workflow's extra, and that's what is followed here.
    """
    hidden = ui.get("extra", {}).get("ue_links") or []
    if not hidden:
        return ui
    ui = {**ui, "nodes": copy.deepcopy(ui["nodes"])}
    nodes = {node["id"]: node for node in ui["nodes"]}
    links = [_link_row(link) for link in ui.get("links", [])]
    link_ids = itertools.count(max([link[0] for link in links] + [ui.get("last_link_id") or 0]) + 1)
    for found in hidden:
        node = nodes.get(int(found["downstream"]))
        inputs = node.get("inputs", []) if node else []
        slot = found["downstream_slot"]
        if slot < len(inputs) and inputs[slot].get("link") is None:
            link_id = next(link_ids)
            links.append([link_id, int(found["upstream"]), found["upstream_slot"], node["id"], slot, found.get("type")])
            inputs[slot]["link"] = link_id
    return {**ui, "links": links}


def _without_subgraphs(ui):
    """The workflow as if every subgraph in it had been unpacked in ComfyUI. Returns a new one."""
    definitions = {subgraph["id"]: subgraph for subgraph in ui.get("definitions", {}).get("subgraphs", [])}
    nodes = list(ui["nodes"])
    if not any(node["type"] in definitions for node in nodes):
        return ui
    links = {row[0]: row for row in map(_link_row, ui.get("links", []))}
    node_ids = itertools.count(max([node["id"] for node in nodes] + [ui.get("last_node_id") or 0]) + 1)
    link_ids = itertools.count(max([*links, ui.get("last_link_id") or 0]) + 1)
    for _ in range(MOST_SUBGRAPHS):
        instance = next((node for node in nodes if node["type"] in definitions), None)
        if not instance:
            return {**ui, "nodes": nodes, "links": list(links.values())}
        nodes.remove(instance)
        _open_subgraph(instance, definitions[instance["type"]], nodes, links, node_ids, link_ids)
    raise Unusable("Its subgraphs contain each other. Unpack them in ComfyUI and save it again.")


class _Graph:
    def __init__(self, ui, info):
        self.info = info
        self.links = {row[0]: row for row in map(_link_row, ui.get("links", []))}
        self.nodes = {node["id"]: node for node in ui["nodes"]}
        self.setters = {(node.get("widgets_values") or [None])[0]: node for node in ui["nodes"] if node["type"] == "SetNode"}

    def skipped(self, node):
        # The form offers every image input, so a bypassed one still counts: left empty, it comes out again.
        return node.get("mode") in (BYPASSED, MUTED) and node["type"] != "LoadImage"

    def source(self, link_id, seen=()):
        """What really feeds this link: [node id, output slot], {'value': a primitive's}, or None."""
        link = self.links.get(link_id)
        origin = self.nodes.get(link[1]) if link else None
        if not origin or origin["id"] in seen:
            return None
        seen = (*seen, origin["id"])
        kind = origin["type"]
        if kind == "PrimitiveNode":
            return {"value": origin["widgets_values"][0]}
        if kind == "GetNode":
            setter = self.setters.get((origin.get("widgets_values") or [None])[0])
            inputs = setter.get("inputs", []) if setter else []
            return self.source(inputs[0].get("link"), seen) if inputs else None
        if kind in ("SetNode", "Reroute"):
            inputs = origin.get("inputs", [])
            return self.source(inputs[0].get("link"), seen) if inputs else None
        if self.skipped(origin):
            return self._through_bypass(origin, link[2], seen) if origin["mode"] == BYPASSED else None
        return [str(origin["id"]), link[2]]

    def _through_bypass(self, node, slot, seen):
        """A bypassed node hands on the input of its output's type: the one in the same position if it can."""
        outputs, inputs = node.get("outputs", []), node.get("inputs", [])
        wanted = outputs[slot].get("type") if slot < len(outputs) else None
        matching = [i for i, entry in enumerate(inputs) if entry.get("type") == wanted and entry.get("link") is not None]
        if not matching:
            return None
        return self.source(inputs[slot if slot in matching else matching[0]].get("link"), seen)

    def inputs_of(self, node):
        inputs = _widgets_of(node, self.info.get(node["type"]))
        for entry in node.get("inputs", []):
            if entry.get("link") is not None:
                source = self.source(entry["link"])
                if isinstance(source, dict):
                    inputs[entry["name"]] = source["value"]
                elif source:
                    inputs[entry["name"]] = source
        return inputs


def _node(class_type, inputs, title):
    return {"inputs": inputs, "class_type": class_type, "_meta": {"title": title}}


def _keep_out_of_output_folder(api, info):
    """Swaps SaveVideo + CreateVideo for the combine node, which can save to ComfyUI's temp folder instead."""
    for node in api.values():
        if node["class_type"] != "SaveVideo":
            continue
        made_by = node["inputs"].get("video")
        maker = api.get(made_by[0]) if isinstance(made_by, list) else None
        if not maker or maker["class_type"] != "CreateVideo":
            raise Unusable("Its SaveVideo isn't fed by CreateVideo, so its clips can't stay out of ComfyUI's output folder.")
        if COMBINE not in info:
            raise Unusable("It uses SaveVideo. Install ComfyUI-VideoHelperSuite to keep its clips out of ComfyUI's output folder.")
        inputs = {**COMBINE_DEFAULTS, "images": maker["inputs"]["images"], "frame_rate": maker["inputs"]["fps"]}
        if "audio" in maker["inputs"]:
            inputs["audio"] = maker["inputs"]["audio"]
        node.update(_node(COMBINE, inputs, "Video Combine"))


def _save_images_to_temp(api, info):
    """
    Swaps each node that saves images for Preview Image, which keeps them out of ComfyUI's output folder. Returns those
    nodes, or, in a workflow that saves nothing, its Preview Images: they're in ComfyUI's temp folder already.
    """
    saves = [node_id for node_id, node in api.items() if node["class_type"] in IMAGE_SAVES]
    if not saves:
        return [node_id for node_id, node in api.items() if node["class_type"] == PREVIEW]
    if saves and PREVIEW not in info:
        raise Unusable("ComfyUI has no Preview Image node to keep its images out of its output folder.")
    for node_id in saves:
        api[node_id] = _node(PREVIEW, {"images": api[node_id]["inputs"].get("images")}, "Preview Image")
    return saves


def _replace_uses(api, node_id, slot, source):
    """Whatever took this output takes source instead, or nothing if there is none."""
    for node in api.values():
        for name, value in list(node["inputs"].items()):
            if value == [node_id, slot]:
                if source is None:
                    del node["inputs"][name]
                else:
                    node["inputs"][name] = source


def _settle_switches(api, enhance=False):
    """
    A switch set on the workflow's face goes the way it's set, for good. So a style LoRA that's off needs
    no file, and with Refine Prompt off nothing rewrites the prompt: the LLM's or yours reaches the model as it is.
    enhance turns it on instead, for the workflow's own enhancer to write the prompt from yours.
    """
    for node in api.values():
        if re.search(r"refine|enhance", node["_meta"]["title"], re.I) and isinstance(node["inputs"].get("value"), bool):
            node["inputs"]["value"] = enhance
    while True:
        for node_id, node in api.items():
            if node["class_type"] not in SWITCHES:
                continue
            switch = node["inputs"].get("switch")
            setter = api.get(switch[0]) if isinstance(switch, list) else None
            value = setter["inputs"].get("value") if setter else switch
            if isinstance(value, bool):
                _replace_uses(api, node_id, 0, node["inputs"].get("on_true" if value else "on_false"))
                del api[node_id]
                break
        else:
            return


def _skip_text_previews(api):
    """A Preview Any between a prompt and what reads it only shows it in the editor: what reads it takes the prompt itself."""
    for node_id, node in list(api.items()):
        if node["class_type"] in TEXT_PREVIEWS and "source" in node["inputs"]:
            _replace_uses(api, node_id, 0, node["inputs"]["source"])
            del api[node_id]


def _skip_image_scalers(api):
    for node_id, node in list(api.items()):
        if node["class_type"] in IMAGE_SCALERS:
            _replace_uses(api, node_id, 0, node["inputs"].get("image"))
            del api[node_id]


def _upstream(api, roots):
    """Only what the video needs: ComfyUI wouldn't run the rest, and a preview would save files of its own."""
    kept, waiting = set(), list(roots)
    while waiting:
        node_id = waiting.pop()
        if node_id in kept or node_id not in api:
            continue
        kept.add(node_id)
        waiting += [value[0] for value in api[node_id]["inputs"].values() if isinstance(value, list) and len(value) == 2]
    return {node_id: api[node_id] for node_id in api if node_id in kept}


def _installed_as(value, options):
    """The installed file a missing one is, in another folder or another precision. None unless just one fits."""
    if not isinstance(value, str) or not value.strip():
        return None
    parts = lambda name: os.path.splitext(os.path.basename(name.replace("\\", "/")))
    stem, extension = parts(value)
    same, formats = [], []
    for option in options:
        if not isinstance(option, str):
            continue
        other, other_extension = parts(option)
        if other_extension.lower() != extension.lower() or not other.lower().startswith(stem.lower()):
            continue
        rest = other[len(stem):]
        if not rest:
            same.append(option)
        elif FORMAT_SUFFIX.fullmatch(rest):
            formats.append(option)
    found = same or formats
    return found[0] if len(found) == 1 else None


def _check_installed(api, info):
    for node in api.values():
        spec = _input_spec(info[node["class_type"]])
        for name, config in spec.items():
            value = node["inputs"].get(name)
            options = _options(config)
            if options and value is not None and not isinstance(value, list) and value not in options:
                installed = _installed_as(value, options)
                if not installed:
                    raise Unusable(f"Needs {value}, which isn't installed.")
                node["inputs"][name] = installed
        for name, config in info[node["class_type"]].get("input", {}).get("required", {}).items():
            # A growing list of inputs is saved as name.a, name.b, and can be empty.
            if name not in node["inputs"] and config[0] != "COMFY_AUTOGROW_V3":
                raise Unusable(f"{node['_meta']['title']} has nothing set for {name}.")


def _set_combo_index(inputs, node):
    """
    A Custom Combo's INDEX: ComfyUI's editor works it out from where the choice is in the list and sends it as
    "index", and the node only passes that on. Without it the index is 0, the first choice, whatever is chosen: the
    SeedVR2 upscaler set to 4K ran at 720p. The saved file keeps the choice, an index (-1), then the list.
    """
    options = [value for value in (node.get("widgets_values") or [])[2:] if isinstance(value, str) and value]
    if inputs.get("choice") in options:
        inputs["index"] = options.index(inputs["choice"])


def to_api(ui, info, images=False, enhance=False):
    """
    The saved workflow as ComfyUI's API takes it. images: it makes images rather than a video. enhance: its own
    prompt enhancer, if it has one, is on. Raises Unusable.
    """
    ui = _without_subgraphs(_with_hidden_links(ui))
    graph = _Graph(ui, info)
    api = {}
    for node in ui["nodes"]:
        kind = node["type"]
        if kind in NOTES or kind in EDITOR_ONLY or kind in PASS_THROUGH or kind == "PrimitiveNode" or graph.skipped(node):
            continue
        if kind not in info:
            raise Unusable(f"Needs the node {kind}, which isn't installed.")
        title = node.get("title") or info[kind].get("display_name") or kind
        api[str(node["id"])] = _node(kind, graph.inputs_of(node), title)
        if kind == CUSTOM_COMBO:
            _set_combo_index(api[str(node["id"])]["inputs"], node)

    if images:
        saves = _save_images_to_temp(api, info)
        if not saves:
            raise Unusable("It doesn't save an image the pipeline can pick up: it needs a Save Image node.")
        _settle_switches(api, enhance)
        _skip_text_previews(api)
        _skip_image_scalers(api)
    else:
        _keep_out_of_output_folder(api, info)
        saves = [node_id for node_id, node in api.items() if "save_output" in node["inputs"]]
        if not saves:
            raise Unusable("It doesn't make a video the pipeline can pick up.")
    api = _upstream(api, saves)
    _check_installed(api, info)
    director.adapt(api)
    return api


def _titled(api, pattern, kinds):
    """The node with this title holding one value of these kinds, as [node id, input]."""
    for node_id, node in api.items():
        if re.search(pattern, node["_meta"]["title"], re.I):
            values = [name for name, value in node["inputs"].items() if type(value) in kinds]
            if len(values) == 1:
                return [node_id, values[0]]
    return None


def _prompt_input(api):
    """Walks up from what the sampler treats as positive, or a node that takes a prompt itself, to the text it started as."""
    if director.INPUTS_ID in api and api[director.INPUTS_ID]["class_type"] == director.INPUTS:
        return [director.INPUTS_ID, "ref_prompt_1"]
    waiting = [value[0] for node in api.values() for name, value in node["inputs"].items()
               if name == "positive" and isinstance(value, list)]
    if not waiting:
        for node_id, node in api.items():
            prompt = node["inputs"].get("prompt")
            if isinstance(prompt, str):
                return [node_id, "prompt"]
            if isinstance(prompt, list):
                waiting.append(prompt[0])
    seen = set()
    while waiting:
        node_id = waiting.pop(0)
        if node_id in seen or node_id not in api:
            continue
        seen.add(node_id)
        inputs = api[node_id]["inputs"]
        for name in PROMPT_INPUTS if not SYSTEM_PROMPT.search(api[node_id]["_meta"]["title"]) else ():
            if isinstance(inputs.get(name), str):
                return [node_id, name]
        waiting += [value[0] for name, value in inputs.items() if isinstance(value, list) and name != "negative"]
    return None


def _one_input(api, class_type, name, what, besides=()):
    found = [node_id for node_id, node in api.items() if node["class_type"] == class_type and node_id not in besides]
    if len(found) > 1:
        raise Unusable(f"It takes {len(found)} {what} inputs, and the pipeline only has one to give.")
    return [found[0], name] if found else None


def _in_number_order(name):
    """ref_image_10 comes after ref_image_9, and not after ref_image_1."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _image_fields(api, info):
    """
    image, image_2... for each Load Image, in the order the node using them numbers
    them. image_link is where one is plugged in, if the workflow also runs without it.
    """
    found = []
    for node_id, node in api.items():
        if node["class_type"] != "LoadImage":
            continue
        users = [(user_id, name) for user_id, user in api.items() for name, value in user["inputs"].items()
                 if isinstance(value, list) and value[0] == node_id]
        found.append((users[0][1] if users else "", int(node_id), users))
    fields = {}
    for number, (_, node_id, users) in enumerate(sorted(found, key=lambda entry: (_in_number_order(entry[0]), entry[1])), 1):
        slot = "image" if number == 1 else f"image_{number}"
        fields[slot] = [str(node_id), "image"]
        # One left empty comes out of everything it's plugged into, as Qwen-Image 2.1's go to its encoder and to what its
        # prompt enhancer looks at: only if each of them runs without it.
        if users and all(api[user_id]["class_type"] == director.ASSETS or
                         _can_be_left_empty(info[api[user_id]["class_type"]], name) for user_id, name in users):
            fields[f"{slot}_link"] = list(users[0])
            if len(users) > 1:
                fields[f"{slot}_also"] = [list(user) for user in users[1:]]
    return fields


def _can_be_left_empty(spec, name):
    """Whether a node runs without this input: an optional one, or one of a growing list that can hold none, as Fibo 2.0's images."""
    inputs = spec.get("input", {})
    group = name.split(".")[0]
    if group in inputs.get("optional", {}):
        return True
    config = inputs.get("required", {}).get(group)
    if not ("." in name and isinstance(config, list) and bool(config) and config[0] == "COMFY_AUTOGROW_V3"):
        return False
    # Those past the fewest it takes: a batch that needs one image runs without its second.
    least = _least(config[1:]) or 0
    entry = name.split(".", 1)[1]
    names = next((found for found in _all_values(config[1:], "names") if isinstance(found, list)), None)
    if names and entry in names:
        return names.index(entry) >= least
    number = re.search(r"(\d+)$", entry)
    return bool(number) and int(number.group(1)) >= least


def _all_values(config, key):
    """Every value under this key, however deep the settings keep it."""
    if isinstance(config, dict):
        for found_key, value in config.items():
            if found_key == key:
                yield value
            yield from _all_values(value, key)
    elif isinstance(config, list):
        for value in config:
            yield from _all_values(value, key)


def _users(api, node_id, slot=None):
    """Every (node id, input) this node's output feeds, or this one output's."""
    return [(user_id, name) for user_id, user in api.items() for name, value in user["inputs"].items()
            if isinstance(value, list) and value[0] == node_id and (slot is None or value[1] == slot)]


def _media_fields(api):
    """
    ref_video, ref_video_2... for each Load Video split into frames and sound for a reference node's
    ref_videos (with _link, _split and _sound saying what to take out when it's left empty), and
    ref_audio, ref_audio_2... for each Load Audio feeding its ref_audios.
    """
    videos, audios = [], []
    for node_id, node in api.items():
        if node["class_type"] == "LoadVideo":
            for split_id, _ in _users(api, node_id):
                if api[split_id]["class_type"] != "GetVideoComponents":
                    continue
                frames = [use for use in _users(api, split_id, 0) if use[1].startswith(f"{VIDEO_GROUP}.")]
                sound = [use for use in _users(api, split_id, 1) if use[1].startswith(f"{VIDEO_SOUND_GROUP}.")]
                if frames:
                    videos.append((frames[0][1], node_id, split_id, frames[0], sound[0] if sound else None))
        elif node["class_type"] == "LoadAudio":
            uses = [use for use in _users(api, node_id) if use[1].startswith(f"{AUDIO_GROUP}.")]
            if uses:
                audios.append((uses[0][1], node_id, uses[0]))
    fields = {}
    for number, (_, node_id, split_id, link, sound) in enumerate(sorted(videos, key=lambda entry: _in_number_order(entry[0])), 1):
        slot = "ref_video" if number == 1 else f"ref_video_{number}"
        fields.update({slot: [node_id, "file"], f"{slot}_link": list(link), f"{slot}_split": [split_id, "video"]})
        if sound:
            fields[f"{slot}_sound"] = list(sound)
    for number, (_, node_id, link) in enumerate(sorted(audios, key=lambda entry: _in_number_order(entry[0])), 1):
        slot = "ref_audio" if number == 1 else f"ref_audio_{number}"
        fields.update({slot: [node_id, "audio"], f"{slot}_link": list(link)})
    return fields


def image_slots(fields):
    """The image inputs a map fills, in order."""
    return [field for field in fields if re.fullmatch(r"image(_\d+)?", field)]


def media_slots(fields, kind):
    """The reference video or audio inputs a map fills, in order: ref_video, ref_video_2..."""
    return [field for field in fields if re.fullmatch(rf"ref_{kind}(_\d+)?", field)]


def takes(fields):
    """
    How many images a workflow can be given. One it can't run without needs a switch to text, or an image.
    And if it takes any, how many reference videos and audio clips, and how many references of every kind together.
    """
    slots = image_slots(fields)
    needed = 0 if "text_to_video" in fields else sum(f"{slot}_link" not in fields for slot in slots)
    found = {"min_images": needed, "max_images": len(slots)}
    videos, audios = len(media_slots(fields, "video")), len(media_slots(fields, "audio"))
    if not videos + audios:
        return found
    most = len(slots) + videos + audios
    limit = fields.get(MOST_REFERENCES_FIELD)
    return {**found, "max_videos": videos, "max_audios": audios,
            "max_references": min(most, int(limit)) if str(limit).isdigit() else most}


def find_fields(api, info, needs_audio=False):
    """Which inputs the pipeline fills: {field: [node id, input]}. Raises Unusable."""
    media = _media_fields(api)
    fields = {"prompt": _prompt_input(api), **_image_fields(api, info), **media}
    if not fields["prompt"]:
        raise Unusable("Couldn't find its prompt: nothing feeds the sampler's positive input, or a prompt input.")
    limit = next((REFERENCE_FILE_LIMITS[node["class_type"]] for node in api.values() if node["class_type"] in REFERENCE_FILE_LIMITS), None)
    if limit:
        fields[MOST_REFERENCES_FIELD] = limit
    references = {node_id for node_id, _ in (media[slot] for slot in media_slots(media, "audio"))}
    audio = _one_input(api, "LoadAudio", "audio", "audio", besides=references)
    if audio:
        fields["audio"] = audio
    elif needs_audio:
        raise Unusable("It has no Load Audio node feeding the video, so she can't lip-sync.")

    switch = _titled(api, r"text.?to.?video", (bool,))
    if switch:
        fields["text_to_video"] = switch
    seconds = _titled(api, r"\b(seconds?|duration)\b", (int, float))
    width, height = _titled(api, r"^width$", (int,)), _titled(api, r"^height$", (int,))
    size_node = next((node_id for node_id, node in api.items()
                      if all(k in node["inputs"] for k in ("width", "height", "length"))), None)
    clip_node = director.director_of(api) or next((node_id for node_id, node in api.items()
                                                    if node["class_type"] == director.DIRECTOR), None)
    if clip_node:
        fields["frames"], fields["fps"] = [clip_node, "total_frames"], [clip_node, "frame_rate"]
        fields["width"], fields["height"] = [clip_node, "width"], [clip_node, "height"]
        return fields
    if seconds:
        fields["seconds"] = seconds
    elif size_node:
        rate_node = next((node_id for node_id, node in api.items() if isinstance(node["inputs"].get("frame_rate"), (int, float))), None)
        if not rate_node:
            raise Unusable("Couldn't find its frame rate, so it can't be told how long a clip is.")
        fields["frames"], fields["fps"] = [size_node, "length"], [rate_node, "frame_rate"]
    else:
        raise Unusable("Couldn't find how to set a clip's length: no input titled in seconds, or length on its latent.")
    if width and height:
        fields["width"], fields["height"] = width, height
    elif size_node:
        fields["width"], fields["height"] = [size_node, "width"], [size_node, "height"]
    return fields


def find_image_fields(api, info):
    """Which inputs the pipeline fills in a workflow that makes an image. Raises Unusable."""
    fields = {"prompt": _prompt_input(api), **_image_fields(api, info)}
    if not fields["prompt"]:
        raise Unusable("Couldn't find its prompt: nothing feeds the sampler's positive input, or a prompt input.")
    width, height = _titled(api, r"^width$", (int,)), _titled(api, r"^height$", (int,))
    if width and height:
        fields["width"], fields["height"] = width, height
        return fields
    sized = [node_id for node_id, node in api.items() if all(k in node["inputs"] for k in ("width", "height"))]
    # The empty latent decides the size. An upscale further on only changes what that size becomes.
    size_node = next((node_id for node_id in sized if "latent" in api[node_id]["class_type"].lower()), sized[0] if sized else None)
    if size_node:
        fields["width"], fields["height"] = [size_node, "width"], [size_node, "height"]
    return fields


# Models that edit an image: image 1 is what they edit, so it's always needed, and the rest are optional.
EDIT_MODELS = ("qwen_image_21_edit",)


def image_takes(fields, model=None, options=None):
    """
    As takes(), whether its size can be chosen (by its own inputs, or by the source image it follows), its model,
    and what else a run can change in it, if anything (core/workflow_options.py). A model that holds fewer images
    than its workflow has room for takes only those (pipelines/image/models.py).
    """
    from pipelines.image import models
    found = {**takes(fields), "size": "width" in fields or bool(image_slots(fields)), "model": model}
    if model in models.MOST_IMAGES:
        found["max_images"] = min(found["max_images"], models.MOST_IMAGES[model])
    if model in EDIT_MODELS and found["max_images"]:
        found["min_images"] = max(found["min_images"], 1)
    return {**found, "options": options} if options else found


def with_options(found, options):
    """takes() with what a run can change in the workflow, if anything (core/workflow_options.py)."""
    return {**found, "options": options} if options else found


def image_model(api):
    """Which image model a workflow runs, if it's one with prompts or limits of its own. None otherwise."""
    for node in api.values():
        if node["class_type"] in IMAGE_MODEL_NODES:
            return IMAGE_MODEL_NODES[node["class_type"]]
    for node in api.values():
        if node["class_type"] == QWEN_IMAGE_21_ENCODER:
            # An edit shows the model its image as well as making one that size, so it holds less: models.py.
            edits = any(name.startswith("images.") and isinstance(value, list) for name, value in node["inputs"].items())
            return "qwen_image_21_edit" if edits else "qwen_image_21"
    for node in api.values():
        if node["class_type"] == "CLIPLoader" and node["inputs"].get("type") in IMAGE_MODEL_ENCODERS:
            return node["inputs"]["type"]
    return None


def makes_images(ui):
    """Whether a saved workflow saves images and no video, subgraphs and all."""
    kinds = {node["type"] for node in ui.get("nodes", [])}
    kinds |= {node["type"] for subgraph in ui.get("definitions", {}).get("subgraphs", []) for node in subgraph.get("nodes", [])}
    return bool(kinds & {*IMAGE_SAVES, PREVIEW}) and not kinds & set(VIDEO_SAVES)


def frames_for(seconds, fps):
    return round(seconds * fps / FRAME_STEP) * FRAME_STEP + 1


# --- Fetching them from ComfyUI, and keeping what the pipeline runs ---

def _write_json(path, data):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(temp, path)


def _label(name):
    return name[:-5] if name.lower().endswith(".json") else name


def display_label(name):
    """A workflow name for the panel, without its categorization prefix."""
    label = _label(name)
    parts = re.split(r"([\\/])", label)
    parts[-1] = re.sub(r"^(image|video|shorts?)_(?=.+)", "", parts[-1], flags=re.I)
    return "".join(parts)


def is_for_shorts(name):
    return _label(name).endswith(SHORTS_SUFFIX)


def is_for_image(name):
    return not is_for_shorts(name) and _label(name).replace("\\", "/").rsplit("/", 1)[-1].startswith(IMAGE_PREFIX)


def shorthand(workflow, default):
    """
    A workflow's short name, for file names and the gallery: its name without the image_, video_ or shorts_ in front
    or the _shorts marker behind. default for the built-in one.
    """
    if not workflow:
        return default
    stem = os.path.splitext(os.path.basename(str(workflow)))[0]
    stem = re.sub(r"^(image|video|shorts?)_", "", stem, flags=re.I)
    stem = re.sub(re.escape(SHORTS_SUFFIX) + "$", "", stem, flags=re.I)
    return re.sub(r"[^\w-]+", "_", stem).strip("_") or default


def title_words(title, default):
    """A title as words for a file name."""
    return re.sub(r"[^\w-]+", "_", title or "").strip("_")[:60] or default


def unique_name(folder, name, extension=""):
    """name, or name_2, name_3... if folder already has one: a second film with the same title keeps the first."""
    candidate, number = name, 1
    while os.path.exists(os.path.join(folder, candidate + extension)):
        number += 1
        candidate = f"{name}_{number}"
    return candidate


def pipeline_of(name, ui=None):
    """Which side a workflow is for: by its name, or for one saved without image_ in it, by what it saves."""
    if is_for_shorts(name):
        return "shorts"
    return "image" if is_for_image(name) or (ui is not None and makes_images(ui)) else "video"


def _file_stem(name):
    return re.sub(r"[^\w.-]+", "_", _label(name))


# How many references a film gets on a node whose reference inputs grow, such as MiniMax's,
# however many the saved workflow happens to wire up.
MOST_REFERENCES = 20
# A reference node's inputs for videos (frames, and each one's own sound) and audio clips, as MiniMax H3 names them.
VIDEO_GROUP, VIDEO_SOUND_GROUP, AUDIO_GROUP = "ref_videos", "ref_video_audios", "ref_audios"
# How many it takes when its own settings don't say.
MOST_MEDIA_REFERENCES = 3
# How many references of every kind together a node takes, where that's fewer than its inputs add up to.
# MiniMax H3 takes 12: 9 images and 3 videos or audio clips, at most 3 of each.
REFERENCE_FILE_LIMITS = {"MiniMaxH3ReferenceToVideo": 12}
MOST_REFERENCES_FIELD = "_most_references"


def _most(config):
    """The largest count a growing input's settings allow, wherever they keep it, or None. A list of its names says too."""
    if isinstance(config, dict):
        found = [value for key, value in config.items() if key == "max" and isinstance(value, int)]
        if not found and isinstance(config.get("names"), list) and config["names"]:
            found = [len(config["names"])]
        return found[0] if found else next(filter(None, (_most(value) for value in config.values())), None)
    if isinstance(config, list):
        return next(filter(None, (_most(value) for value in config)), None)
    return None


def _least(config):
    """The fewest a growing input's settings allow, wherever they keep it, or None if they don't say."""
    if isinstance(config, dict):
        if isinstance(config.get("min"), int):
            return config["min"]
        return next((found for found in map(_least, config.values()) if found is not None), None)
    if isinstance(config, list):
        return next((found for found in map(_least, config) if found is not None), None)
    return None


def _grow_references(api, info):
    """
    Wires more Load Image nodes into a reference input that grows (ref_images.ref_image_0, _1...),
    up to MOST_REFERENCES or the node's own limit. Each new one is optional, so one left empty
    comes out of the workflow again. Inputs that already share their images, as Qwen-Image 2.1's encoder and the
    batch its prompt enhancer looks at do, grow together: each new image goes to all of them, never one of its own.
    """
    growing = []
    for node_id, node in list(api.items()):
        spec = _input_spec(info.get(node["class_type"], {}))
        for group, config in spec.items():
            if not (isinstance(config, list) and config and config[0] == "COMFY_AUTOGROW_V3"):
                continue
            wired = {}
            for name, value in node["inputs"].items():
                match = re.fullmatch(rf"{re.escape(group)}\.(\D*?)(\d+)", name)
                if match and isinstance(value, list) and api.get(value[0], {}).get("class_type") == "LoadImage":
                    wired[int(match.group(2))] = (match.group(1), value[0])
            if not wired:
                if group in (VIDEO_GROUP, AUDIO_GROUP) and not any(name.startswith(f"{group}.") for name in node["inputs"]):
                    _add_media_references(api, node, group, _most(config[1:]) or MOST_MEDIA_REFERENCES, VIDEO_SOUND_GROUP in spec)
                continue
            growing.append({"node": node, "group": group, "wired": wired, "most": min(MOST_REFERENCES, _most(config[1:]) or MOST_REFERENCES)})
    clusters = []
    for entry in growing:
        sources = {source for _, source in entry["wired"].values()}
        joined = [cluster for cluster in clusters if cluster["sources"] & sources]
        cluster = {"members": [entry], "sources": set(sources)}
        for other in joined:
            clusters.remove(other)
            cluster["members"] += other["members"]
            cluster["sources"] |= other["sources"]
        clusters.append(cluster)
    for cluster in clusters:
        members = cluster["members"]
        # The one wired to the most leads; the others take the same new images, in order, after theirs.
        lead = max(members, key=lambda entry: len(entry["wired"]))
        most = min(entry["most"] for entry in members)
        example = lead["wired"][max(lead["wired"])][1]
        for count in range(len(lead["wired"]) + 1, most + 1):
            added = _add_node(api, "LoadImage", dict(api[example]["inputs"]), f"Reference {count} (added by the pipeline)")
            for entry in members:
                prefix = entry["wired"][max(entry["wired"])][0]
                number = max(entry["wired"]) + 1
                entry["wired"][number] = (prefix, added)
                entry["node"]["inputs"][f"{entry['group']}.{prefix}{number}"] = [added, 0]


def _add_node(api, class_type, inputs, title):
    """Adds a node under the next free id, and returns that id."""
    node_id = str(max(int(key) for key in api if key.isdigit()) + 1)
    api[node_id] = _node(class_type, inputs, title)
    return node_id


def _add_media_references(api, node, group, most, sounds):
    """
    Load Video or Load Audio nodes for each of a reference node's video or audio inputs, which a saved
    workflow seldom wires. A video is split into the frames and its own sound, each to its input.
    """
    prefix = group[:-1] + "_"
    for number in range(most):
        if group == AUDIO_GROUP:
            audio = _add_node(api, "LoadAudio", {"audio": ""}, f"Reference audio {number + 1} (added by the pipeline)")
            node["inputs"][f"{group}.{prefix}{number}"] = [audio, 0]
            continue
        video = _add_node(api, "LoadVideo", {"file": ""}, f"Reference video {number + 1} (added by the pipeline)")
        split = _add_node(api, "GetVideoComponents", {"video": [video, 0]}, f"Reference video {number + 1}, split")
        node["inputs"][f"{group}.{prefix}{number}"] = [split, 0]
        if sounds:
            node["inputs"][f"{VIDEO_SOUND_GROUP}.{VIDEO_SOUND_GROUP[:-1]}_{number}"] = [split, 1]


def _prepare(name, ui, info, folder, pipeline):
    """Writes what the pipeline runs. Returns its path, and how many images it takes."""
    stem = os.path.join(folder, _file_stem(name))
    # Here and not at the top: it builds on this module.
    from core import workflow_options
    api, fields, model, _ = workflow_options.convert(ui, info, pipeline=pipeline)
    options = workflow_options.offered(ui, info, pipeline)
    if options:
        # A run with other options makes its workflow again from these, with what the pipeline adds to it.
        kinds = {node["type"] for node in _without_subgraphs(_with_hidden_links(ui))["nodes"]}
        kinds |= {PREVIEW, COMBINE, "LoraLoader", "LoadImage", "LoadVideo", "LoadAudio", "GetVideoComponents",
                  workflow_options.H3_MEMORY, workflow_options.H3_SPARSE, director.INPUTS, director.ASSETS}
        _write_json(stem + ".ui.json", ui)
        _write_json(stem + ".info.json", {kind: info[kind] for kind in kinds if kind in info})
    _write_json(stem + ".json", api)
    _write_json(stem + ".map.json", {"_saved_as": name, "_pipeline": pipeline, **({"_model": model} if model else {}),
                                     **({"_options": options} if options else {}),
                                     **{field: value if field.startswith("_") else f"{value[0]}.{value[1]}" for field, value in fields.items()}})
    return stem + ".json", image_takes(fields, model, options) if pipeline == "image" else with_options(takes(fields), options)


def _from_comfyui(url, folder):
    names = requests.get(f"{url}/api/userdata", params={"dir": "workflows", "recurse": "true"}, timeout=5).json()
    # Takes a few seconds: ComfyUI asks every node it has.
    info = requests.get(f"{url}/object_info", timeout=60).json()
    os.makedirs(folder, exist_ok=True)
    found = []
    for name in sorted(n for n in names if n.lower().endswith(".json")):
        entry = {"name": name, "label": _label(name), "pipeline": pipeline_of(name)}
        try:
            response = requests.get(f"{url}/api/userdata/{quote('workflows/' + name, safe='')}", timeout=10)
            response.raise_for_status()
            ui = response.json()
            if "nodes" not in ui:
                raise Unusable("It's in API format, which has no way to tell which inputs to fill. Save it from the editor.")
            entry["pipeline"] = pipeline_of(name, ui)
            entry["path"], entry["takes"] = _prepare(name, ui, info, folder, entry["pipeline"])
        except Unusable as e:
            entry["problem"] = str(e)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError, requests.RequestException) as e:
            entry["problem"] = f"Couldn't read it ({e})."
        found.append(entry)
    _forget_deleted(folder, names)
    # The panel only offers what runs, so this is where to find out why one isn't there.
    _write_json(os.path.join(folder, "not_listed.json"), {entry["label"]: entry["problem"] for entry in found if "problem" in entry})
    return found


def _forget_deleted(folder, names):
    # Only what ComfyUI no longer has: a saved run may still use one the list no longer offers.
    keep = {_file_stem(name) + ".json" for name in names}
    for file in os.listdir(folder):
        if file.endswith(".map.json") and file[:-len(".map.json")] + ".json" not in keep:
            stem = file[:-len(".map.json")]
            for stale in (file, stem + ".json", stem + ".ui.json", stem + ".info.json"):
                if os.path.exists(os.path.join(folder, stale)):
                    os.remove(os.path.join(folder, stale))


def _from_earlier(folder):
    """With ComfyUI off: the ones it had last time. A run starts it, so they can still be chosen."""
    found = []
    if not os.path.isdir(folder):
        return found
    for file in sorted(os.listdir(folder)):
        if file.endswith(".map.json"):
            path = os.path.join(folder, file[:-len(".map.json")] + ".json")
            try:
                with open(os.path.join(folder, file), encoding="utf-8") as f:
                    fields = json.load(f)
                name = fields["_saved_as"]
            except (OSError, ValueError, KeyError):
                continue
            pipeline = fields.get("_pipeline") or pipeline_of(name)
            if os.path.exists(path) and ("audio" in fields or not is_for_shorts(name)):
                found.append({"name": name, "label": _label(name), "pipeline": pipeline, "path": path,
                              "takes": image_takes(fields, fields.get("_model"), fields.get("_options")) if pipeline == "image"
                              else with_options(takes(fields), fields.get("_options"))})
    return found


def list_saved(url, folder):
    """Each saved workflow: {name, label, and path and takes to run it, or problem saying why it can't}."""
    try:
        return _from_comfyui(url, folder)
    except (OSError, ValueError, requests.RequestException):
        return _from_earlier(folder)

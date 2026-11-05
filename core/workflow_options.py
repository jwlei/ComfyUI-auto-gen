"""
What a run can change in an image workflow besides its prompt, images and size, and making the workflow with
those changes. Community workflows such as LatentHeart's Krea2 SFW/NSFW one are built for switching parts on
and off in ComfyUI's editor: groups that rgthree's Fast Groups Bypasser turns on and off (2nd pass, upscaler,
grain, sharpen...), a choice on a group's subgraph (the upscaler's resolution), LoRAs in rgthree's Power Lora
Loader, and the Krea2T Enhancer's prompt-adherence boost. ComfyUI's own templates put theirs on a subgraph's
face instead: Krea-2's enable_lora with its LoRA and strength, Qwen-Image-Edit's enable_turbo_mode. MiniMax H3's
template leaves a named Boolean on the canvas, Enable Lightning LoRA, that picks the turbo LoRA and its steps. And
any workflow whose sampler's (or scheduler's) steps can be found gets Steps. Each becomes an option on the form;
turbo and Steps are marked advanced, for the form's Advanced part, on every side.

The editor bypasses a group by setting its nodes' mode, and that changes which nodes the API gets, so a run
with options makes its workflow again from what was saved (core/saved_workflows.py keeps a copy for it).

Each option is {id, kind, label, default, ...}, and a run's choice is a string:
    switch    "1" or "0": a group on or off, or a switch on a subgraph's face
    choice    one of its choices: a value on a subgraph, shown with the switch it goes with
    number    a number: a strength that goes with a switch, or the sampler's steps
    strength  "0" for off, or the strength: a LoRA, or the Krea2T Enhancer
"""
import copy
import json
import os
import re

from core import lora_check
from core import saved_workflows as sw

BYPASSERS = ("Fast Groups Bypasser (rgthree)", "Fast Groups Muter (rgthree)")
# LiteGraph's group colours, as rgthree's matchColors names them.
COLOR_NAMES = {"red": "#a88", "brown": "#b06634", "green": "#8a8", "blue": "#88a", "pale_blue": "#3f789e",
               "cyan": "#8aa", "purple": "#a1309b", "yellow": "#b58b2a", "black": "#444"}
# The pipeline writes the prompt and sets the size itself, so the groups that would are its own to set: a prompt
# writer or enhancer stays off, and what sizes the image stays as saved.
PROMPT_GROUPS = re.compile(r"prompt", re.I)
SIZE_GROUPS = re.compile(r"image size|resolution", re.I)
LORA_LOADER = "Power Lora Loader (rgthree)"
ENHANCER = "ComfyUI-Krea2T-Enhancer"
UPSCALER_LOADER = "SeedVR2LoadDiTModel"
CHOOSER = "CustomCombo"
ON, OFF = "1", "0"
BYPASSED = 4
# What a subgraph's face sets that isn't for the form: what the pipeline sets itself (prompt, size, seed), the
# settings of a prompt writer it keeps off, and which model files it loads.
FACE_SKIP = re.compile(r"prompt|enhance|refine|thinking|max_tok|max_length|width|height|seed|unet|clip|vae|ckpt|trigger"
                       r"|text.?to.?video|duration|length|frames|fps|audio", re.I)
# The two ways a workflow can write its own prompt, offered beside Gemini and Claude on the Image side: a prompt
# enhancer that expands yours, and image to prompt, which describes a picture. Each is a group, or for the enhancer
# a Refine Prompt switch as Krea-2's template has. On Video and Shorts the LLM always writes them, so they stay off.
ENHANCER_GROUP = re.compile(r"enhanc", re.I)
IMAGE_TO_PROMPT = re.compile(r"image.?to.?prompt|img.?2.?prompt|caption", re.I)
REFINE_SWITCH = re.compile(r"refine|enhance", re.I)
ROLES = ("prompt_enhancer", "image_to_prompt")
# Words a switch's name has that say nothing about what it switches.
SWITCH_WORDS = {"enable", "enabled", "use", "mode", "on"}
MOST_STEPS = 150
# A switch left on the canvas as a named Boolean, as ComfyUI's MiniMax H3 template has "Boolean (Enable Lightning
# LoRA)": it picks the turbo LoRA and its step count through If/Else switches. Offered under Advanced, with Steps.
PRIMITIVE_SWITCH = "PrimitiveBoolean"
TYPE_IN_TITLE = re.compile(r"\s*(?:boolean|bool)\s*\((.+)\)\s*", re.I)
TURBO_WORDS = re.compile(r"lightning|lightx|turbo|\d+.?steps?\b", re.I)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "option"


def _inside(node, group):
    x, y = (node.get("pos") or [0, 0])[:2]
    left, top, width, height = group["bounding"]
    return left <= x <= left + width and top <= y <= top + height


def _switched_groups(ui):
    """The groups a Fast Groups Bypasser in the workflow turns on and off: by title, colour, or both if it says both."""
    rules = []
    for node in ui["nodes"]:
        if node["type"] in BYPASSERS and not node.get("mode"):
            found = node.get("properties") or {}
            colors = [COLOR_NAMES.get(name.strip().lower(), name.strip().lower()) for name in (found.get("matchColors") or "").split(",") if name.strip()]
            rules.append((found.get("matchTitle") or "", colors))
    switched = []
    for group in ui.get("groups", []):
        title, color = group.get("title") or "", (group.get("color") or "").lower()
        for pattern, colors in rules:
            try:
                by_title = not pattern or re.search(pattern, title, re.I)
            except re.error:
                by_title = False
            # One with neither would switch every group, which isn't a part of the workflow worth offering.
            if (pattern or colors) and by_title and (not colors or color in colors):
                switched.append(group)
                break
    return switched


def _subgraphs(ui):
    return {subgraph["id"]: subgraph for subgraph in ui.get("definitions", {}).get("subgraphs", [])}


def _face_choices(instance, subgraph, prefix, label):
    """A choice set on a subgraph's face that feeds a Custom Combo inside it, with that combo's list."""
    found = []
    widgets = sw._face_widgets(instance, subgraph)
    combos = {node["id"]: node for node in subgraph.get("nodes", []) if node["type"] == CHOOSER}
    for link in map(sw._link_row, subgraph.get("links", [])):
        if link[1] != sw.SUBGRAPH_IN or link[3] not in combos:
            continue
        name = subgraph["inputs"][link[2]]["name"]
        values = combos[link[3]].get("widgets_values") or []
        choices = [value for value in values[2:] if isinstance(value, str) and value]
        if name in widgets and len(choices) > 1:
            found.append({"id": f"{prefix}.{_slug(name)}", "kind": "choice", "label": f"{label}: {name}", "default": str(widgets[name]),
                          "choices": choices, "parent": prefix, "node": instance["id"], "input": name})
    return found


def _words(label):
    return " ".join(label.replace("_", " ").split()).capitalize().replace("Lora", "LoRA").replace("lora", "LoRA")


def _inner_input(subgraph, slot):
    """The node type and input a subgraph's input feeds first, inside it."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    for link in map(sw._link_row, subgraph.get("links", [])):
        if link[1] == sw.SUBGRAPH_IN and link[2] == slot and link[3] in nodes:
            inputs = nodes[link[3]].get("inputs", [])
            if link[4] < len(inputs):
                return nodes[link[3]]["type"], inputs[link[4]]["name"]
    return None, None


def _face_settings(instance, subgraph, info):
    """
    The switches on a subgraph's face, as ComfyUI's templates have them (enable_lora, enable_turbo_mode), and what goes
    with each by name: lora_name and lora_strength with enable_lora. A value that goes with no switch isn't offered.
    """
    widgets = sw._face_widgets(instance, subgraph)
    faced = [(slot, entry, entry.get("label") or entry["name"]) for slot, entry in enumerate(subgraph.get("inputs", []))
             if entry["name"] in widgets and not FACE_SKIP.search(entry.get("label") or entry["name"])]
    switches = [{"id": f"face_{_slug(label)}", "kind": "switch", "label": _words(label), "default": ON if widgets[entry["name"]] else OFF,
                 "node": instance["id"], "input": entry["name"], **({"advanced": True} if TURBO_WORDS.search(label) else {})}
                for _, entry, label in faced if entry["type"] == "BOOLEAN" and isinstance(widgets[entry["name"]], bool)]
    found = list(switches)
    for slot, entry, label in faced:
        words = set(_slug(label).split("_"))
        parent = next((switch for switch in switches if words & (set(switch["id"][5:].split("_")) - SWITCH_WORDS)), None)
        value = widgets[entry["name"]]
        if not parent or entry["type"] == "BOOLEAN":
            continue
        common = {"id": f"face_{_slug(label)}", "label": _words(label), "default": str(value), "parent": parent["id"],
                  "node": instance["id"], "input": entry["name"]}
        if entry["type"] == "FLOAT" and isinstance(value, (int, float)):
            found.append({**common, "kind": "number", "min": -2 if "strength" in label else 0, "max": 2 if "strength" in label else 100, "step": 0.05})
        elif entry["type"] == "COMBO":
            kind, name = _inner_input(subgraph, slot)
            spec = sw._input_spec(info.get(kind, {})).get(name) if kind else None
            choices = sw._options(spec) if spec else None
            if choices and len(choices) > 1:
                found.append({**common, "kind": "choice", "choices": [str(choice) for choice in choices]})
    return found


def _primitive_label(title):
    """'Boolean (Enable Lightning LoRA)' as 'Lightning LoRA (turbo)': what it switches, and that it's the fast way."""
    match = TYPE_IN_TITLE.fullmatch(title)
    words = (match.group(1) if match else title).split()
    while len(words) > 1 and words[0].lower() in SWITCH_WORDS:
        words = words[1:]
    label = " ".join(words)
    label = label[0].upper() + label[1:]
    return f"{label} (turbo)" if TURBO_WORDS.search(label) and "turbo" not in label.lower() else label


def _primitive_switches(ui):
    """The named Booleans on the canvas that something reads: each a switch. One for the prompt is the pipeline's."""
    used = {sw._link_row(link)[1] for link in ui.get("links", [])}
    found = []
    for node in ui["nodes"]:
        title = (node.get("title") or "").strip()
        values = node.get("widgets_values") or []
        if (node["type"] != PRIMITIVE_SWITCH or node.get("mode") or not title or node["id"] not in used
                or not values or not isinstance(values[0], bool) or REFINE_SWITCH.search(title) or FACE_SKIP.search(title)):
            continue
        label = _primitive_label(title)
        found.append({"id": f"switch_{_slug(label.replace('(turbo)', ''))}", "kind": "switch", "label": label,
                      "default": ON if values[0] else OFF, "node": node["id"], "primitive": True, "advanced": True})
    return found


def _role(role, label, group=None):
    return {"id": role, "kind": "switch", "role": "llm", "label": label, "default": OFF, **({"group": group} if group else {})}


def describe(ui, info=None, pipeline="image"):
    """The options this workflow offers, as saved. A problem with one is for the caller to find, by trying it."""
    options = []
    subgraphs = _subgraphs(ui)
    for node in ui["nodes"]:
        if node["type"] in subgraphs and not node.get("mode"):
            options += _face_settings(node, subgraphs[node["type"]], info or {})
    for group in _switched_groups(ui):
        title = group.get("title") or "Group"
        if PROMPT_GROUPS.search(title):
            role = "prompt_enhancer" if ENHANCER_GROUP.search(title) else "image_to_prompt" if IMAGE_TO_PROMPT.search(title) else None
            if pipeline == "image" and role and role not in [option["id"] for option in options]:
                options.append(_role(role, title, title))
            continue
        if SIZE_GROUPS.search(title):
            continue
        members = [node for node in ui["nodes"] if _inside(node, group) and node["type"] not in sw.NOTES + sw.EDITOR_ONLY]
        if not members:
            continue
        group_id = f"group_{_slug(title)}"
        options.append({"id": group_id, "kind": "switch", "label": title, "default": OFF if all(node.get("mode") == BYPASSED for node in members) else ON})
        for node in members:
            if node["type"] in subgraphs:
                options += _face_choices(node, subgraphs[node["type"]], group_id, title)
    for node in ui["nodes"]:
        if node["type"] == LORA_LOADER and not node.get("mode"):
            loras = [value for value in node.get("widgets_values") or [] if isinstance(value, dict) and value.get("lora")]
            for number, lora in enumerate(loras, 1):
                name = os.path.splitext(os.path.basename(str(lora["lora"]).replace("\\", "/")))[0]
                options.append({"id": f"lora_{_slug(name)}", "kind": "strength", "label": f"LoRA: {name}", "node": node["id"], "entry": f"lora_{number}",
                                "default": str(lora.get("strength", 1)) if lora.get("on") else OFF, "min": -2, "max": 2, "step": 0.05,
                                "usual": str(lora.get("strength", 1) or 1)})
        elif node["type"] == ENHANCER and not node.get("mode"):
            enabled, strength = (list(node.get("widgets_values") or []) + [True, 1.0])[:2]
            options.append({"id": "prompt_boost", "kind": "strength", "label": "Krea2T Enhancer (prompt adherence)", "node": node["id"],
                            "default": str(strength) if enabled else OFF, "min": 0, "max": 2, "step": 0.05, "usual": str(strength or 1)})
    options += _primitive_switches(ui)
    every = list(ui["nodes"]) + [node for subgraph in subgraphs.values() for node in subgraph.get("nodes", [])]
    if pipeline == "image" and "prompt_enhancer" not in [option["id"] for option in options] and any(
            node["type"] == "PrimitiveBoolean" and REFINE_SWITCH.search(node.get("title") or "") for node in every):
        options.append(_role("prompt_enhancer", "Prompt enhancer"))
    return options


def _set_mode(ui, node, mode):
    node["mode"] = mode
    # The editor sets a subgraph's insides with it, and the saved file has them that way.
    for subgraph in ui.get("definitions", {}).get("subgraphs", []):
        if subgraph["id"] == node["type"]:
            for inner in subgraph.get("nodes", []):
                inner["mode"] = mode


def _before(ui, options, chosen):
    """The saved workflow with the groups and face choices as chosen, and every prompt writer or enhancer off."""
    ui = copy.deepcopy(ui)
    for group in _switched_groups(ui):
        title = group.get("title") or ""
        option = next((option for option in options if option["kind"] == "switch" and option["label"] == title), None)
        if option and option.get("role"):
            wanted = chosen.get(option["id"], OFF)
        else:
            wanted = OFF if PROMPT_GROUPS.search(title) else chosen.get(option["id"], option["default"]) if option else None
        if wanted is None:
            continue
        for node in ui["nodes"]:
            if _inside(node, group) and node["type"] not in sw.NOTES + sw.EDITOR_ONLY:
                _set_mode(ui, node, 0 if wanted == ON else BYPASSED)
    nodes = {node["id"]: node for node in ui["nodes"]}
    subgraphs = _subgraphs(ui)
    for option in options:
        # A named Boolean on the canvas: set before the workflow is made, as the Image side settles switches then.
        if option.get("primitive") and option["id"] in chosen and option["node"] in nodes:
            nodes[option["node"]]["widgets_values"][0] = chosen[option["id"]] == ON
            continue
        # A value on a subgraph's face: a choice, a switch or a number there.
        if "input" in option and option["id"] in chosen and option["node"] in nodes:
            node = nodes[option["node"]]
            names = (sw.face_names(node, subgraphs[node["type"]]) if node["type"] in subgraphs
                     else [entry["widget"]["name"] for entry in node.get("inputs", []) if "widget" in entry])
            values = node.get("widgets_values")
            value = chosen[option["id"]]
            value = value == ON if option["kind"] == "switch" else float(value) if option["kind"] == "number" else value
            if isinstance(values, list) and option["input"] in names and names.index(option["input"]) < len(values):
                values[names.index(option["input"])] = value
    return ui


def _flag(api, value):
    """A switch's on or off: as set, or from the Boolean it reads. None if that can't be told."""
    if isinstance(value, list):
        value = api.get(value[0], {}).get("inputs", {}).get("value")
    return value if isinstance(value, bool) else None


def _number_source(api, node_id, name, depth=0, kinds=(int,)):
    """Where the number an input takes is set: [node id, input], through a primitive or the side a switch takes. Or None."""
    value = api.get(node_id, {}).get("inputs", {}).get(name)
    if type(value) in kinds:
        return [node_id, name]
    if not isinstance(value, list) or depth > 4:
        return None
    source = api.get(value[0], {}).get("inputs", {})
    if type(source.get("value")) in kinds:
        return [value[0], "value"]
    if api.get(value[0], {}).get("class_type") in sw.SWITCHES:
        on = _flag(api, source.get("switch"))
        return None if on is None else _number_source(api, value[0], "on_true" if on else "on_false", depth + 1, kinds)
    return None


def _text_source(api, node_id, name):
    """[node id, input] if the input holds its text itself, rather than taking it from another node. Or None."""
    return [node_id, name] if isinstance(api.get(node_id, {}).get("inputs", {}).get(name), str) else None


def _distinct(found):
    return [target for number, target in enumerate(found) if target and target not in found[:number]]


def _combo(info, class_type, name):
    """The values ComfyUI offers this node's input, or []."""
    inputs = sw._input_spec(info.get(class_type, {}))
    return list(sw._options(inputs[name]) or []) if name in inputs else []


FLOATS = (int, float)
MOST_CFG = 30
MOST_TEXT = 4000
FOLLOWERS = ("steps", "cfg", "sampler", "scheduler")
NEGATIVE_TEXT = ("CLIPTextEncode",)
LTX_FPS = "LTXVConditioning"
SLOT = "lora_slot"
NO_LORA = "None"
DISTILLED = "The workflow sets its own steps with fixed sigmas, as a distilled one does."
H3_MEMORY, H3_SPARSE = "H3MemoryOptimization", "H3SparseAttention"
H3_SAMPLING = ("BasicGuider", "CFGGuider", "KSampler", "KSamplerAdvanced", "SamplerCustom")


def _negatives(api):
    """Where a negative prompt is typed, if one node has it: [node id, input]. Or None."""
    positive = {value[0] for node in api.values() for name, value in node["inputs"].items() if name == "positive" and isinstance(value, list)}
    found = [[node_id, "text"] for node_id, node in api.items() if node["class_type"] in NEGATIVE_TEXT and node_id not in positive
             and isinstance(node["inputs"].get("text"), str)
             and any(isinstance(value, list) and value[0] == node_id for other in api.values() for name, value in other["inputs"].items() if name == "negative")]
    found += [[node_id, "negative_prompt"] for node_id, node in api.items() if isinstance(node["inputs"].get("negative_prompt"), str)]
    return found[0] if len(found) == 1 else None


def _tune(api, info, chosen):
    """
    What the sampling takes, wherever the graph has it: sampler, scheduler and CFG (under Steps), the negative prompt,
    and for LTX its frame rate, image strength, NAG scale and a LoRA to add. Applies what's chosen. Returns the options.
    """
    options = []
    locked = any(node["class_type"] == "ManualSigmas" for node in api.values())

    def offer(option, targets, value=str):
        option = {"advanced": True, **option, "default": str(api[targets[0][0]]["inputs"][targets[0][1]])}
        if locked and option.get("parent") == "steps":
            option["problem"] = DISTILLED
        options.append(option)
        if option["id"] in chosen and not option.get("problem"):
            for node_id, name in targets:
                api[node_id]["inputs"][name] = value(chosen[option["id"]])

    for name, label in (("sampler_name", "Sampler"), ("scheduler", "Scheduler")):
        targets = _distinct([_text_source(api, node_id, name) for node_id, node in api.items() if name in node["inputs"]])
        choices = _combo(info, api[targets[0][0]]["class_type"], name) if targets else []
        if choices and (len(targets) == 1 or locked):
            offer({"id": name.split("_")[0], "kind": "choice", "label": label, "parent": "steps", "choices": choices}, targets[:1])
    cfg = _distinct([_number_source(api, node_id, "cfg", kinds=FLOATS) for node_id, node in api.items() if "cfg" in node["inputs"]])
    if cfg and (len(cfg) == 1 or locked):
        offer({"id": "cfg", "kind": "number", "label": "CFG", "parent": "steps", "min": 0, "max": MOST_CFG, "step": 0.1}, cfg[:1], float)
    negative = _negatives(api)
    # With the CFG fixed at 1 by distilled sigmas, a negative prompt is never used, unless NAG reads it.
    if negative and (not locked or any(node["class_type"] == "LTX2_NAG" for node in api.values())):
        offer({"id": "negative", "kind": "text", "label": "Negative prompt"}, [negative])
    if any(node["class_type"] == LTX_FPS for node in api.values()):
        fps = _distinct([_number_source(api, node_id, "frame_rate", kinds=FLOATS) for node_id, node in api.items() if "frame_rate" in node["inputs"]])
        if len(fps) == 1:
            offer({"id": "fps", "kind": "number", "label": "Frames per second", "min": 1, "max": 120, "step": 1}, fps, float)
    strength = _distinct([_number_source(api, node_id, "strength", kinds=FLOATS) for node_id, node in api.items()
                          if node["class_type"] in ("LTXVImgToVideoInplace", "LTXVImgToVideo")])
    if strength:
        offer({"id": "image_strength", "kind": "number", "label": "Image strength", "min": 0, "max": 1, "step": 0.05}, strength, float)
    nag = _distinct([_number_source(api, node_id, "nag_scale", kinds=FLOATS) for node_id, node in api.items() if node["class_type"] == "LTX2_NAG"])
    if nag:
        offer({"id": "nag_scale", "kind": "number", "label": "NAG scale", "min": 0, "max": MOST_CFG, "step": 0.5}, nag, float)
    loaders = [node_id for node_id, node in api.items() if node["class_type"] == LORA_LOADER]
    if len(loaders) == 1 and not any(re.fullmatch(r"lora_\d+", name) for name in api[loaders[0]]["inputs"]):
        loras = _combo(info, "LoraLoader", "lora_name") or _combo(info, "LoraLoaderModelOnly", "lora_name")
        if loras:
            options.append({"id": SLOT, "kind": "choice", "label": "Add a LoRA", "choices": [NO_LORA, *loras], "default": NO_LORA, "advanced": True})
            options.append({"id": f"{SLOT}_strength", "kind": "number", "label": "LoRA strength", "parent": SLOT, "default": "1", "min": -2, "max": 2,
                            "step": 0.05, "advanced": True})
            if chosen.get(SLOT, NO_LORA) != NO_LORA:
                api[loaders[0]]["inputs"]["lora_1"] = {"on": True, "lora": chosen[SLOT], "strength": float(chosen.get(f"{SLOT}_strength", 1)), "strengthTwo": None}
    options += _h3(api, info, chosen)
    return options


def _node_defaults(info, class_type):
    """The inputs a node needs besides its model, as ComfyUI offers them by default."""
    values = {}
    for name, config in sw._input_spec(info.get(class_type, {})).items():
        if name == "model":
            continue
        extra = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
        choices = sw._options(config)
        value = extra.get("default", choices[0] if choices else None)
        if value is not None:
            values[name] = value
    return values


def _h3(api, info, chosen):
    """
    MiniMax H3's two optimizers (the H3-Optimizations custom node), which a workflow doesn't have: each is put between the
    model and what samples with it, when chosen. Sparse attention also takes how much of the video it attends to.
    """
    if not any(node["class_type"].startswith("MiniMaxH3") for node in api.values()) or not all(name in info for name in (H3_MEMORY, H3_SPARSE)):
        return []
    sparse = _node_defaults(info, H3_SPARSE)
    budget = float(sparse.get("video_budget", 0.15))
    options = [
        {"id": "h3_memory", "kind": "switch", "label": "H3 Memory Optimization", "default": OFF, "advanced": True},
        {"id": "h3_sparse", "kind": "switch", "label": "H3 Sparse Attention", "default": OFF, "advanced": True},
        {"id": "h3_budget", "kind": "number", "label": "Video attention", "parent": "h3_sparse", "default": f"{budget:g}", "min": 0.01, "max": 1, "step": 0.01,
         "advanced": True},
    ]
    wanted = [(class_type, inputs) for flag, class_type, inputs in (
        ("h3_memory", H3_MEMORY, _node_defaults(info, H3_MEMORY)),
        ("h3_sparse", H3_SPARSE, {**sparse, "video_budget": float(chosen.get("h3_budget", budget))})) if chosen.get(flag) == ON]
    if wanted:
        sources = {tuple(node["inputs"]["model"]) for node in api.values() if node["class_type"] in H3_SAMPLING and isinstance(node["inputs"].get("model"), list)}
        for number, source in enumerate(sorted(sources)):
            link = list(source)
            for class_type, inputs in wanted:
                node_id = f"h3_{number}_{class_type}"
                api[node_id] = {"class_type": class_type, "inputs": {**inputs, "model": link}, "_meta": {"title": class_type}}
                link = [node_id, 0]
            for node in api.values():
                if node["class_type"] in H3_SAMPLING and node["inputs"].get("model") == list(source):
                    node["inputs"]["model"] = link
    return options


def _steps(api):
    """
    Where the steps are set, if just one sampler has them as a number: [node id, input], or None. A sampler fed sigmas
    (SamplerCustomAdvanced) has none of its own, so then its scheduler's (BasicScheduler) are the ones.
    """
    for takes in (lambda node: "positive" in node["inputs"], lambda node: node["class_type"].endswith("Scheduler")):
        found = [source for node_id, node in api.items() if "steps" in node["inputs"] and takes(node)
                 for source in [_number_source(api, node_id, "steps")] if source]
        if found:
            return found[0] if len(found) == 1 else None
    return None


def _after(api, options, chosen, info):
    """The LoRAs and the enhancer as chosen, and the upscaler on attention every install has."""
    loras = (info.get("LoraLoader") or info.get("LoraLoaderModelOnly") or {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
    for option in options:
        if option["kind"] != "strength":
            continue
        node = api.get(str(option["node"]))
        if not node:
            continue
        value = chosen.get(option["id"], option["default"])
        strength = float(value) if value != OFF else 0.0
        if option["id"] == "prompt_boost":
            node["inputs"].update(enabled=value != OFF, strength=strength if value != OFF else node["inputs"].get("strength", 1.0))
            continue
        entry = node["inputs"].get(option["entry"])
        if isinstance(entry, dict):
            entry = {**entry, "on": value != OFF, "strength": strength if value != OFF else entry.get("strength", 1)}
            # Saved under a folder of the author's, it may be installed without one.
            wanted = os.path.basename(str(entry["lora"]).replace("\\", "/"))
            if entry["lora"] not in loras:
                entry["lora"] = next((name for name in loras if os.path.basename(name.replace("\\", "/")) == wanted), entry["lora"])
            node["inputs"][option["entry"]] = entry
    for node in api.values():
        # Sage attention isn't on every install, and the workflow's own note says to use SDPA without it.
        if node["class_type"] == UPSCALER_LOADER and str(node["inputs"].get("attention_mode", "")).startswith("sage"):
            node["inputs"]["attention_mode"] = "sdpa"


def lora_problem(option, info, model=None):
    """
    Why a LoRA can't be switched on, or does nothing: its file isn't installed, or it was made for an earlier version
    of the model (core/lora_check.py). None if it's fine.
    """
    if option["kind"] != "strength" or option["id"] == "prompt_boost":
        return None
    loras = (info.get("LoraLoader") or info.get("LoraLoaderModelOnly") or {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
    name = option["label"].split(": ", 1)[1]
    found = next((lora for lora in loras if os.path.splitext(os.path.basename(lora.replace("\\", "/")))[0] == name), None)
    if not found:
        return f"Its file {name} isn't installed."
    return lora_check.problem(found, model)


def convert(ui, info, chosen=None, pipeline="image"):
    """The workflow made for this side (image, video or shorts) with these options. Returns (api, fields, model, options). Raises Unusable."""
    chosen = chosen or {}
    options = describe(ui, info, pipeline)
    api = sw.to_api(_before(ui, options, chosen), info, images=pipeline == "image", enhance=chosen.get("prompt_enhancer") == ON)
    # References that grow, such as Fibo 2.0's 16 or MiniMax H3's: a saved workflow wires one or two. Shorts gives one.
    if pipeline != "shorts":
        sw._grow_references(api, info)
    _after(api, options, chosen, info)
    steps = _steps(api)
    if steps:
        node_id, name = steps
        options.append({"id": "steps", "kind": "number", "label": "Steps", "default": str(api[node_id]["inputs"][name]), "min": 1, "max": MOST_STEPS, "step": 1,
                        "advanced": True})
        if "steps" in chosen:
            api[node_id]["inputs"][name] = int(float(chosen["steps"]))
    options += _tune(api, info, chosen)
    if pipeline == "image":
        return api, sw.find_image_fields(api, info), sw.image_model(api), options
    return api, sw.find_fields(api, info, needs_audio=pipeline == "shorts"), None, options


def offered(ui, info, pipeline="image"):
    """
    The options as the form shows them: each with a problem, if choosing the other way can't run. A way for the
    workflow to write its own prompt also says how many images it takes then: image to prompt needs its picture.
    """
    _, _, model, options = convert(ui, info, pipeline=pipeline)
    # A switch such as turbo has its own steps, CFG and sampler: they show the ones that go with it as it's set.
    flips = {}
    for option in options:
        problem = lora_problem(option, info, model)
        if not problem and option["kind"] == "switch":
            try:
                _, fields, model, flipped = convert(ui, info, {option["id"]: ON if option["default"] == OFF else OFF}, pipeline)
                flips[option["id"]] = flipped
                if option.get("role"):
                    found = sw.image_takes(fields, model) if pipeline == "image" else sw.takes(fields)
                    option["takes"] = {key: found[key] for key in ("min_images", "max_images")}
            except sw.Unusable as e:
                problem = str(e)
        if problem:
            option["problem"] = problem
    switches = {option["id"]: option for option in options if option["kind"] == "switch"}
    for target in options:
        if target["id"] not in FOLLOWERS or target.get("problem"):
            continue
        for switch_id, flipped in flips.items():
            other = next((found["default"] for found in flipped if found["id"] == target["id"]), None)
            if other and other != target["default"] and not switches[switch_id].get("problem"):
                flipped_to = ON if switches[switch_id]["default"] == OFF else OFF
                target["follows"] = {"switch": switch_id, switches[switch_id]["default"]: target["default"], flipped_to: other}
                break
    return options


def remake(workflow_path, chosen, pipeline=None):
    """
    The workflow a run makes with these options, from the copy kept when it was listed: (api, {field: (node, input)}).
    For ComfyClient and Shorts' AvatarGenerator. Raises RuntimeError, saying why.
    """
    stem = os.path.splitext(workflow_path)[0]
    try:
        with open(stem + ".map.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        offered_ = saved.get("_options") or []
        problem = check(offered_, chosen)
        if problem:
            raise RuntimeError(problem)
        with open(stem + ".ui.json", "r", encoding="utf-8") as f:
            ui = json.load(f)
        with open(stem + ".info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
    except OSError:
        raise RuntimeError("This workflow's options need it listed again: open the panel with ComfyUI on.")
    try:
        api, fields, _, _ = convert(ui, info, chosen, pipeline or saved.get("_pipeline") or "video")
    except sw.Unusable as e:
        raise RuntimeError(f"With these options the workflow can't run: {e}")
    return api, {field: tuple(value) for field, value in fields.items() if isinstance(value, list)}


def chosen_in_env():
    """The options the panel chose for this run, kept with it: COMFYUI_OPTIONS, as JSON."""
    try:
        found = json.loads(os.getenv("COMFYUI_OPTIONS") or "{}")
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def check(options, chosen):
    """Why these choices can't be made, or None."""
    known = {option["id"]: option for option in options}
    for key, value in chosen.items():
        option = known.get(key)
        if not option:
            return f"This workflow has no option {key}."
        if option.get("problem") and value != option["default"]:
            return f"{option['label']} can't be changed: {option['problem']}"
        if option["kind"] == "switch" and value not in (ON, OFF):
            return f"{option['label']} is on or off."
        if option["kind"] == "choice" and value not in option["choices"]:
            return f"{option['label']} is one of {', '.join(option['choices'])}."
        if option["kind"] == "text" and (not isinstance(value, str) or len(value) > MOST_TEXT):
            return f"{option['label']} is text of up to {MOST_TEXT} characters."
        if option["kind"] in ("strength", "number"):
            try:
                number = float(value)
            except ValueError:
                return f"{option['label']} needs a number."
            if (option["kind"] == "number" or value != OFF) and not option["min"] <= number <= option["max"]:
                return f"{option['label']} goes from {option['min']:g} to {option['max']:g}."
    return None


# --- Which LoRAs a run used, for the gallery ---

OUTPUTS = {sw.PREVIEW, *sw.IMAGE_SAVES, *sw.VIDEO_SAVES}
LORA_STRENGTHS = ("strength_model", "lora_strength", "strength")
_loras = {}


def _feeding_outputs(api):
    """The nodes an output reads from, however far back: a LoRA loader switched out of the way isn't one."""
    todo = [key for key, node in api.items() if isinstance(node, dict) and node.get("class_type") in OUTPUTS]
    seen = set()
    while todo:
        key = todo.pop()
        if key in seen or key not in api:
            continue
        seen.add(key)
        for value in (api[key].get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
                todo.append(str(value[0]))
    return seen


def _lora_label(name, strength):
    short = os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        return short
    return short if strength == 1 else f"{short} ({strength:g})"


def loras_in(api):
    """The LoRAs a workflow in API format runs with, as "name" or "name (strength)", in node order."""
    used = []
    for key in sorted(_feeding_outputs(api), key=lambda k: (len(k), k)):
        inputs = api[key].get("inputs") or {}
        name = inputs.get("lora_name")
        if isinstance(name, str) and name.lower() not in ("", "none"):
            strength = next((inputs[s] for s in LORA_STRENGTHS if s in inputs and not isinstance(inputs[s], list)), 1)
            if strength != 0:
                used.append(_lora_label(name, strength))
        # rgthree's Power Lora Loader: one {"on", "lora", "strength"} per row.
        for value in inputs.values():
            if isinstance(value, dict) and value.get("lora") and value.get("on") and value.get("strength", 1) != 0:
                used.append(_lora_label(value["lora"], value.get("strength", 1)))
    return list(dict.fromkeys(used))


def loras_used(workflow_path, chosen=None, pipeline=None):
    """The LoRAs a run used: its workflow with the options it was made with. [] if that can't be told."""
    if not workflow_path:
        return []
    stem = os.path.splitext(workflow_path)[0]
    try:
        stamp = max(os.path.getmtime(path) for path in (workflow_path, stem + ".ui.json") if os.path.exists(path))
    except ValueError:
        return []
    key = (workflow_path, stamp, json.dumps(chosen or {}, sort_keys=True), pipeline)
    if key not in _loras:
        try:
            api = remake(workflow_path, chosen or {}, pipeline)[0]
        except RuntimeError:
            # Listed before options were kept: the workflow as saved.
            try:
                with open(workflow_path, "r", encoding="utf-8") as f:
                    api = json.load(f)
            except (OSError, ValueError):
                api = {}
        _loras[key] = loras_in(api) if isinstance(api, dict) else []
    return _loras[key]

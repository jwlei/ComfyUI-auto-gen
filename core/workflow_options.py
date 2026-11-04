"""
What a run can change in an image workflow besides its prompt, images and size, and making the workflow with
those changes. Community workflows such as LatentHeart's Krea2 SFW/NSFW one are built for switching parts on
and off in ComfyUI's editor: groups that rgthree's Fast Groups Bypasser turns on and off (2nd pass, upscaler,
grain, sharpen...), a choice on a group's subgraph (the upscaler's resolution), LoRAs in rgthree's Power Lora
Loader, and the Krea2T Enhancer's prompt-adherence boost. ComfyUI's own templates put theirs on a subgraph's
face instead: Krea-2's enable_lora with its LoRA and strength, Qwen-Image-Edit's enable_turbo_mode. And any
workflow whose sampler's steps can be found gets Steps. Each becomes an option on the form.

The editor bypasses a group by setting its nodes' mode, and that changes which nodes the API gets, so a run
with options makes its workflow again from what was saved (core/saved_workflows.py keeps a copy for it).

Each option is {id, kind, label, default, ...}, and a run's choice is a string:
    switch    "1" or "0": a group on or off, or a switch on a subgraph's face
    choice    one of its choices: a value on a subgraph, shown with the switch it goes with
    number    a number: a strength that goes with a switch, or the sampler's steps
    strength  "0" for off, or the strength: a LoRA, or the Krea2T Enhancer
"""
import copy
import os
import re

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
FACE_SKIP = re.compile(r"prompt|enhance|refine|thinking|max_tok|max_length|width|height|seed|unet|clip|vae|ckpt|trigger", re.I)
# Words a switch's name has that say nothing about what it switches.
SWITCH_WORDS = {"enable", "enabled", "use", "mode", "on"}
MOST_STEPS = 150


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
    widgets = sw._widgets_of(instance)
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
    widgets = sw._widgets_of(instance)
    faced = [(slot, entry, entry.get("label") or entry["name"]) for slot, entry in enumerate(subgraph.get("inputs", []))
             if entry["name"] in widgets and not FACE_SKIP.search(entry.get("label") or entry["name"])]
    switches = [{"id": f"face_{_slug(label)}", "kind": "switch", "label": _words(label), "default": ON if widgets[entry["name"]] else OFF,
                 "node": instance["id"], "input": entry["name"]}
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


def describe(ui, info=None):
    """The options this workflow offers, as saved. A problem with one is for the caller to find, by trying it."""
    options = []
    subgraphs = _subgraphs(ui)
    for node in ui["nodes"]:
        if node["type"] in subgraphs and not node.get("mode"):
            options += _face_settings(node, subgraphs[node["type"]], info or {})
    for group in _switched_groups(ui):
        title = group.get("title") or "Group"
        if PROMPT_GROUPS.search(title) or SIZE_GROUPS.search(title):
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
        wanted = OFF if PROMPT_GROUPS.search(title) else chosen.get(option["id"], option["default"]) if option else None
        if wanted is None:
            continue
        for node in ui["nodes"]:
            if _inside(node, group) and node["type"] not in sw.NOTES + sw.EDITOR_ONLY:
                _set_mode(ui, node, 0 if wanted == ON else BYPASSED)
    nodes = {node["id"]: node for node in ui["nodes"]}
    for option in options:
        # A value on a subgraph's face: a choice, a switch or a number there.
        if "input" in option and option["id"] in chosen and option["node"] in nodes:
            node = nodes[option["node"]]
            names = [entry["widget"]["name"] for entry in node.get("inputs", []) if "widget" in entry]
            values = node.get("widgets_values")
            value = chosen[option["id"]]
            value = value == ON if option["kind"] == "switch" else float(value) if option["kind"] == "number" else value
            if isinstance(values, list) and option["input"] in names and names.index(option["input"]) < len(values):
                values[names.index(option["input"])] = value
    return ui


def _steps(api):
    """Where the sampler's steps are set, if just one sampler has them as a number: [node id, input], or None."""
    found = []
    for node_id, node in api.items():
        steps = node["inputs"].get("steps")
        if "positive" not in node["inputs"]:
            continue
        if isinstance(steps, int) and not isinstance(steps, bool):
            found.append([node_id, "steps"])
        elif isinstance(steps, list) and type(api.get(steps[0], {}).get("inputs", {}).get("value")) is int:
            found.append([steps[0], "value"])
    return found[0] if len(found) == 1 else None


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


def lora_problem(option, info):
    """Why a LoRA can't be switched on: its file isn't installed. None if it can."""
    if option["kind"] != "strength" or option["id"] == "prompt_boost":
        return None
    loras = (info.get("LoraLoader") or info.get("LoraLoaderModelOnly") or {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
    name = option["label"].split(": ", 1)[1]
    if not any(os.path.splitext(os.path.basename(lora.replace("\\", "/")))[0] == name for lora in loras):
        return f"Its file {name} isn't installed."
    return None


def convert(ui, info, chosen=None):
    """The workflow made for the Image side with these options. Returns (api, fields, model, options). Raises Unusable."""
    chosen = chosen or {}
    options = describe(ui, info)
    api = sw.to_api(_before(ui, options, chosen), info, images=True)
    # Fibo 2.0 edits with up to 16 references, of which a saved workflow wires one.
    sw._grow_references(api, info)
    _after(api, options, chosen, info)
    steps = _steps(api)
    if steps:
        node_id, name = steps
        options.append({"id": "steps", "kind": "number", "label": "Steps", "default": str(api[node_id]["inputs"][name]), "min": 1, "max": MOST_STEPS, "step": 1})
        if "steps" in chosen:
            api[node_id]["inputs"][name] = int(float(chosen["steps"]))
    return api, sw.find_image_fields(api, info), sw.image_model(api), options


def offered(ui, info):
    """The options as the form shows them: each with a problem, if choosing the other way can't run."""
    _, _, _, options = convert(ui, info)
    for option in options:
        problem = lora_problem(option, info)
        if not problem and option["kind"] == "switch":
            try:
                convert(ui, info, {option["id"]: ON if option["default"] == OFF else OFF})
            except sw.Unusable as e:
                problem = str(e)
        if problem:
            option["problem"] = problem
    return options


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
        if option["kind"] in ("strength", "number"):
            try:
                number = float(value)
            except ValueError:
                return f"{option['label']} needs a number."
            if (option["kind"] == "number" or value != OFF) and not option["min"] <= number <= option["max"]:
                return f"{option['label']} goes from {option['min']:g} to {option['max']:g}."
    return None

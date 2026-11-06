"""The panel's Image side: sets of images from text or from an image, with prompts the LLM writes or you do."""
import json
import os
import secrets
import sys
from types import SimpleNamespace

from flask import abort, request, send_file
from PIL import Image

from core import seeds, workflow_options
from core.saved_workflows import display_label, shorthand
from panel.server import (read_outcome, IMAGE_TIMINGS, PIPELINES, _runs_cache, active_runs, app, busy_runs, form_options, password_required, run_queue, saved_at, queue_form_snapshot,
                          saved_workflows)
from panel.video import clean_inputs, queue_label, shot_seeds
from pipelines.image import media, models, regions
from pipelines.image import planner as image_planner
from pipelines.image.run_eta import RunEta
from pipelines.image.run_state import STAGES, RunState, list_runs
from pipelines.video import media as video_media
from pipelines.video import paths as video_paths
from pipelines.video import planner

COMMAND = [sys.executable, "-m", "pipelines.image.main"]
MOST_IMAGES_IN = 20


def option(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags[:-1] else default


def chosen_workflow(form, image_count=None):
    """
    What makes a run use the workflow chosen on the form, and the images it takes. Left on Automatic, what you give it
    decides: text alone goes to a workflow that makes images from text, and an image to one that edits. image_count is
    None when that isn't known yet, as for the estimate. Returns (env, takes, error).
    """
    entries = [entry for entry in saved_workflows() if entry.get("pipeline") == "image"]
    name = form.get("workflow")
    if name:
        entry = next((entry for entry in entries if entry["name"] == name), None)
        if not entry:
            return None, None, f"ComfyUI has no saved workflow called {display_label(name)} any more."
        if "problem" in entry:
            return None, None, f"{display_label(entry['label'])} can't run: {entry['problem']}"
        return {"COMFYUI_WORKFLOW": entry["path"]}, entry["takes"], None
    entry = automatic([entry for entry in entries if "problem" not in entry], image_count or 0, form.get("image_mode") or "edit")
    if entry:
        return {"COMFYUI_WORKFLOW": entry["path"]}, entry["takes"], None
    if image_count is None:
        return {}, {"min_images": 0, "max_images": 1, "size": True}, None
    wanted = "edits an image" if image_count else "makes images from text"
    why = "; ".join(f"{display_label(entry['label'])}: {entry['problem']}" for entry in entries if "problem" in entry)
    return None, None, (f"No workflow that {wanted} can run. {why}" if why
                        else f"No workflow that {wanted} is saved in ComfyUI with a name starting image_.")


def automatic(usable, image_count, mode="edit"):
    """
    The workflow Automatic picks, in name order. One made for what you gave it comes first: text alone goes to one that
    takes no image, and an image to one that needs one, so Fibo 2.0, which does both, is for when nothing else fits.
    A new picture from references, or a change to part of an image, needs a model that can (models.py).
    """
    can = {"compose": models.COMPOSE_SWITCH, "areas": models.REGION_EDITS, "expand": models.REGION_EDITS}.get(mode)
    if can is not None:
        usable = [entry for entry in usable if entry["takes"].get("model") in can]
    fits = [entry for entry in usable if entry["takes"]["min_images"] <= image_count <= entry["takes"]["max_images"]]
    made_for = [entry for entry in fits if (entry["takes"]["min_images"] >= 1 if image_count else entry["takes"]["max_images"] == 0)]
    return (made_for or fits or [None])[0]


def chosen_options(form, takes):
    """The workflow's own options as the form sets them, as --option flags. Returns (flags, error)."""
    chosen, error = form_options(form, takes)
    return (None, error) if error else ([flag for key, value in chosen.items() for flag in ("--option", f"{key}={value}")], None)


# What the LLM select says when the workflow writes its own prompts (core/workflow_options.py).
ROLE_NAMES = {"prompt_enhancer": "the workflow's prompt enhancer", "image_to_prompt": "the workflow, from the image"}


def role_of(flags):
    """Which way of writing its own prompt the workflow was asked for, if any."""
    return next((value.split("=")[0] for flag, value in zip(flags, flags[1:])
                 if flag == "--option" and value.split("=")[0] in workflow_options.ROLES), None)


# What can be done with the images given, by the form's image_mode, and what a model needs to do it.
IMAGE_MODES = ("edit", "compose", "areas", "expand")


def mode_problem(mode, takes, flags):
    """Why the chosen workflow can't do this with the images, or None."""
    model, label = takes.get("model"), models.label(takes.get("model"))
    if mode == "compose" and model not in models.COMPOSE_SWITCH:
        return f"{label} can't make a new picture from references: choose Qwen-Image 2.1's edit workflow, or Automatic."
    if mode in ("areas", "expand") and model not in models.REGION_EDITS:
        return f"{label} can't change part of an image or expand it: choose Qwen-Image 2.1's edit workflow, or Automatic."
    if "--transparent" in flags and model not in models.TRANSPARENT:
        return f"{label} doesn't make transparent images: choose a Qwen-Image 2.1 workflow."
    return None


def region_flags(form, files, source_size):
    """
    The flags for a change to painted areas or an expansion, from the form, with what they need saved under inputs.
    Returns (flags, error).
    """
    mode = form.get("image_mode")
    if mode == "areas":
        mask = files.get("areas_mask")
        try:
            areas = json.loads(form.get("areas") or "[]")
        except ValueError:
            return None, "The painted areas didn't come through: paint them again."
        if not (mask and mask.filename) or not isinstance(areas, list) or not areas:
            return None, "Paint the area to change on image 1 first."
        for area in areas:
            if not isinstance(area, dict) or (area.get("action") or "change") not in regions.ACTIONS:
                return None, "The painted areas didn't come through: paint them again."
            if not (area.get("text") or "").strip() and not (len(areas) == 1 and area.get("action") == "remove"):
                return None, "Say what goes in each painted area. Only Remove can go without."
        os.makedirs(video_paths.INPUTS_DIR, exist_ok=True)
        stem = os.path.join(video_paths.INPUTS_DIR, secrets.token_hex(8))
        mask.save(stem + "_mask.png")
        with open(stem + "_areas.json", "w", encoding="utf-8") as f:
            json.dump({"mask": stem + "_mask.png", "areas": [{"colour": str(area.get("colour") or "red"), "action": area.get("action") or "change",
                                                               "text": str(area.get("text") or "")} for area in areas]}, f)
        return ["--areas", stem + "_areas.json"], None
    if mode == "expand":
        try:
            sides = [float(form.get(f"expand_{side}") or 0) for side in ("left", "top", "right", "bottom")]
        except ValueError:
            return None, "Expand by whole numbers."
        if min(sides) < 0 or not any(sides):
            return None, "Say how far to expand it on at least one side."
        if form.get("expand_units") == "percent":
            sides = [value / 100 * (source_size[0] if index % 2 == 0 else source_size[1]) for index, value in enumerate(sides)]
        return ["--expand", ",".join(str(round(value)) for value in sides)], None
    return [], None


def model_problem(flags, takes, source):
    """Why the chosen workflow's model shouldn't make this: too big, or too many prompts for the LLM. None if it's fine."""
    model = takes.get("model")
    if "--areas" in flags or "--expand" in flags:
        return models.count_problem(model, int(option(flags, "--count")), True)
    if not takes.get("size"):
        size = None
    elif not source and media.FROM_IMAGE in (option(flags, "--aspect"), option(flags, "--megapixels")) and not option(flags, "--size"):
        size = None  # Comes from an image not chosen yet, which is held to the model's limit when it is
    else:
        compose = "--compose" in flags
        size = media.pick_size(option(flags, "--size"), option(flags, "--aspect"), option(flags, "--megapixels"), source,
                               models.most_megapixels(model, compose))
        return models.size_problem_for(model, *size, compose) or models.count_problem(model, int(option(flags, "--count")), "--direct" in flags)
    return models.count_problem(model, int(option(flags, "--count")), "--direct" in flags)


def resume_argv(state, redo=None):
    argv = [*COMMAND, "--resume", state.id]
    return argv + ["--redo", str(redo)] if redo else argv


def image_files(state):
    """What the gallery shows of a run. The time makes a remade image load anew rather than from the browser's cache."""
    files = []
    for position, shot in enumerate(state.shots or [], 1):
        if shot.get("file"):
            version = f"{os.stat(state.image_path(shot)).st_mtime_ns}-{shot['number']}"
            files.append({"position": position, "title": shot["title"], "version": version})
    return files


def describe_run(state):
    """A saved run, as the panel lists it."""
    done, total = len(state.done), len(state.shots)
    parts = [f"{done} of {total} images" if done else f"Prompts ready, {total} {'image' if total == 1 else 'images'}", f"{state.width}x{state.height}"]
    env = {"COMFYUI_WORKFLOW": state.workflow} if state.workflow else {}
    return {"id": state.id, "pipeline": "image", "label": state.title or state.brief, "detail": " · ".join(parts),
            "model": shorthand(state.workflow, "image"), "outcome": read_outcome(state.folder), "loras": workflow_options.loras_used(state.workflow, state.options, "image"),
            "when": saved_at(state.path), "clips": done, "finished": state.finished, "total": total, "images": image_files(state),
            "size": f"{state.width}x{state.height}", "estimate": RunEta(resume_argv(state), env, IMAGE_TIMINGS).plan()["total"]}


# What a size falls back on with nothing to go by.
FALLBACK_ASPECT = "1:1"


def size_flags(form):
    """The size flags from the form. Returns (flags, error)."""
    aspect = form.get("aspect") or media.FROM_IMAGE
    # An exact size is typed in, or one of the screens on the list.
    size = (form.get("size") or "").strip().lower() if aspect == "custom" else aspect if media.parse_size(aspect) else None
    if aspect == "custom" and not size:
        # Exact, with nothing typed (the box isn't remembered across a reload): square, one megapixel.
        return ["--aspect", FALLBACK_ASPECT, "--megapixels", str(media.DEFAULT_MEGAPIXELS)], None
    if aspect == "custom" and not media.parse_size(size):
        return [], f"An exact size must look like 1920x1080, with each side {media.MIN_EDGE}-{media.MAX_EDGE}."
    if size:
        return ["--size", size], None
    if aspect != media.FROM_IMAGE and not media.parse_aspect(aspect):
        return [], "Unknown aspect ratio."
    megapixels = form.get("megapixels") or str(media.DEFAULT_MEGAPIXELS)
    if megapixels != media.FROM_IMAGE and not media.parse_megapixels(megapixels):
        return [], f"Megapixels must be a number up to {media.MAX_MEGAPIXELS:g}."
    return ["--aspect", aspect, "--megapixels", megapixels], None


def build_options(form):
    """Turns the form's options into main.py's flags. Returns (flags, error). What it's of isn't needed for that."""
    try:
        count = int(form.get("count") or 1)
    except ValueError:
        return None, "How many images must be a whole number."
    if not 1 <= count <= image_planner.MAX_IMAGES:
        return None, f"A set has 1-{image_planner.MAX_IMAGES} images."
    flags = ["--count", str(count)]
    try:
        seed = seeds.parse(form.get("seed"))
    except ValueError:
        return None, f"The seed must be a whole number from 0 to {seeds.MOST}, or empty for random."
    if seed is not None:
        flags += ["--seed", str(seed)]

    mode = form.get("prompt_mode") or "llm"
    if mode not in ("llm", "direct"):
        return None, "Unknown prompt mode."
    if mode == "direct":
        flags.append("--direct")
    else:
        llm = form.get("llm") or planner.default_llm()
        if llm in workflow_options.ROLES:
            # The workflow writes them from yours, on its own: no LLM is asked.
            flags += ["--direct", "--option", f"{llm}={workflow_options.ON}"]
            # Its prompts can be read first: it writes them in ComfyUI, and the run stops there.
            if llm == "prompt_enhancer" and form.get("review"):
                flags.append("--enhance-only")
        elif llm not in dict(planner.LLMS):
            return None, "Unknown LLM."
        else:
            flags += ["--llm", llm]
            if form.get("review"):
                flags.append("--plan-only")

    if form.get("try_first"):
        flags.append("--first-only")

    mode = form.get("image_mode") or "edit"
    if mode not in IMAGE_MODES:
        return None, "Unknown way to use the images."
    if mode == "compose":
        flags.append("--compose")
    if form.get("transparent") and mode not in ("areas", "expand"):
        flags.append("--transparent")
    if form.get("detail") == "full":
        flags += ["--reference-detail", "full"]
    for value in form.getlist("source_seed") if hasattr(form, "getlist") else []:
        if str(value).strip().isdigit():
            flags += ["--source-seed", str(int(value))]
    if mode in ("areas", "expand"):
        # Their prompt is written from what's painted, or says what the new parts show: no LLM writes it.
        flags = [flag for flag in flags if flag not in ("--plan-only", "--enhance-only")]
        if "--llm" in flags:
            index = flags.index("--llm")
            del flags[index:index + 2]
        flags = [flag for flag in flags if flag != "--direct"]
        return flags, None

    flags_, error = size_flags(form)
    return (None, error) if error else (flags + flags_, None)


@app.route("/api/estimate/image", methods=["POST"], endpoint="image_estimate")
@password_required
def estimate():
    """What a run with the form's current options should take. Starts nothing."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}
    count = request.form.get("image_count")
    env, takes, error = chosen_workflow(request.form, int(count) if (count or "").isdigit() else None)
    error = error or model_problem(flags, takes, None)
    if error:
        return {"ok": False, "error": error}
    # Reviewing first only writes the prompts now, but what you're deciding on is the whole set.
    return {"ok": True, **RunEta([arg for arg in flags if arg != "--plan-only"], env, IMAGE_TIMINGS).plan()}


def form_uploads(files):
    """The images chosen in the form's boxes (image, image_2...), in order."""
    field = lambda number: "image" if number == 1 else f"image_{number}"
    return [files[field(number)] for number in range(1, MOST_IMAGES_IN + 1) if field(number) in files and files[field(number)].filename]


@app.route("/generate/image", methods=["POST"], endpoint="image_generate")
@password_required
def generate():
    """Starts a set of images, or queues it behind what's in progress."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    prompt = (request.form.get("prompt") or "").strip()
    mode = request.form.get("image_mode") or "edit"
    if not prompt and mode not in ("areas", "expand"):
        return {"ok": False, "error": "Say what you want." if "--direct" not in flags else "Write the prompt to use."}, 400
    llm = option(flags, "--llm")
    role = role_of(flags)
    if llm and not planner.llm_configured(llm):
        return {"ok": False, "error": f"{llm} has no credentials. See .env.example, or use your own prompt."}, 400

    uploads = form_uploads(request.files)
    env, takes, error = chosen_workflow(request.form, len(uploads))
    if error:
        return {"ok": False, "error": error}, 400
    if role:
        found = next((option for option in takes.get("options") or [] if option["id"] == role), None)
        if not found:
            return {"ok": False, "error": f"This workflow has no {'prompt enhancer' if role == 'prompt_enhancer' else 'image to prompt'} of its own: choose Gemini or Claude, or another workflow."}, 400
        if found.get("problem"):
            return {"ok": False, "error": f"{found['label']} can't run: {found['problem']}"}, 400
        # Image to prompt needs the picture it describes.
        takes = {**takes, **found.get("takes", {})}
    if len(uploads) < takes["min_images"]:
        return {"ok": False, "error": "This workflow edits an image: choose one." if takes["min_images"] == 1 else f"This workflow needs {takes['min_images']} images."}, 400
    if len(uploads) > takes["max_images"]:
        return {"ok": False, "error": "This workflow makes images from text alone: leave the image out." if not takes["max_images"] else f"This workflow takes at most {takes['max_images']} images."}, 400
    try:
        images = [video_media.load_image(upload.stream) for upload in uploads]
    except RuntimeError:
        return {"ok": False, "error": "That file isn't an image this can read."}, 400
    if mode in ("compose", "areas", "expand") and not images:
        return {"ok": False, "error": "Choose the image first."}, 400
    problem = mode_problem(mode, takes, flags)
    if problem:
        return {"ok": False, "error": problem}, 400
    extra, error = region_flags(request.form, request.files, images[0].size if images else None)
    if error:
        return {"ok": False, "error": error}, 400
    flags += extra
    # What made the images given, so an edit never starts from the noise that made one (core/seeds.py avoid).
    for upload in uploads:
        flags += [flag for seed in sorted(media.seeds_in(upload.stream)) for flag in ("--source-seed", str(seed))]
    problem = model_problem(flags, takes, images[0].size if images else None)
    if problem:
        return {"ok": False, "error": problem}, 400
    option_flags, problem = chosen_options(request.form, takes)
    if problem:
        return {"ok": False, "error": problem}, 400
    flags += option_flags

    # A change to part of an image is a new one each time: what's painted isn't something to look for in a stopped run.
    state = None if extra or request.form.get("queue_edit_id") else run_to_continue(prompt, flags, env, images)
    if state:
        if "--plan-only" in flags:
            return {"ok": True, "queued": False, "shots": state.id, "continued": state.title}
        adopt_size(state, flags, images, takes.get("model"))
        # Options change how the rest are made, not their prompts: what's paid for stays.
        state.options = dict(value.split("=", 1) for flag, value in zip(flags, flags[1:]) if flag == "--option") or None
        state.save()
        reply = queue_saved_run(state)
        return reply if isinstance(reply, tuple) else {**reply, "continued": state.title}

    # Kept in a folder of its own, under a name of our own: queued, it may wait for hours.
    # The painted areas wait with the images: queued, it may be hours before they're read.
    region_files = []
    if "--areas" in extra:
        areas_file = extra[extra.index("--areas") + 1]
        region_files = [areas_file, areas_file.replace("_areas.json", "_mask.png")]
    clean_inputs(keep=region_files)
    image_paths = [os.path.join(video_paths.INPUTS_DIR, f"{secrets.token_hex(8)}.png") for _ in images]
    for image, path in zip(images, image_paths):
        media.save_source(image, path)

    argv = [*COMMAND, *flags, *[flag for path in image_paths for flag in ("--image", path)], *(["--", prompt] if prompt else [])]
    plan_only = "--plan-only" in flags or "--enhance-only" in flags
    count = int(option(flags, "--count"))
    what = {"compose": "a new picture from the images", "areas": "changes painted areas", "expand": "expands the image"}.get(
        mode, "edits an image" if images else "from text")
    writer = "its own instruction" if mode in ("areas", "expand") else ROLE_NAMES.get(role) or llm or "your prompt, word for word"
    detail = [f"{count} {'image' if count == 1 else 'images'}", what, writer, *(["review the prompts first"] if plan_only else [])]
    if env.get("COMFYUI_WORKFLOW"):
        detail.append(os.path.splitext(os.path.basename(env["COMFYUI_WORKFLOW"]))[0])
    count_label = f"{count} {'image' if count == 1 else 'images'}"
    chosen_size = None if mode in ("areas", "expand") else media.pick_size(
        option(flags, "--size"), option(flags, "--aspect"), option(flags, "--megapixels"),
        images[0].size if images else None, models.most_megapixels(takes.get("model"), "--compose" in flags))
    item = {"pipeline": "image", "label": f"{count_label}: {queue_label(prompt)}" if prompt else count_label, "prompt": prompt, "detail": " · ".join(detail), "argv": argv, "env": env,
            "resolution": f"{chosen_size[0]}x{chosen_size[1]}" if chosen_size else ("Expanded source size" if mode == "expand" else "Source image size"),
            "image": image_paths[0] if image_paths else None, "references": image_paths[1:] + region_files, "plan_only": plan_only, "first_only": "--first-only" in flags,
            "form": queue_form_snapshot(),
            "form_files": {**{("image" if i == 0 else f"image_{i + 1}"): path for i, path in enumerate(image_paths)},
                           **({"areas_mask": region_files[1]} if region_files else {})},
            "estimate": RunEta(argv, env, IMAGE_TIMINGS).plan()["total"]}
    queued = run_queue.add(item)
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


def run_to_continue(prompt, flags, env, images):
    """
    The stopped run Generate should finish rather than pay for the same prompts again:
    the same brief, images, count, workflow and way of writing them.
    """
    busy = busy_runs()
    inputs = video_media.inputs_fingerprint(images, [])
    count, direct = int(option(flags, "--count")), "--direct" in flags
    for state in reversed(list_runs()):
        if (state.id not in busy and state.shots and not state.finished and state.inputs == inputs and state.brief == prompt
                and state.count == count and state.direct == direct and state.workflow == env.get("COMFYUI_WORKFLOW")):
            return state
    return None


def adopt_size(state, flags, images, model=None):
    """The size chosen now, for what's still to be made. What has been made stays as it is."""
    state.width, state.height = media.pick_size(option(flags, "--size"), option(flags, "--aspect"), option(flags, "--megapixels"),
                                                images[0].size if images else None, models.MOST_MEGAPIXELS.get(model))
    state.save()


def queue_saved_run(state, redo=None):
    argv = resume_argv(state, redo)
    what = f"Redo image {redo}" if redo else "Continue"
    env = {"COMFYUI_WORKFLOW": state.workflow} if state.workflow else {}
    item = {"pipeline": "image", "label": queue_label(state.title or state.brief), "detail": what, "argv": argv, "env": {},
            "resume": state.id, "estimate": RunEta(argv, env, IMAGE_TIMINGS).plan()["total"]}
    queued = run_queue.add(item)
    _runs_cache["at"] = 0.0
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


def load_idle_run(run_id, allow_pending=False):
    """The saved run, or the reply saying why it can't be touched now."""
    state = RunState.load(run_id)
    if not state or not state.shots:
        return None, ({"ok": False, "error": "That run is no longer saved."}, 404)
    if run_id in (active_runs() if allow_pending else busy_runs()):
        return None, ({"ok": False, "error": "That run is going or queued. Stop it or take it out of the queue first."}, 409)
    return state, None


@app.route("/image-runs/<run_id>")
@password_required
def image_run_detail(run_id):
    """A set's images and prompts, for reading and changing them."""
    state = RunState.load(run_id)
    if not state or not state.shots:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    with run_queue.lock:
        busy = run_id in active_runs()
        pending = not busy and run_id in busy_runs()
    return {"ok": True, "id": state.id, "title": state.title, "brief": state.brief, "llm": state.llm, "direct": state.direct,
            "busy": busy, "pending": pending, "finished": state.finished, "is_image": True,
            "size": f"{state.width}x{state.height}", "image_count": state.image_count, "settings": settings_of(state),
            "shots": [{"title": shot["title"], "prompt": shot["prompt"], "rendered": bool(shot.get("file")),
                       "seed": shot.get("seed"), "used_seed": shot.get("used_seed"), "enhanced_prompt": shot.get("enhanced_prompt"),
                       "version": int(os.path.getmtime(state.image_path(shot))) if shot.get("file") else 0} for shot in state.shots]}


def settings_of(state):
    """What a set was made with, as the form sets it: for "the same again", to change before pressing Generate."""
    workflow = next((w["name"] for w in saved_workflows() if state.workflow and w.get("path") and os.path.abspath(w["path"]) == os.path.abspath(state.workflow)), "")
    return {"prompt": state.brief, "direct": state.direct, "count": len(state.shots or []) or state.count, "llm": state.llm,
            "workflow": workflow, "size": f"{state.width}x{state.height}", "options": state.options or {},
            "inputs": sum(1 for path in state.input_paths if os.path.exists(path))}


@app.route("/image-runs/<run_id>/inputs/<int:number>")
@password_required
def image_input(run_id, number):
    """An image a set was given, for making it again."""
    state = RunState.load(run_id)
    if not state or not 1 <= number <= len(state.input_paths) or not os.path.exists(state.input_paths[number - 1]):
        abort(404)
    return send_file(state.input_paths[number - 1], mimetype="image/png")


def shot_of(run_id, position):
    state = RunState.load(run_id)
    if not state or not 1 <= position <= len(state.shots or []) or not state.shots[position - 1].get("file"):
        abort(404)
    return state, state.shots[position - 1]


@app.route("/image-runs/<run_id>/files/<int:position>")
@password_required
def image_file(run_id, position):
    state, shot = shot_of(run_id, position)
    return send_file(state.image_path(shot), conditional=True, as_attachment="download" in request.args)


@app.route("/image-runs/<run_id>/thumbnails/<int:position>")
@password_required
def image_thumbnail(run_id, position):
    state, shot = shot_of(run_id, position)
    return send_file(media.thumbnail(state.image_path(shot), state.thumbnail_path(shot)), mimetype="image/jpeg", max_age=3600)


@app.route("/image-runs/<run_id>/shots", methods=["POST"])
@password_required
def save_image_shots(run_id):
    """Saves changed prompts. An image already made stays as it is until you redo it."""
    with run_queue.lock:
        state, refused = load_idle_run(run_id, allow_pending=True)
        if refused:
            return refused
        body = request.get_json(silent=True) or {}
        prompts = body.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != len(state.shots) or not all(isinstance(p, str) and p.strip() for p in prompts):
            return {"ok": False, "error": "Every image needs a prompt."}, 400
        chosen, error = shot_seeds(body, len(state.shots))
        if error:
            return {"ok": False, "error": error}, 400
        for shot, prompt, seed in zip(state.shots, prompts, chosen):
            shot["prompt"] = prompt.strip()
            shot["seed"] = seed
        state.save()
    return {"ok": True}


@app.route("/image-runs/<run_id>/redo/<int:position>", methods=["POST"])
@password_required
def redo_image(run_id, position):
    """Makes one image again, on its seed if one is set or else a new one, leaving the others as they are."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    if not 1 <= position <= len(state.shots):
        return {"ok": False, "error": "That image isn't in this set."}, 400
    return queue_saved_run(state, redo=position)


@app.route("/image-runs/<run_id>/delete/<int:position>", methods=["POST"])
@password_required
def delete_image(run_id, position):
    """Takes one image and its prompt out of the set, so a Continue doesn't make it again. The last one takes the set with it."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    if not 1 <= position <= len(state.shots):
        return {"ok": False, "error": "That image isn't in this set."}, 400
    shot = state.shots[position - 1]
    if len(state.shots) == 1:
        state.remove()
        images = []
    else:
        if shot.get("file"):
            os.remove(state.image_path(shot))
        del state.shots[position - 1]
        state.save()
        images = image_files(state)
    _runs_cache["at"] = 0.0
    return {"ok": True, "images": images}


def continue_run(run_id):
    """Makes what's left of a planned or stopped set, or queues that behind the run in progress."""
    state, refused = load_idle_run(run_id)
    return refused or queue_saved_run(state)


def discard_run(run_id):
    """
    Gives up on what a stopped set has still to make. The images it did make stay, in Output, as a set of those;
    one that made none goes altogether.
    """
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    made = state.done
    if made:
        state.shots = made
        state.save()
    else:
        state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


UPSCALE = [sys.executable, "-m", "pipelines.image.upscale"]
UPSCALE_SECONDS = 180  # About what SeedVR2 takes for one image on this card
FAST_UPSCALE_SECONDS = 20  # Superscale-SP: seconds, as Kokon measured it (13 s for 1472² on a 5070 Ti)
UPSCALE_RESOLUTIONS = (720, 1080, 1440, 2160)  # SeedVR2's shorter side, as the Krea 2 workflow's 720p, 1080p, 1440p and 4K
UPSCALE_MOST_SIDE = 1440  # The shorter side of an image that can still be upscaled, as the viewer's button


@app.route("/image-runs/<run_id>/upscale/<int:position>", methods=["POST"])
@password_required
def upscale_image(run_id, position):
    """Queues one image of a set to be made bigger with SeedVR2, as a new image in the set beside it."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    if not 1 <= position <= len(state.shots) or not state.shots[position - 1].get("file"):
        return {"ok": False, "error": "That image isn't made yet."}, 400
    try:
        with Image.open(state.image_path(state.shots[position - 1])) as image:
            shorter = min(image.size)
    except OSError:
        shorter = 0
    if shorter > UPSCALE_MOST_SIDE:
        return {"ok": False, "error": f"It's already {shorter}px on its shorter side: images past {UPSCALE_MOST_SIDE}px aren't upscaled."}, 400
    method = request.args.get("method") or (request.get_json(silent=True) or {}).get("method") or "seedvr2"
    if method not in ("seedvr2", "fast"):
        return {"ok": False, "error": "Unknown way to upscale."}, 400
    try:
        strength = float(request.args.get("strength") or (request.get_json(silent=True) or {}).get("strength") or 1)
    except ValueError:
        return {"ok": False, "error": "Strength is a number from 0 to 1."}, 400
    try:
        resolution = int(request.args.get("resolution") or 0) if method == "seedvr2" else 0
    except ValueError:
        return {"ok": False, "error": "Resolution is a number of pixels."}, 400
    if resolution and resolution not in UPSCALE_RESOLUTIONS:
        return {"ok": False, "error": "Unknown resolution."}, 400
    if resolution and resolution <= shorter:
        return {"ok": False, "error": f"It's already {shorter}px on its shorter side: pick a bigger resolution."}, 400
    how = (f"SeedVR2, {resolution}p" if resolution else "SeedVR2, twice the size") if method == "seedvr2" else f"Superscale-SP, twice the size, strength {round(strength * 100)}%"
    item = {"pipeline": "image", "label": queue_label(f"Upscale: {state.shots[position - 1]['title']}"), "detail": how,
            "argv": [*UPSCALE, "--resume", state.id, "--image", str(position), "--method", method, "--strength", f"{min(1, max(0, strength)):g}",
                     *(["--resolution", str(resolution)] if resolution else [])],
            "env": {}, "resume": state.id, "estimate": UPSCALE_SECONDS if method == "seedvr2" else FAST_UPSCALE_SECONDS}
    queued = run_queue.add(item)
    _runs_cache["at"] = 0.0
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


@app.route("/image-runs/<run_id>/delete", methods=["POST"])
@password_required
def delete_image_set(run_id):
    """Deletes a whole set from Output: its prompts, and the images it made."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


def context():
    return {
        "image_llms": [{"name": name, "label": label, "ready": planner.llm_configured(name)} for name, label in planner.LLMS],
        "image_stages": STAGES,
        "image_most": image_planner.MAX_IMAGES,
        "image_aspects": media.ASPECTS,
        "image_megapixels": media.MEGAPIXELS,
        "image_size_step": media.SIZE_STEP,
        "image_desktop_sizes": media.DESKTOP_SIZES,
        "image_phone_sizes": media.PHONE_SIZES,
        "image_models": models.for_the_page(),
    }


PIPELINES["image"] = SimpleNamespace(
    load_run=RunState.load, describe_runs=lambda: [describe_run(state) for state in list_runs() if state.shots], continue_run=continue_run,
    discard_run=discard_run, delete_video=lambda folder, path: abort(404), captions=lambda: lambda name: {}, context=context)

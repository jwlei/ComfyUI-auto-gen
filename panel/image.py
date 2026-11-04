"""The panel's Image side: sets of images from text or from an image, with prompts the LLM writes or you do."""
import os
import secrets
import sys
from types import SimpleNamespace

from flask import abort, request, send_file

from core import seeds, workflow_options
from core.saved_workflows import shorthand
from panel.server import (IMAGE_TIMINGS, PIPELINES, _runs_cache, app, busy_runs, form_options, password_required, run_queue, saved_at,
                          saved_workflows)
from panel.video import clean_inputs, queue_label, shot_seeds
from pipelines.image import media, models
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
            return None, None, f"ComfyUI has no saved workflow called {name} any more."
        if "problem" in entry:
            return None, None, f"{entry['label']} can't run: {entry['problem']}"
        return {"COMFYUI_WORKFLOW": entry["path"]}, entry["takes"], None
    entry = automatic([entry for entry in entries if "problem" not in entry], image_count or 0)
    if entry:
        return {"COMFYUI_WORKFLOW": entry["path"]}, entry["takes"], None
    if image_count is None:
        return {}, {"min_images": 0, "max_images": 1, "size": True}, None
    wanted = "edits an image" if image_count else "makes images from text"
    why = "; ".join(f"{entry['label']}: {entry['problem']}" for entry in entries if "problem" in entry)
    return None, None, (f"No workflow that {wanted} can run. {why}" if why
                        else f"No workflow that {wanted} is saved in ComfyUI with a name starting image_.")


def automatic(usable, image_count):
    """
    The workflow Automatic picks, in name order. One made for what you gave it comes first: text alone goes to one that
    takes no image, and an image to one that needs one, so Fibo 2.0, which does both, is for when nothing else fits.
    """
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


def model_problem(flags, takes, source):
    """Why the chosen workflow's model shouldn't make this: too big, or too many prompts for the LLM. None if it's fine."""
    model = takes.get("model")
    if not takes.get("size"):
        size = None
    elif not source and media.FROM_IMAGE in (option(flags, "--aspect"), option(flags, "--megapixels")) and not option(flags, "--size"):
        size = None  # Comes from an image not chosen yet, which is held to the model's limit when it is
    else:
        size = media.pick_size(option(flags, "--size"), option(flags, "--aspect"), option(flags, "--megapixels"), source,
                               models.MOST_MEGAPIXELS.get(model))
    return (size and models.size_problem(model, *size)) or models.count_problem(model, int(option(flags, "--count")), "--direct" in flags)


def resume_argv(state, redo=None):
    argv = [*COMMAND, "--resume", state.id]
    return argv + ["--redo", str(redo)] if redo else argv


def image_files(state):
    """What the gallery shows of a run. The time makes a remade image load anew rather than from the browser's cache."""
    files = []
    for position, shot in enumerate(state.shots or [], 1):
        if shot.get("file"):
            files.append({"position": position, "title": shot["title"], "version": int(os.path.getmtime(state.image_path(shot)))})
    return files


def describe_run(state):
    """A saved run, as the panel lists it."""
    done, total = len(state.done), len(state.shots)
    parts = [f"{done} of {total} images" if done else f"Prompts ready, {total} {'image' if total == 1 else 'images'}", f"{state.width}x{state.height}"]
    env = {"COMFYUI_WORKFLOW": state.workflow} if state.workflow else {}
    return {"id": state.id, "pipeline": "image", "label": state.title or state.brief, "detail": " · ".join(parts),
            "model": shorthand(state.workflow, "image"), "loras": workflow_options.loras_used(state.workflow, state.options, "image"),
            "when": saved_at(state.path), "clips": done, "finished": state.finished, "total": total, "images": image_files(state),
            "size": f"{state.width}x{state.height}", "estimate": RunEta(resume_argv(state), env, IMAGE_TIMINGS).plan()["total"]}


def size_flags(form):
    """The size flags from the form. Returns (flags, error)."""
    aspect = form.get("aspect") or media.FROM_IMAGE
    # An exact size is typed in, or one of the screens on the list.
    size = (form.get("size") or "").strip().lower() if aspect == "custom" else aspect if media.parse_size(aspect) else None
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
        elif llm not in dict(planner.LLMS):
            return None, "Unknown LLM."
        else:
            flags += ["--llm", llm]
            if form.get("review"):
                flags.append("--plan-only")

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
    if not prompt:
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
    problem = model_problem(flags, takes, images[0].size if images else None)
    if problem:
        return {"ok": False, "error": problem}, 400
    option_flags, problem = chosen_options(request.form, takes)
    if problem:
        return {"ok": False, "error": problem}, 400
    flags += option_flags

    state = run_to_continue(prompt, flags, env, images)
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
    clean_inputs()
    image_paths = [os.path.join(video_paths.INPUTS_DIR, f"{secrets.token_hex(8)}.png") for _ in images]
    for image, path in zip(images, image_paths):
        media.save_source(image, path)

    argv = [*COMMAND, *flags, *[flag for path in image_paths for flag in ("--image", path)], "--", prompt]
    plan_only = "--plan-only" in flags
    count = int(option(flags, "--count"))
    detail = [f"{count} {'image' if count == 1 else 'images'}", "edits an image" if images else "from text",
              ROLE_NAMES.get(role) or llm or "your prompt, word for word", *(["review the prompts first"] if plan_only else [])]
    if env.get("COMFYUI_WORKFLOW"):
        detail.append(os.path.splitext(os.path.basename(env["COMFYUI_WORKFLOW"]))[0])
    item = {"pipeline": "image", "label": queue_label(prompt), "detail": " · ".join(detail), "argv": argv, "env": env,
            "image": image_paths[0] if image_paths else None, "references": image_paths[1:], "plan_only": plan_only,
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


def load_idle_run(run_id):
    """The saved run, or the reply saying why it can't be touched now."""
    state = RunState.load(run_id)
    if not state or not state.shots:
        return None, ({"ok": False, "error": "That run is no longer saved."}, 404)
    if run_id in busy_runs():
        return None, ({"ok": False, "error": "That run is going or queued. Stop it or take it out of the queue first."}, 409)
    return state, None


@app.route("/image-runs/<run_id>")
@password_required
def image_run_detail(run_id):
    """A set's images and prompts, for reading and changing them."""
    state = RunState.load(run_id)
    if not state or not state.shots:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    return {"ok": True, "id": state.id, "title": state.title, "brief": state.brief, "llm": state.llm, "direct": state.direct,
            "busy": run_id in busy_runs(), "finished": state.finished, "is_image": True,
            "size": f"{state.width}x{state.height}", "image_count": state.image_count,
            "shots": [{"title": shot["title"], "prompt": shot["prompt"], "rendered": bool(shot.get("file")),
                       "seed": shot.get("seed"), "used_seed": shot.get("used_seed"),
                       "version": int(os.path.getmtime(state.image_path(shot))) if shot.get("file") else 0} for shot in state.shots]}


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
    state, refused = load_idle_run(run_id)
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
    else:
        if shot.get("file"):
            os.remove(state.image_path(shot))
        del state.shots[position - 1]
        state.save()
    _runs_cache["at"] = 0.0
    return {"ok": True}


def continue_run(run_id):
    """Makes what's left of a planned or stopped set, or queues that behind the run in progress."""
    state, refused = load_idle_run(run_id)
    return refused or queue_saved_run(state)


def discard_run(run_id):
    """Deletes a set: its prompts, and the images it has made in output/."""
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

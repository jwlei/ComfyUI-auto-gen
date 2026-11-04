"""The panel's Video side: films from an image and a prompt, whose prompts you can read and change before rendering."""
import json
import os
import re
import secrets
import sys
from types import SimpleNamespace

from flask import abort, request, send_file

from core import seeds, workflow_options
from panel.server import (read_outcome, form_options, PIPELINES, VIDEO_TIMINGS, _runs_cache, app, busy_runs, item_images, job,
                          password_required, saved_at, run_queue, saved_runs, saved_workflows)
from panel.shorts import VOICES
from pipelines.video import media, paths, planner, roles
from pipelines.video.comfy import workflow_model
from pipelines.video.main import BUILT_IN
from pipelines.video.run_eta import RunEta
from core.saved_workflows import shorthand
from pipelines.video.run_state import STAGES, RunState, films_of_runs, folder_size, list_runs, run_of_film

COMMAND = [sys.executable, "-m", "pipelines.video.main"]
DEFAULT_FILM_SECONDS = 120
BUILT_IN_TAKES = {"min_images": 1, "max_images": 1, "max_videos": 0, "max_audios": 0, "max_references": 1}
# The most the form offers, however many a workflow could take.
MOST_IMAGES = 20
# The most reference video and audio boxes it offers: MiniMax H3 takes 3 of each.
MOST_MEDIA = 6


def option(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags[:-1] else default


def chosen_workflow(form):
    """What makes a run use the workflow chosen on the form, and the images it takes. Returns (env, takes, error)."""
    name = form.get("workflow")
    if not name:
        return {}, BUILT_IN_TAKES, None
    entry = next((w for w in saved_workflows() if w["name"] == name), None)
    if not entry:
        return None, None, f"ComfyUI has no saved workflow called {name} any more."
    if "problem" in entry:
        return None, None, f"{entry['label']} can't run: {entry['problem']}"
    return {"COMFYUI_WORKFLOW": entry["path"]}, entry["takes"], None


def clean_inputs(keep=()):
    """Images and audio whose job has finished or gone: the run keeps its own copy, so nothing needs these any more."""
    needed = {path for item in run_queue.pending() for path in item_images(item)} | set(keep)
    try:
        for name in os.listdir(paths.INPUTS_DIR):
            path = os.path.join(paths.INPUTS_DIR, name)
            if path not in needed:
                os.remove(path)
    except OSError:
        pass  # Only costs a little disk until the next one


def resume_argv(state, redo=None):
    argv = [*COMMAND, "--resume", state.id]
    return argv + ["--redo", str(redo)] if redo else argv


def describe_run(state):
    """A saved run, as the panel lists it."""
    parts = []
    if state.final:
        parts.append("Finished")
    elif not state.clips:
        parts.append(f"Plan ready, {len(state.shots)} shots")
    elif state.next_stage() == "Clips":
        parts.append(f"{len(state.clips)} of {len(state.shots)} clips kept")
    else:
        parts.append("Every clip is done, only stitching is left")
    parts.append(f"{folder_size(state.folder) / 2 ** 20:.0f} MB")
    return {"id": state.id, "pipeline": "video", "label": state.title or state.brief, "detail": " · ".join(parts), "when": saved_at(state.path),
            "clips": len(state.clips), "finished": bool(state.final), "outcome": read_outcome(state.folder),
            "estimate": RunEta(resume_argv(state), VIDEO_TIMINGS).plan()["total"]}


def build_options(form):
    """
    Turns the form's options into main.py's flags. Returns (flags, error). The image
    and prompt aren't needed for that: the estimate asks before they are filled in.
    """
    flags = []
    try:
        seconds = int(form.get("duration") or DEFAULT_FILM_SECONDS)
    except ValueError:
        return None, "Film length must be a whole number of seconds."
    if not planner.MIN_FILM_SECONDS <= seconds <= planner.MAX_FILM_SECONDS:
        return None, f"Film length must be {planner.MIN_FILM_SECONDS}-{planner.MAX_FILM_SECONDS} seconds."
    flags += ["--duration", str(seconds)]

    start = form.get("start") or "chain"
    if start not in ("chain", "image"):
        return None, "Unknown start mode."
    flags += ["--start", start]

    # One role per image from image 2, in order. The form only sends one for an image that's chosen.
    for number in range(2, MOST_IMAGES + 1):
        role = form.get(f"image_{number}_role")
        if role is None:
            break
        if role not in planner.IMAGE_ROLES:
            return None, f"Unknown role for image {number}."
        flags += ["--image-role", role]

    llm = form.get("llm") or planner.default_llm()
    if llm not in dict(planner.LLMS):
        return None, "Unknown LLM."
    flags += ["--llm", llm]

    size = (form.get("size") or "").strip().lower()
    if size:
        if not re.fullmatch(r"\d{2,4}x\d{2,4}", size):
            return None, "Render size must look like 1280x736."
        flags += ["--size", size]

    try:
        seed = seeds.parse(form.get("seed"))
    except ValueError:
        return None, f"The seed must be a whole number from 0 to {seeds.MOST}, or empty for random."
    if seed is not None:
        flags += ["--seed", str(seed)]

    if form.get("review"):
        flags.append("--plan-only")
    return flags, None


@app.route("/api/estimate/video", methods=["POST"], endpoint="video_estimate")
@password_required
def estimate():
    """What a run with the form's current options should take. Starts nothing."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}
    env, _, _ = chosen_workflow(request.form)
    # Reviewing first only writes the prompts now, but what you're deciding on is the whole film.
    return {"ok": True, **RunEta([arg for arg in flags if arg != "--plan-only"], VIDEO_TIMINGS, env).plan()}


def queue_label(text):
    text = " ".join(text.split())
    return text if len(text) <= 80 else text[:77] + "..."


def form_uploads(files, takes):
    """The images chosen in the form's boxes (image, image_2...), in order. Returns (uploads, error)."""
    field = lambda number: "image" if number == 1 else f"image_{number}"
    filled = {name for name, upload in files.items() if upload.filename and name.startswith("image")}
    uploads = [files[field(number)] for number in range(1, len(filled) + 1) if field(number) in filled]
    if len(uploads) < len(filled):
        return None, "Fill the image boxes in order: each one adds to those before it."
    low, high = takes["min_images"], takes["max_images"]
    # Clips with room for one image still leave image 2 for the LLM.
    room = min(MOST_IMAGES, 2 if high == 1 else high)
    if len(uploads) < low:
        return None, "Choose the image the film opens on." if low == 1 else f"This workflow needs {low} images."
    if len(uploads) > room:
        return None, "This workflow makes its clips from text alone: leave the image out." if not room else f"This workflow takes at most {room} images."
    return uploads, None


def form_references(files, form, takes, images):
    """
    The reference videos and audio clips chosen in the form's boxes (ref_media, ref_media_2...), saved
    to the inputs folder and sorted by what they are. Returns ([(path, role)] videos, [(path, role)] audio clips, error).
    """
    field = lambda number: "ref_media" if number == 1 else f"ref_media_{number}"
    chosen = [number for number in range(1, MOST_MEDIA + 1) if files.get(field(number)) and files[field(number)].filename]
    videos, audios = [], []
    for number in chosen:
        upload = files[field(number)]
        extension = os.path.splitext(upload.filename)[1].lower()
        path = os.path.join(paths.INPUTS_DIR, f"{secrets.token_hex(8)}{extension if re.fullmatch(r'[.][a-z0-9]{1,5}', extension) else ''}")
        os.makedirs(paths.INPUTS_DIR, exist_ok=True)
        upload.save(path)
        kind = media.media_kind(path)
        if not kind:
            os.remove(path)
            return None, None, f"Reference {number} isn't a video or audio file this can read."
        role = form.get(f"{field(number)}_role")
        if kind == "video":
            videos.append((path, role if role in roles.VIDEO_ROLES else roles.MOTION))
        else:
            audios.append((path, role if role in roles.AUDIO_ROLES else roles.SOUND))
    error = None
    if len(videos) > takes.get("max_videos", 0):
        error = f"This workflow takes {takes.get('max_videos', 0) or 'no'} reference videos."
    elif len(audios) > takes.get("max_audios", 0):
        error = f"This workflow takes {takes.get('max_audios', 0) or 'no'} reference audio clips."
    elif images + len(videos) + len(audios) > takes.get("max_references", takes["max_images"]):
        error = f"This workflow takes {takes['max_references']} references at most, images, videos and audio together."
    if error:
        for path, _ in videos + audios:
            os.remove(path)
        return None, None, error
    return videos, audios, None


@app.route("/generate/video", methods=["POST"], endpoint="video_generate")
@password_required
def generate():
    """Starts a film, or queues it behind the one in progress."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    env, takes, error = chosen_workflow(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    chosen, error = form_options(request.form, takes)
    if error:
        return {"ok": False, "error": error}, 400
    if chosen:
        # Kept with the run, for the rest of its clips (pipelines/video/main.py).
        env = {**env, "COMFYUI_OPTIONS": json.dumps(chosen)}
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "Say what should happen in the video."}, 400
    llm = flags[flags.index("--llm") + 1]
    if not planner.llm_configured(llm):
        return {"ok": False, "error": f"{llm} has no credentials. See .env.example."}, 400

    voices = dict(VOICES)
    audio_mode = request.form.get("audio_mode") or "none"
    narration = (request.form.get("narration") or "").strip() if audio_mode == "text" else ""
    voice = request.form.get("voice") or ""
    audio_upload = request.files.get("audio_file") if audio_mode == "file" else None
    if audio_mode == "text" and not narration:
        return {"ok": False, "error": "Write the text to voice, or choose no audio."}, 400
    if audio_mode == "text" and voice not in voices:
        return {"ok": False, "error": "Unknown voice."}, 400
    if audio_mode == "file" and not (audio_upload and audio_upload.filename):
        return {"ok": False, "error": "Choose the audio file, or choose no audio."}, 400

    uploads, error = form_uploads(request.files, takes)
    if error:
        return {"ok": False, "error": error}, 400
    try:
        images = [media.load_image(upload.stream) for upload in uploads]
    except RuntimeError:
        return {"ok": False, "error": "That file isn't an image this can read."}, 400

    # One box for image 2 is the LLM's alone when the clips take one image, and doesn't count.
    references = len(images) if takes["max_images"] != 1 else min(len(images), 1)
    ref_videos, ref_audios, error = form_references(request.files, request.form, takes, references)
    if error:
        return {"ok": False, "error": error}, 400
    media_paths = [path for path, _ in ref_videos + ref_audios]

    audio_path = None
    if audio_upload:
        extension = os.path.splitext(audio_upload.filename)[1].lower()
        audio_path = os.path.join(paths.INPUTS_DIR, f"{secrets.token_hex(8)}{extension if re.fullmatch(r'[.][a-z0-9]{1,5}', extension) else ''}")
        os.makedirs(paths.INPUTS_DIR, exist_ok=True)
        audio_upload.save(audio_path)
    audio = media.audio_fingerprint(path=audio_path, text=narration or None, voice=voice or None)

    state = run_to_continue(prompt, flags, env, takes, images, audio, ref_videos + ref_audios)
    if state:
        for path in ([audio_path] if audio_path else []) + media_paths:
            os.remove(path)
        if "--plan-only" in flags:
            return {"ok": True, "queued": False, "shots": state.id, "continued": state.title}
        # Options change how the rest is made, not the plan: what's paid for stays.
        state.options = json.loads(env.get("COMFYUI_OPTIONS") or "{}") or None
        state.adopt(flags)
        reply = queue_saved_run(state)
        return reply if isinstance(reply, tuple) else {**reply, "continued": state.title}

    # Kept in a folder of its own, under a name of our own: queued, it may wait for hours.
    clean_inputs(keep=([audio_path] if audio_path else []) + media_paths)
    image_paths = [os.path.join(paths.INPUTS_DIR, f"{secrets.token_hex(8)}.png") for _ in images]
    for image, path in zip(images, image_paths):
        media.save_input(image, path)

    references = [flag for path in image_paths[1:] for flag in ("--reference", path)]
    audio_flags = ["--audio", audio_path] if audio_path else ["--narration", narration, *(["--voice", voice] if voice else [])] if narration else []
    media_flags = [flag for path, role in ref_videos for flag in ("--ref-video", path, "--ref-video-role", role)]
    media_flags += [flag for path, role in ref_audios for flag in ("--ref-audio", path, "--ref-audio-role", role)]
    argv = [*COMMAND, *flags, *references, *audio_flags, *media_flags, *([] if images else ["--no-image"]), "--", *image_paths[:1], prompt]
    plan_only = "--plan-only" in flags
    seconds = int(flags[flags.index("--duration") + 1])
    length = f"{seconds}s" if seconds < 60 else f"{seconds // 60}:{seconds % 60:02d}"
    detail = ["film as long as its audio" if audio else f"{length} film", llm, "review the prompts first" if plan_only else "renders straight away"]
    if env:
        detail.append(request.form["workflow"].removesuffix(".json"))
    item = {"pipeline": "video", "label": queue_label(prompt), "detail": " · ".join(detail), "argv": argv, "env": env,
            "image": image_paths[0] if image_paths else None, "references": image_paths[1:] + ([audio_path] if audio_path else []) + media_paths, "plan_only": plan_only,
            "estimate": RunEta(argv, VIDEO_TIMINGS, env).plan()["total"]}
    queued = run_queue.add(item)
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


def run_to_continue(prompt, flags, env, takes, images, audio=None, references=()):
    """
    The stopped run Generate should finish rather than pay for the same plan again:
    the same brief, images, references, audio, length and workflow. Anything else changes the prompts.
    """
    busy = busy_runs()
    inputs = media.inputs_fingerprint(images, [flags[i + 1] for i, flag in enumerate(flags) if flag == "--image-role"], audio, references)
    seconds, start = int(option(flags, "--duration")), option(flags, "--start")
    for state in reversed(list_runs()):
        if (state.id not in busy and state.shots and not state.final and state.inputs == inputs and state.brief == prompt
                and (audio or state.seconds == seconds) and state.workflow == env.get("COMFYUI_WORKFLOW")
                and (state.start == start or takes["max_images"] != 1)):
            return state
    return None


def queue_saved_run(state, redo=None):
    argv = resume_argv(state, redo)
    what = f"Redo from shot {redo}" if redo else "Continue"
    item = {"pipeline": "video", "label": queue_label(state.title or state.brief), "detail": what, "argv": argv, "env": {},
            "resume": state.id, "estimate": RunEta(argv, VIDEO_TIMINGS).plan()["total"]}
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


@app.route("/runs/<run_id>")
@password_required
def run_detail(run_id):
    """A run's shots, for reading and changing its prompts."""
    state = RunState.load(run_id)
    if not state or not state.shots:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    return {"ok": True, "id": state.id, "title": state.title, "brief": state.brief, "start": state.start,
            "llm": state.llm, "busy": run_id in busy_runs(), "finished": bool(state.final),
            "shots": [{"title": shot["title"], "seconds": shot["seconds"], "prompt": shot["prompt"],
                       "seed": shot.get("seed"), "used_seed": shot.get("used_seed"),
                       "rendered": i in state.clips} for i, shot in enumerate(state.shots)]}


@app.route("/runs/<run_id>/image")
@password_required
def run_image(run_id):
    state = RunState.load(run_id)
    if not state or not os.path.exists(state.image_path):
        abort(404)
    return send_file(state.image_path, mimetype="image/png", conditional=True)


@app.route("/runs/<run_id>/clips/<int:shot>")
@password_required
def run_clip(run_id, shot):
    state = RunState.load(run_id)
    if not state or shot - 1 not in state.clips:
        abort(404)
    return send_file(state.clip_path(shot - 1), mimetype="video/mp4", conditional=True)


@app.route("/runs/<run_id>/shots", methods=["POST"])
@password_required
def save_shots(run_id):
    """Saves changed prompts. A clip already rendered stays as it is until you redo it."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    body = request.get_json(silent=True) or {}
    prompts = body.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(state.shots) or not all(isinstance(p, str) and p.strip() for p in prompts):
        return {"ok": False, "error": "Every shot needs a prompt."}, 400
    chosen, error = shot_seeds(body, len(state.shots))
    if error:
        return {"ok": False, "error": error}, 400
    for shot, prompt, seed in zip(state.shots, prompts, chosen):
        shot["prompt"] = prompt.strip()
        shot["seed"] = seed
    state.save()
    return {"ok": True}


def shot_seeds(body, count):
    """The seed set for each shot or image in the editor, None for random. Without any sent, every one is random."""
    given = body.get("seeds")
    if given is None:
        return [None] * count, None
    if not isinstance(given, list) or len(given) != count:
        return None, "Every shot needs a seed, or none for random."
    try:
        return [seeds.parse(seed) for seed in given], None
    except ValueError:
        return None, f"A seed must be a whole number from 0 to {seeds.MOST}, or empty for random."


def continue_run(run_id):
    """Renders a planned or stopped run from where it got to, or queues that behind the run in progress."""
    state, refused = load_idle_run(run_id)
    return refused or queue_saved_run(state)


@app.route("/runs/<run_id>/redo/<int:shot>", methods=["POST"])
@password_required
def redo_shot(run_id, shot):
    """Renders a shot again and joins the film anew. In a chain, the shots after it start from it, so they go too."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    if not 1 <= shot <= len(state.shots):
        return {"ok": False, "error": "That shot isn't in this film."}, 400
    return queue_saved_run(state, redo=shot)


def discard_run(run_id):
    """Deletes a run's plan and clips. A film already finished from it stays."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


def delete_video(folder, path):
    """Deletes a film and the run that made it, whose clips would otherwise sit on the disk unseen."""
    run_id = run_of_film(os.path.basename(path))
    if run_id and run_id in busy_runs():
        return {"ok": False, "error": "That film's run is going or queued. Stop it or take it out of the queue first."}, 409
    os.remove(path)
    state = RunState.load(run_id)
    if state:
        state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


def captions():
    titles = {run["id"]: run["label"] for run in saved_runs() if run["pipeline"] == "video"}
    films = films_of_runs()

    def caption(name):
        run = run_of_film(name, films)
        made_by = films.get(name)
        return {"run": run if run in titles else None, "title": titles.get(run),
                "model": shorthand(made_by.workflow, BUILT_IN) if made_by else None,
                "loras": workflow_options.loras_used(made_by.workflow, made_by.options, "video") if made_by else []}
    return caption


def context():
    return {
        "stages": STAGES,
        "llms": [{"name": name, "label": label, "ready": planner.llm_configured(name)} for name, label in planner.LLMS],
        "default_llm": planner.default_llm(),
        "film_limits": [planner.MIN_FILM_SECONDS, planner.MAX_FILM_SECONDS, DEFAULT_FILM_SECONDS],
        "most_images": MOST_IMAGES,
        "most_media": MOST_MEDIA,
        "video_roles": [[role, roles.VIDEO_LABELS[role]] for role in roles.VIDEO_ROLES],
        "audio_roles": [[role, roles.AUDIO_LABELS[role]] for role in roles.AUDIO_ROLES],
        "clip_seconds": planner.clip_seconds(workflow_model()),
        "size_presets": media.SIZE_PRESETS[::-1],
    }


PIPELINES["video"] = SimpleNamespace(
    load_run=RunState.load, describe_runs=lambda: [describe_run(state) for state in list_runs() if state.shots], continue_run=continue_run,
    discard_run=discard_run, delete_video=delete_video, captions=captions, context=context)

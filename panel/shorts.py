"""The panel's Shorts side: its options, its runs and YouTube uploads."""
import json
import os
import re
import sys
from types import SimpleNamespace

from flask import request

from core import seeds
from core.paths import PROJECT_ROOT, WORKFLOWS_DIR
from panel.server import (read_outcome, form_options, PIPELINES, SHORTS_TIMINGS, _runs_cache, app, busy_runs, job, list_videos, password_required, saved_at,
                          run_queue, safe_video_path, saved_workflows, upload_queue, queue_form_snapshot)
from pipelines.shorts.avatar_generator import avatar_clip_range, on_screen_range
from pipelines.shorts.paths import avatar_still_path, thumbnail_path
from pipelines.shorts.run_eta import RunEta
from pipelines.shorts.run_state import RunState, folder_size, list_runs, option
from pipelines.shorts.topic_history import forget_video, read_history

COMMAND = [sys.executable, "-m", "pipelines.shorts.main"]

# Gemini voices only: edge-tts is never used.
VOICES = [
    ("", "Default (UK: Presenter, female)"),
    ("en-gb-tutor-13", "Presenter — UK female, Winchester"),
    ("en-gb-storyteller-3", "Storyteller — UK female, Newcastle"),
    ("en-gb-storyteller-2", "Storyteller — UK female, mature"),
    ("en-gb-storyteller-1", "Storyteller — UK male"),
    ("en-us-storyteller-6", "Storyteller — US female"),
    ("en-us-storyteller-14", "Storyteller — US male"),
]
AVATAR_STYLES = {
    "lipsync": os.path.join(WORKFLOWS_DIR, "avatar_lipsync.json"),
    "reaction": os.path.join(WORKFLOWS_DIR, "avatar_workflow.json"),
}
# The panel's sliders: form field prefix, the variables they set, and their limits.
AVATAR_SLIDERS = (
    ("avatar_clips", "AVATAR_CLIPS_MIN", "AVATAR_CLIPS_MAX", 1, 9),
    ("avatar_seconds", "AVATAR_MIN_SECONDS", "AVATAR_MAX_SECONDS", 2, 10),
)


def saved_workflow_path(name):
    """The file that makes a run use this saved workflow. Returns (path, error)."""
    entry = next((w for w in saved_workflows() if w["name"] == name), None)
    if not entry:
        return None, f"ComfyUI has no saved workflow called {name} any more."
    if "problem" in entry:
        return None, f"{entry['label']} can't run: {entry['problem']}"
    return entry["path"], None


def resume_argv(state):
    return [*COMMAND, *state.argv, "--resume", state.id]


def describe_stopped(state):
    """A run waiting to be continued, as the panel lists it."""
    stage = state.next_stage()
    parts = ["Only the last step is left" if stage == "Upload" else f"Continues at {stage}"]
    if state.avatar_clips and not state.avatar_done:
        parts.append(f"{len(state.avatar_clips)} of {len(state.avatar_indices)} avatar clips kept")
    if state.videos_after:
        parts.append(f"{state.videos_after} more {'video' if state.videos_after == 1 else 'videos'} after it")
    parts.append(f"{folder_size(state.folder) / 2 ** 20:.0f} MB")
    return {"id": state.id, "pipeline": "shorts", "outcome": read_outcome(state.folder), "label": state.topic or f"Video {state.number}, no topic yet",
            "detail": " · ".join(parts), "when": saved_at(state.path), "clips": len(state.avatar_clips),
            "estimate": RunEta(resume_argv(state), state.env, SHORTS_TIMINGS).plan()["total"]}


def language_of(voice):
    """en-US for a US voice, else en-GB: the default voice is the UK presenter."""
    return "en-US" if voice.startswith("en-us") else "en-GB"


def run_to_continue(topic, argv):
    """
    The stopped run Generate should finish rather than pay for the same work again:
    the one on this topic, or the oldest if no topic was asked for. A script in
    another language is no use.
    """
    busy = busy_runs()
    runs = [state for state in reversed(list_runs())
            if state.id not in busy and option(state.argv, "--lang", "en-GB") == option(argv, "--lang", "en-GB")]
    if not topic:
        return runs[0] if runs else None
    return next((state for state in runs if (state.topic or "").casefold() == topic.casefold()), None)


def build_run(form):
    """
    Turns the form into a main.py command line plus environment overrides.
    Returns (argv, env, error).
    """
    argv = list(COMMAND)
    env = {}

    topic = (form.get("topic") or "").strip()
    if topic:
        argv += ["--topic", topic]

    try:
        runs = max(1, min(10, int(form.get("runs", 1))))
    except ValueError:
        runs = 1
    if topic and runs > 1:
        return None, None, "A topic makes one video. Leave Topic empty to make several."
    argv += ["--runs", str(runs)]

    voice = form.get("voice") or ""
    if voice not in dict(VOICES):
        return None, None, "Unknown voice."
    # The voice says the language: its accent, and the spelling the script is written in.
    argv += ["--lang", language_of(voice)]
    if voice:
        argv += ["--voice", voice]

    clips = form.get("avatar_clips") or "auto"
    if clips not in ("auto", "0", "2", "3"):
        return None, None, "Unknown avatar layout."
    argv += ["--avatar-clips", clips]
    try:
        seed = seeds.parse(form.get("seed"))
    except ValueError:
        return None, None, f"The seed must be a whole number from 0 to {seeds.MOST}, or empty for random."
    if seed is not None:
        argv += ["--seed", str(seed)]

    # The panel always renders her with ComfyUI; the looped avatars.mp4 is left to the command line.
    env["AVATAR_SOURCE"] = "comfyui"
    size = (form.get("avatar_size") or "").strip().lower()
    if size:
        if size not in dict(AVATAR_SIZES):
            return None, None, "Unknown resolution."
        env["AVATAR_SIZE"] = size

    style = form.get("avatar_style") or "lipsync"
    if style not in AVATAR_STYLES:
        return None, None, "Unknown avatar style."
    env["COMFYUI_WORKFLOW"] = AVATAR_STYLES[style]

    workflow = form.get("workflow")
    if workflow:
        env["COMFYUI_WORKFLOW"], error = saved_workflow_path(workflow)
        if error:
            return None, None, error
        entry = next(w for w in saved_workflows() if w["name"] == workflow)
        chosen, error = form_options(form, entry.get("takes") or {})
        if error:
            return None, None, error
        if chosen:
            # Kept with each video's run (run_state.RUN_ENV), for continuing it.
            env["COMFYUI_OPTIONS"] = json.dumps(chosen)

    for field, low_name, high_name, lowest, highest in AVATAR_SLIDERS:
        try:
            low, high = int(form[f"{field}_min"]), int(form[f"{field}_max"])
        except KeyError:
            continue  # A page opened before the sliders existed: keep the .env values
        except ValueError:
            return None, None, "Avatar sliders must be whole numbers."
        if not lowest <= low <= high <= highest:
            return None, None, f"Avatar slider out of range: {low}-{high}, allowed {lowest}-{highest}."
        env[low_name], env[high_name] = str(low), str(high)

    if form.get("upload"):
        privacy = form.get("privacy") or "private"
        if privacy not in ("private", "unlisted", "public"):
            return None, None, "Unknown privacy setting."
        argv += ["--upload", "--privacy", privacy]

    if form.get("keep_cache"):
        argv.append("--keep-cache")
    if form.get("fail_fast"):
        argv.append("--fail-fast")

    output = (form.get("output") or "").strip()
    if output:
        if runs != 1:
            return None, None, "A custom file name only works for a single video."
        if not output.lower().endswith(".mp4"):
            output += ".mp4"
        if not re.fullmatch(r"[\w.-]+\.mp4", output):
            return None, None, "File name may only contain letters, numbers, '-', '_' and '.'."
        argv += ["--output", output]

    return argv, env, None


@app.route("/api/estimate/shorts", methods=["POST"], endpoint="shorts_estimate")
@password_required
def estimate():
    """What a run with the form's current options should take. Starts nothing."""
    argv, env, error = build_run(request.form)
    if error:
        return {"ok": False, "error": error}
    state = run_to_continue((request.form.get("topic") or "").strip(), argv)
    if state:
        return {"ok": True, **RunEta(resume_argv(state), state.env, SHORTS_TIMINGS).plan(), "continues": state.topic or "the stopped video"}
    return {"ok": True, **RunEta(argv, env, SHORTS_TIMINGS).plan()}


def describe_run(form, argv):
    """The queue's label and small print for a run built from this form."""
    topic = (form.get("topic") or "").strip()
    runs = argv[argv.index("--runs") + 1]
    label = topic or ("1 video" if runs == "1" else f"{runs} videos")
    avatar = "no avatar" if form.get("avatar_clips") == "0" else "ComfyUI avatar"
    if form.get("workflow") and avatar == "ComfyUI avatar":
        avatar = f"ComfyUI avatar ({form['workflow'].removesuffix('.json')})"
    upload = f"upload {form.get('privacy') or 'private'}" if form.get("upload") else "no upload"
    return label, " · ".join((argv[argv.index("--lang") + 1], avatar, upload))


@app.route("/generate/shorts", methods=["POST"], endpoint="shorts_generate")
@password_required
def generate():
    """Starts a run, or queues it behind the one in progress. Stopped work on it is continued, not paid for again."""
    argv, env, error = build_run(request.form)
    if error:
        return {"ok": False, "error": error}, 400

    # The second run would stop on the repeated topic or overwrite the file.
    topic = (request.form.get("topic") or "").strip()
    output = argv[argv.index("--output") + 1] if "--output" in argv else ""
    for other in run_queue.pending():
        if str(other.get("id")) == request.form.get("queue_edit_id"):
            continue
        if topic and other.get("topic", "").lower() == topic.lower():
            return {"ok": False, "error": "That topic is already running or queued."}, 409
        if output and other.get("output", "").lower() == output.lower():
            return {"ok": False, "error": "That file name is already used by a running or queued job."}, 409

    state = None if request.form.get("queue_edit_id") else run_to_continue(topic, argv)
    if state:
        state.adopt(argv[len(COMMAND):], env, int(option(argv, "--runs", 1)) - 1)
        reply = queue_continued(state)
        return (reply, 500) if reply.get("error") else {**reply, "continued": state.topic or "the stopped video"}

    label, detail = describe_run(request.form, argv)
    item = {"pipeline": "shorts", "label": label, "detail": detail, "argv": argv, "env": env, "topic": topic, "output": output,
            "form": queue_form_snapshot(), "estimate": RunEta(argv, env, SHORTS_TIMINGS).plan()["total"]}
    queued = run_queue.add(item)
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


def queue_continued(state):
    argv = resume_argv(state)
    topic = state.topic or ""
    output = argv[argv.index("--output") + 1] if "--output" in argv else ""
    clips = argv[argv.index("--avatar-clips") + 1] if "--avatar-clips" in argv else "auto"
    item = {"pipeline": "shorts", "label": topic or f"Video {state.number}", "detail": f"Continues at {state.next_stage()}",
            "argv": argv, "env": dict(state.env), "topic": topic, "output": output, "resume": state.id,
            "form": {"avatar_source": state.env.get("AVATAR_SOURCE") or "comfyui", "avatar_clips": clips},
            "estimate": RunEta(argv, state.env, SHORTS_TIMINGS).plan()["total"]}
    queued = run_queue.add(item)
    _runs_cache["at"] = 0.0
    return {"ok": False, "error": item["error"]} if item.get("error") else {"ok": True, "queued": queued}


def continue_run(run_id):
    """Starts a stopped run again from where it got to, or queues that behind the run in progress."""
    state = RunState.load(run_id)
    if not state:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    if run_id in busy_runs():
        return {"ok": False, "error": "That run is already going or queued."}, 409
    topic = state.topic or ""
    argv = resume_argv(state)
    output = argv[argv.index("--output") + 1] if "--output" in argv else ""
    for other in run_queue.pending():
        if topic and other.get("topic", "").lower() == topic.lower():
            return {"ok": False, "error": "That topic is already running or queued."}, 409
        if output and other.get("output", "").lower() == output.lower():
            return {"ok": False, "error": "That file name is already used by a running or queued job."}, 409

    reply = queue_continued(state)
    return (reply, 500) if reply.get("error") else reply


def form_of(state):
    """What the form held for this run, as its fields name it: to change before pressing Generate."""
    argv, env = state.argv, state.env
    workflow = next((w["name"] for w in saved_workflows() if env.get("COMFYUI_WORKFLOW") and w.get("path")
                     and os.path.abspath(w["path"]) == os.path.abspath(env["COMFYUI_WORKFLOW"])), "")
    style = next((name for name, path in AVATAR_STYLES.items() if path == env.get("COMFYUI_WORKFLOW")), "lipsync")
    try:
        options = json.loads(env.get("COMFYUI_OPTIONS") or "{}")
    except ValueError:
        options = {}
    form = {"topic": state.topic or "", "runs": option(argv, "--runs", "1"), "voice": option(argv, "--voice", ""),
            "avatar_clips": option(argv, "--avatar-clips", "auto"), "seed": option(argv, "--seed", ""),
            "avatar_size": env.get("AVATAR_SIZE", ""), "avatar_style": style, "workflow": workflow,
            "upload": "--upload" in argv, "privacy": option(argv, "--privacy", "private"), "output": option(argv, "--output", "")}
    for field, low_name, high_name, _, _ in AVATAR_SLIDERS:
        if env.get(low_name) and env.get(high_name):
            form[f"{field}_min"], form[f"{field}_max"] = int(env[low_name]), int(env[high_name])
    return {"ok": True, "form": form, "options": options}


@app.route("/shorts-runs/<run_id>/form")
@password_required
def shorts_run_form(run_id):
    state = RunState.load(run_id)
    if not state:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    return form_of(state)


def discard_run(run_id):
    """Deletes a stopped run's script, voice and clips."""
    state = RunState.load(run_id)
    if not state:
        return {"ok": False, "error": "That run is no longer saved."}, 404
    if run_id in busy_runs():
        return {"ok": False, "error": "That run is going or queued. Stop it or take it out of the queue first."}, 409
    state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


def youtube_configured():
    return all(os.getenv(k) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"))


@app.route("/publish", methods=["POST"])
@password_required
def publish():
    """
    Uploads one video from 0_output (form 'video'), or all of them, as a run, or
    queues that behind the upload in progress. All means the videos there now:
    what the page asked about, not whatever has been made by the time it starts.
    """
    privacy = request.form.get("privacy") or "private"
    if privacy not in ("private", "unlisted", "public"):
        return {"ok": False, "error": "Unknown privacy setting."}, 400
    if not youtube_configured():
        return {"ok": False, "error": "Set the YouTube keys in .env first (run scripts/authorize_youtube.py once)."}, 400

    claimed = {video for other in upload_queue.pending() for video in other["videos"]}
    ready = list_videos("output")
    argv = [sys.executable, "-m", "pipelines.shorts.publish", "--privacy", privacy]
    name = request.form.get("video")
    if name:
        name = os.path.basename(safe_video_path("output", name))
        if name in claimed:
            return {"ok": False, "error": "That video is already uploading or queued."}, 409
        videos = [name]
        argv.append(name)
        topic = (request.form.get("topic") or "").strip()
        if topic:
            argv += ["--topic", topic]
        recorded = next((v["topic"] for v in ready if v["name"] == name), None)
        label = topic or recorded or name
    else:
        videos = sorted(v["name"] for v in ready if v["name"] not in claimed)
        if not videos:
            return {"ok": False, "error": "Every ready video is already uploading or queued."}, 409
        argv += videos
        label = f"{len(videos)} {'video' if len(videos) == 1 else 'videos'}"

    queued = upload_queue.add({"pipeline": "shorts", "label": label, "detail": privacy, "argv": argv, "env": None, "videos": videos})
    return {"ok": True, "queued": queued}


def delete_video(folder, path):
    os.remove(path)
    for companion in (thumbnail_path(path), avatar_still_path(path)):
        if os.path.exists(companion):
            os.remove(companion)
    # An uploaded video's topic stays used: it's still on YouTube.
    if folder == "output":
        try:
            forget_video(os.path.basename(path))
        except (OSError, ValueError) as e:
            return {"ok": True, "warning": f"The video is deleted, but its topic couldn't be freed: {e}"}
    return {"ok": True}


def captions():
    try:
        history = {entry.get("video"): entry for entry in read_history()}
    except (OSError, ValueError):
        history = {}  # A broken history only costs the captions
    return lambda name: {"topic": history.get(name, {}).get("topic"), "model": history.get(name, {}).get("model")}


# The avatar clips' size: the finished Short is always 1080x1920, cropped and scaled from them.
AVATAR_SIZES = (("576x1024", "576x1024 (faster)"), ("736x1280", "736x1280 (usual)"), ("1088x1920", "1088x1920 (full size, slower)"))


def context():
    return {
        "voices": VOICES,
        "avatar_sizes": AVATAR_SIZES,
        "default_avatar_size": (os.getenv("AVATAR_SIZE") or "736x1280").lower(),
        "defaults": {
            "avatar_clips": (os.getenv("AVATAR_CLIPS") or "auto").lower(),
            "avatar_clips_min": avatar_clip_range()[0],
            "avatar_clips_max": avatar_clip_range()[1],
            "avatar_seconds_min": on_screen_range()[0],
            "avatar_seconds_max": on_screen_range()[1],
        },
        "can_upload": youtube_configured(),
    }


PIPELINES["shorts"] = SimpleNamespace(
    load_run=RunState.load, describe_runs=lambda: [describe_stopped(state) for state in list_runs()], continue_run=continue_run,
    discard_run=discard_run, delete_video=delete_video, captions=captions, context=context)

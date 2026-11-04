"""
The web panel: one server, one GPU queue, and three pipelines: Shorts, Video and Image.

Starts their jobs, queues more while one is going, streams the log, shows whether
ComfyUI is up, and lists the finished videos for playback and download, from the
desktop or a phone, at home or anywhere over Tailscale. panel/shorts.py,
panel/video.py and panel/image.py hold what is particular to each.

    python webapp.py --host 0.0.0.0 --port 8800
    (or double-click start_webapp.pyw / the "ComfyUI auto-gen" desktop shortcut)

Set APP_PASSWORD to expose it beyond localhost. Without it the server refuses
any host but 127.0.0.1: an open instance lets anyone spend your API quota and GPU.
"""
import argparse
import json
import logging
import os
import re
import shutil
import secrets
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime
from functools import wraps

import requests
import websocket
from dotenv import load_dotenv
from flask import (Flask, abort, has_request_context, redirect, render_template, request,
                   send_file, session, url_for)

from core.paths import LOGS_DIR, PROJECT_ROOT, SAVED_WORKFLOWS_DIR, comfyui_client_id
from core.quota import quota_summary
from core.saved_workflows import list_saved, pipeline_of
from panel import tailscale
from pipelines.image.paths import OUTPUT_DIR as IMAGE_OUTPUT
from pipelines.image.run_eta import RunEta as ImageEta
from pipelines.shorts.paths import OUTPUT_DIR as SHORTS_READY, UPLOADED_DIR as SHORTS_UPLOADED
from pipelines.shorts.run_eta import RunEta as ShortsEta
from pipelines.video.paths import OUTPUT_DIR as VIDEO_OUTPUT
from pipelines.video import planner as video_planner
from pipelines.video.comfy import workflow_model
from pipelines.video.run_eta import RunEta as VideoEta

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

FOLDERS = {"output": SHORTS_READY, "uploaded": SHORTS_UPLOADED, "video": VIDEO_OUTPUT, "image": IMAGE_OUTPUT}
FOLDER_PIPELINES = {"output": "shorts", "uploaded": "shorts", "video": "video", "image": "image"}
SHORTS_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_shorts.json")
VIDEO_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_video.json")
IMAGE_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_image.json")
ETAS = {"shorts": lambda argv, env: ShortsEta(argv, env, SHORTS_TIMINGS),
        "video": lambda argv, env: VideoEta(argv, VIDEO_TIMINGS, env),
        "image": lambda argv, env: ImageEta(argv, env, IMAGE_TIMINGS)}
# What panel/shorts.py, panel/video.py and panel/image.py each add: load_run, describe_runs, continue_run,
# discard_run, delete_video, captions and context.
PIPELINES = {}
# A phone photo or a song is well under this; the limit only stops a mistake filling the disk.
MAX_UPLOAD_BYTES = 200 * 2 ** 20
# Keeps the pipeline and nvidia-smi from flashing a console window when the
# server itself runs windowless (start_webapp.pyw).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# How long a killed process gets to disappear before it counts as surviving.
STOP_WAIT = 10
STOP_FAILED = "It could not be stopped. End it in Task Manager (python.exe)."
# start_webapp.pyw compares this with the code on disk to replace a server that
# predates an update (templates are cached and Python only loads at start).
STARTED_AT = time.time()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# What MediaPipe's native code prints to stderr while it loads, on every lip-sync
# check. Warnings and notices only; its errors (E0000, F0000) still reach the log.
NATIVE_NOISE = re.compile(r"^[WI]\d{4} \d\d:\d\d:\d\d|^WARNING: Logging before InitGoogle|^INFO: Created TensorFlow Lite")


def kill_tree(pid):
    """Kills a process and every process it started: terminate() alone would leave its ffmpeg renders running."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, creationflags=NO_WINDOW)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Job:
    """
    Tracks one process at a time. Rendering saturates the CPU and GPU, so a second
    concurrent run would only make both slower. Uploads hardly touch either, so
    they get a Job of their own and can run during a render.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        # {"n": sequence number, "t": "HH:MM:SS", "text": ...}; the page appends by n.
        self.lines = []
        self.seq = 0
        self.started_at = None
        self.finished_at = None
        self.returncode = None
        self.command = None
        self.progress = {}
        self.eta = None
        self.stopped = False
        self.closed = False
        self.pipeline = "shorts"

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    @property
    def idle(self):
        """Past running: _drain is done with the last process too, so a new one can't get mixed up with it."""
        return self.process is None or self.finished_at is not None

    def start(self, argv, env=None, keep_log=False, pipeline="shorts"):
        with self.lock:
            if self.closed or not self.idle:
                return False
            eta = ETAS[pipeline](argv, env)
            process = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                # Unbuffered child output, otherwise the log only appears at the end.
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})},
                creationflags=NO_WINDOW,
                # Its own process group, so stop() can take ffmpeg down with it.
                start_new_session=os.name != "nt",
            )
            if not keep_log:
                self.lines = []
            self.started_at = datetime.now()
            self.first_line = self.seq  # Its own lines start after this, if the log was kept from the job before
            self.finished_at = None
            self.returncode = None
            self.command = " ".join(argv[1:])
            self.progress = {}
            self.eta = eta
            self.pipeline = pipeline
            self.stopped = False
            self.process = process

        threading.Thread(target=self._drain, daemon=True).start()
        return True

    def _drain(self):
        process = self.process
        for line in process.stdout:
            line = line.rstrip()
            # ffmpeg's progress carriage-returns would flood the log.
            if not line or line.startswith("frame=") or line.startswith("[") or NATIVE_NOISE.match(line):
                continue
            self.log(line)
            self._track(line)
            # Keep memory bounded on long batches.
            if len(self.lines) > 2000:
                del self.lines[:500]
        process.wait()
        if process.returncode == 0:
            self.eta.finish()
        self.returncode = process.returncode
        self.finished_at = datetime.now()

    def log(self, text):
        self.seq += 1
        self.lines.append({"n": self.seq, "t": datetime.now().strftime("%H:%M:%S"), "text": text})

    # Lines main.py and its modules print, turned into structured progress.
    PATTERNS = {
        "shorts": [
            ("run", re.compile(r"===== RUN (\d+)/(\d+) =====")),
            ("run_id", re.compile(r"Run folder: (run_\w+)")),
            ("stage", re.compile(r"\u25b6 Stage (\d+)/(\d+): (.+)")),
            ("scenes", re.compile(r"narration for (\d+) scenes|Generation for (\d+) scenes")),
            ("clip", re.compile(r"Clip (\d+)/(\d+) \u00b7 scene (\d+): (\d+)s ([\w-]+)")),
            ("rejected", re.compile(r"Rejected: (.*) \(attempt (\d+)/(\d+)\)")),
            ("render", re.compile(r"Processing Scene (\d+)")),
            ("upload", re.compile(r"\U0001f4fc (\d+)/(\d+): (.+)")),
        ],
        "video": [
            ("run_id", re.compile(r"Run folder: (run_\w+)")),
            ("stage", re.compile(r"\u25b6 Stage (\d+)/(\d+): (.+)")),
            ("shot", re.compile(r"Clip (\d+)/(\d+) \u00b7 (.+) \u00b7 (\d+)s")),
            ("retry", re.compile(r"(?:Rejected: |\u274c )(.*) \(attempt (\d+)/(\d+)\)")),
            ("part", re.compile(r"Preparing clip (\d+)/(\d+)")),
        ],
        "image": [
            ("run_id", re.compile(r"Run folder: (image_\w+)")),
            ("stage", re.compile(r"\u25b6 Stage (\d+)/(\d+): (.+)")),
            ("image", re.compile(r"Image (\d+)/(\d+) \u00b7 (.+)")),
            ("retry", re.compile(r"(?:Rejected: |\u274c )(.*) \(attempt (\d+)/(\d+)\)")),
        ],
    }

    def _track(self, line):
        p = self.progress
        for kind, pattern in self.PATTERNS[self.pipeline]:
            m = pattern.search(line)
            if not m:
                continue
            if kind == "run":
                p["run"] = [int(m.group(1)), int(m.group(2))]
                self.eta.video(int(m.group(1)), int(m.group(2)))
            elif kind == "run_id":
                p["run_id"] = m.group(1)
            elif kind == "stage":
                p["stage"] = [int(m.group(1)), int(m.group(2)), m.group(3).strip()]
                p["stage_started"] = time.time()
                p.pop("detail", None)
                self.eta.stage(*p["stage"])
            elif kind == "scenes":
                p["scenes"] = int(m.group(1) or m.group(2))
                self.eta.scenes(p["scenes"])
            elif kind == "clip":
                p["detail"] = f"Clip {m.group(1)} of {m.group(2)} - scene {m.group(3)}, {m.group(4)}s {m.group(5)}"
                self.eta.clip(int(m.group(1)), int(m.group(2)), int(m.group(4)))
            elif kind == "rejected":
                self.eta.clip_rejected(int(m.group(2)), int(m.group(3)), m.group(1))
            elif kind == "render":
                total = p.get("scenes")
                p["detail"] = f"Scene {m.group(1)}" + (f" of {total}" if total else "")
                self.eta.scene()
            elif kind == "shot":
                p["detail"] = f"Clip {m.group(1)} of {m.group(2)}: {m.group(3)}, {m.group(4)}s"
                self.eta.clip(int(m.group(1)), int(m.group(2)), int(m.group(4)))
            elif kind == "image":
                p["detail"] = f"Image {m.group(1)} of {m.group(2)}: {m.group(3)}"
                self.eta.image(int(m.group(1)), int(m.group(2)))
            elif kind == "retry":
                self.eta.clip_retry(int(m.group(2)), int(m.group(3)), m.group(1))
            elif kind == "part":
                p["detail"] = f"Preparing clip {m.group(1)} of {m.group(2)}"
            elif kind == "upload":
                p["upload"] = [int(m.group(1)), int(m.group(2)), m.group(3).strip()]
            return

    def stop(self):
        """
        Kills the process and everything it started, and waits for it to go.
        False if nothing was running, or if it survived: then `running` stays True.
        """
        with self.lock:
            if not self.running:
                return False
            self.stopped = True
            pid = self.process.pid
            kill_tree(pid)
            try:
                self.process.wait(timeout=STOP_WAIT)
            except subprocess.TimeoutExpired:
                self.stopped = False
                self.log(f"⚠️ Could not stop process {pid}.")
                return False
            self.log("🛑 Stopped by user.")
        return True

    def close(self):
        """Nothing starts from here on, so a job can't slip in while the server shuts down."""
        with self.lock:
            self.closed = True

    @property
    def uploading(self):
        """In a run's own Upload step, when its video sits in Ready with no topic recorded yet."""
        return self.running and self.progress.get("stage", [None, None, None])[2] == "Upload"

    def snapshot(self):
        return {
            "running": self.running,
            "lines": list(self.lines),
            "returncode": self.returncode,
            "stopped": self.stopped,
            "command": self.command,
            "pipeline": self.pipeline,
            "started_at": self.started_at.strftime("%H:%M:%S") if self.started_at else None,
            "elapsed": int(((self.finished_at or datetime.now()) - self.started_at).total_seconds()) if self.started_at else 0,
            "progress": {**self.progress,
                         "stage_elapsed": int(time.time() - self.progress["stage_started"])
                         if "stage_started" in self.progress else 0,
                         "eta": self.eta.estimate() if self.eta and self.running else None},
        }


job = Job()
uploads = Job()


class JobQueue:
    """
    The jobs waiting for a Job, oldest first, each keeping the options it was
    queued with. A failed job doesn't stop the ones after it, as in a batch.
    Stop does: pressing it means nothing should start by itself, so what is
    queued waits for Resume. Saved to `path`, so what was waiting comes back after
    a Shut down or crash, on hold for the same reason.
    """

    KEPT_RESULTS = 5  # Finished shows the last five, and starts empty after a restart

    def __init__(self, job, prepare, blocked=lambda: False, path=None):
        self.job = job
        self.prepare = prepare  # Last step before a job starts; returns why it can't run, or None.
        self.blocked = blocked
        self.path = path
        self.lock = threading.RLock()  # snapshot_with_job() holds it around snapshot()
        self.items = []
        self.current = None
        self.results = []
        self.held = False
        self.next_id = 1
        self.watcher = None
        self._restore()

    def _restore(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                items = [{**item, "waited": True} for item in json.load(f)]
            self.next_id = max((item["id"] for item in items), default=0) + 1
        except (OSError, ValueError, TypeError, KeyError):
            return  # Nothing saved, or a broken file: an empty queue
        self.items = items
        self.held = bool(items)

    def _save(self):
        # Once closed, the file keeps what was queued at Shut down.
        if not self.path or self.job.closed:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False)
            os.replace(self.path + ".tmp", self.path)
        except OSError as e:
            self.job.log(f"⚠️ Could not save the queue, it won't survive a restart: {e}")

    def _ensure_watcher(self):
        # Here and not at import: gunicorn forks after importing, and threads don't survive that.
        if not self.watcher:
            self.watcher = threading.Thread(target=self._watch, daemon=True)
            self.watcher.start()

    def add(self, item):
        """Starts the job if nothing is ahead of it, otherwise queues it. Returns True if it is queued."""
        # Generate's "Start later": it waits in the queue until it's started.
        if has_request_context() and request.path.startswith("/generate/") and request.form.get("start_later"):
            item["later"] = True
        with self.lock:
            self._ensure_watcher()
            item["id"] = self.next_id
            self.next_id += 1
            # Ahead of those waiting to be started, which stay at the end.
            at = len(self.items) if item.get("later") else next((i for i, waiting in enumerate(self.items) if waiting.get("later")), len(self.items))
            self.items.insert(at, item)
            self._advance()
            item["waited"] = item in self.items
            self._save()
            return item["waited"]

    def _watch(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                self._advance()

    def _advance(self):
        if self.current and self.job.idle:
            self._finish_current()
        if not self.items:
            self.held = False
        elif self.job.idle and not self.held and not self.blocked():
            # The first that may start: one waiting to be started lets those behind it go first.
            ready = next((i for i, waiting in enumerate(self.items) if not waiting.get("later")), None)
            if ready is None:
                return
            item = self.items.pop(ready)
            self._save()
            self._start(item)

    def _finish_current(self):
        item, self.current = self.current, None
        code = self.job.returncode
        if self.job.stopped:
            self._record(item, "stopped")
            note_outcome(self.job, item, None)
        elif code == 0:
            self._record(item, "done")
            note_outcome(self.job, item, NOT_RENDERED if item.get("plan_only") else FIRST_ONLY if item.get("first_only") else None)
        else:
            self._record(item, "failed", f"exit {code}")
            note_outcome(self.job, item, {"outcome": "failed", "why": why_it_failed(self.job, code)})
            if self.items:
                self.job.log(f"❌ {item['label']} failed (exit {code}).")

    def _start(self, item):
        if item.get("waited"):
            self.job.log(f"===== Next in the queue: {item['label']} =====")
        try:
            error = self.prepare(item)
            # A job that waited adds to the log, so the one before it can still be read.
            if not error and self.job.start(item["argv"], item["env"], keep_log=item.get("waited", False), pipeline=item.get("pipeline", "shorts")):
                self.current = item
                return
            error = error or "Another job took its place."  # A setup check, started in the moment between two jobs
        except Exception as e:  # One bad job must not take the watcher thread, and the queue, down with it
            error = str(e)
        item["error"] = error
        self.job.log(f"⚠️ Skipped {item['label']}: {error}")
        self._record(item, "skipped", error)

    def _record(self, item, outcome, note=None):
        notify_ended(item, outcome, note)
        self.results.append({"label": item["label"], "outcome": outcome, "note": note, "pipeline": item.get("pipeline"),
                             "at": datetime.now().strftime("%H:%M"), "t": time.time()})
        del self.results[:-self.KEPT_RESULTS]

    def hold(self):
        with self.lock:
            self.held = bool(self.items)

    def resume(self):
        with self.lock:
            self._ensure_watcher()
            self.held = False
            self._advance()

    def start(self, item_id):
        """Starts a job that was waiting to be: it goes next, ahead of the rest."""
        with self.lock:
            item = next((item for item in self.items if item["id"] == item_id), None)
            if not item:
                return False
            item.pop("later", None)
            self.items.remove(item)
            self.items.insert(0, item)
            self._advance()
            self._save()
            return True

    def remove(self, item_id):
        with self.lock:
            kept = [item for item in self.items if item["id"] != item_id]
            removed = len(kept) != len(self.items)
            self.items = kept
            self._advance()
            self._save()
            return removed

    def close(self):
        """Lets nothing start again, and leaves what is waiting in the file. The running job is the caller's to stop."""
        with self.lock:
            self._save()
            self.job.close()
            self.items = []
            self.held = False

    def pending(self):
        """Every job not finished yet, the running one included."""
        with self.lock:
            self._advance()
            return ([self.current] if self.current else []) + list(self.items)

    def snapshot_with_job(self):
        """
        The job and this queue at one moment. Read apart, a status could fall between a job ending and the next one
        starting: the old job done and the queue already empty, so nothing at all for a poll.
        """
        with self.lock:
            return self.job.snapshot(), self.snapshot()

    def snapshot(self):
        with self.lock:
            self._advance()
            public = lambda item: {key: item.get(key) for key in ("id", "label", "detail", "estimate", "videos", "pipeline", "later")}
            return {"items": [public(item) for item in self.items],
                    "current": public(self.current) if self.current else None,
                    "results": list(self.results),
                    "held": self.held}


def notify_ended(item, outcome, note=None):
    """
    Tells your phone a job ended, through ntfy (https://ntfy.sh), when NTFY_TOPIC is set in .env: install the ntfy app
    and subscribe to that topic. NTFY_SERVER for your own server. Sent in the background: a slow server holds nothing up.
    """
    topic = os.getenv("NTFY_TOPIC")
    if not topic or outcome not in ("done", "failed"):
        return
    title = f"{'Finished' if outcome == 'done' else 'Failed'}: {item.get('label') or 'a job'}"
    side = {"shorts": "Shorts", "video": "Video", "image": "Image"}.get(item.get("pipeline"), "Upload")
    message = f"{side}{f' · {note}' if note else ''}"
    url = f"{(os.getenv('NTFY_SERVER') or 'https://ntfy.sh').rstrip('/')}/{topic}"

    def send():
        try:
            requests.post(url, data=message.encode("utf-8"), timeout=10,
                          headers={"Title": title.encode("utf-8"), "Tags": "white_check_mark" if outcome == "done" else "x"})
        except requests.RequestException:
            pass  # A notification that doesn't arrive costs nothing else
    threading.Thread(target=send, daemon=True).start()


def form_options(form, takes):
    """
    The workflow's own options as the form sets them (core/workflow_options.py): only what differs from the workflow as
    saved. Returns (chosen, error). A checkbox sends a hidden "0" before its own "1", so the last value is the one that
    counts. A way for the workflow to write its own prompt is chosen with the LLM instead, so it isn't here.
    """
    from core import workflow_options
    options = takes.get("options") or []
    chosen = {}
    for option in options:
        name = f"opt_{option['id']}"
        if option.get("role"):
            continue
        if option["kind"] == "strength":
            if f"{name}_on" in form:
                on = form.getlist(f"{name}_on")[-1] == workflow_options.ON
                chosen[option["id"]] = (form.get(f"{name}_strength") or option.get("usual") or "1").strip() if on else workflow_options.OFF
        elif name in form:
            chosen[option["id"]] = form.getlist(name)[-1].strip()
    chosen = {key: value for key, value in chosen.items() if value != next(option["default"] for option in options if option["id"] == key)}
    error = workflow_options.check(options, chosen)
    return (None, error) if error else (chosen, None)


def password_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not os.getenv("APP_PASSWORD") or session.get("authed"):
            return view(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return wrapper


def list_videos(folder):
    directory = FOLDERS[folder]
    if not os.path.isdir(directory):
        return []

    captions = PIPELINES[FOLDER_PIPELINES[folder]].captions()
    videos = []
    for name in os.listdir(directory):
        if not name.endswith(".mp4"):
            continue
        stat = os.stat(os.path.join(directory, name))
        videos.append({
            "name": name,
            **captions(name),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "created": datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m %H:%M"),
            "mtime": stat.st_mtime,
        })

    videos.sort(key=lambda v: v["mtime"], reverse=True)
    return videos


def safe_video_path(folder, name):
    """
    Resolves name inside one of the video folders, refusing anything that escapes it.
    """
    directory = FOLDERS.get(folder)
    if not directory:
        abort(404)
    path = os.path.abspath(os.path.join(directory, name))
    if os.path.dirname(path) != os.path.abspath(directory) or not os.path.exists(path):
        abort(404)
    return path


def saved_at(path):
    """When a run was last saved, as the panel shows it. Empty if that can't be told."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d %b %H:%M")
    except OSError:
        return ""


def lan_ip():
    """The address other devices on the home network reach this PC at."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 80))  # No packet is sent; this only picks the outgoing interface.
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


_comfyui_cache = {"at": 0.0, "online": False}
_comfyui_start_lock = threading.Lock()
_comfyui_start_requested_at = 0.0
_comfyui_process = None  # Only one this server started
# Seconds a run waits for a ComfyUI that Generate just launched to load its models.
COMFYUI_STARTUP_WAIT = 180


def comfyui_online():
    # Cached briefly: every open browser polls status every 2s.
    if time.time() - _comfyui_cache["at"] > 10:
        url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
        try:
            _comfyui_cache["online"] = requests.get(f"{url}/system_stats", timeout=2).status_code == 200
        except requests.RequestException:
            _comfyui_cache["online"] = False
        _comfyui_cache["at"] = time.time()
    return _comfyui_cache["online"]


# What each node of the workflows is doing, in plain words.
COMFY_LABELS = {
    "UNETLoader": "Loading the model", "UnetLoaderGGUF": "Loading the model", "CLIPLoader": "Loading the text encoder",
    "DualCLIPLoader": "Loading the text encoder", "DualCLIPLoaderGGUF": "Loading the text encoder",
    "VAELoader": "Loading the VAE", "VAELoaderKJ": "Loading the audio VAE",
    "LatentUpscaleModelLoader": "Loading the upscaler",
    "LoadImage": "Loading the reference image", "LoadAudio": "Loading the narration",
    "CLIPTextEncode": "Reading the prompt", "LTXVAudioVAEEncode": "Encoding the narration",
    "LTXVImgToVideoInplace": "Placing the reference image", "LTXVPreprocess": "Preparing the image",
    "SamplerCustomAdvanced": "Generating video", "LTXVLatentUpsampler": "Upscaling",
    "VAEDecodeTiled": "Decoding frames", "LTXVAudioVAEDecode": "Decoding audio",
    "VHS_VideoCombine": "Saving the video", "TextEncodeQwenImageEditPlus": "Reading the prompt", "KSampler": "Generating the image",
    "VAEEncode": "Encoding the image", "VAEDecode": "Decoding the image", "PreviewImage": "Saving the image",
}


class ComfyMonitor:
    """
    Listens on ComfyUI's websocket (as the pipeline's client id, since ComfyUI
    only sends progress to the client that queued the job) and keeps the latest
    state: which node is running, sampling steps, queue size.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"connected": False, "queue": 0}
        self.nodes = {}
        self.samplers_seen = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _base(self):
        return (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")

    def _loop(self):
        client_id = comfyui_client_id()
        while True:
            url = self._base().replace("http", "ws", 1) + f"/ws?clientId={client_id}"
            try:
                ws = websocket.create_connection(url, timeout=10)
                ws.settimeout(None)
                with self.lock:
                    self.state["connected"] = True
                while True:
                    message = ws.recv()
                    if isinstance(message, str):  # Binary frames are sampler previews
                        self._handle(json.loads(message))
            except Exception:
                with self.lock:
                    self.state = {"connected": False, "queue": 0}
                time.sleep(5)

    def _load_nodes(self, prompt_id):
        """Node id -> class for the running prompt, from ComfyUI's queue."""
        try:
            queue = requests.get(f"{self._base()}/queue", timeout=5).json()
            for entry in queue.get("queue_running", []):
                if entry[1] == prompt_id:
                    return {k: v.get("class_type", "") for k, v in entry[2].items()}
        except Exception:
            pass
        return {}

    def _handle(self, msg):
        kind, data = msg.get("type"), msg.get("data") or {}
        with self.lock:
            s = self.state
            if kind == "status":
                s["queue"] = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
            elif kind == "execution_start":
                self.nodes = self._load_nodes(data.get("prompt_id"))
                self.samplers_seen = []
                s.update(active=True, started=time.time(), node=None, label="Starting", value=None, max=None)
            elif kind == "executing":
                node = data.get("node")
                if node is None:
                    s.update(active=False, label="Idle", node=None, value=None, max=None)
                    return
                cls = self.nodes.get(str(node), "")
                label = COMFY_LABELS.get(cls, cls or f"Node {node}")
                if cls == "SamplerCustomAdvanced":
                    if node not in self.samplers_seen:
                        self.samplers_seen.append(node)
                    passes = sum(1 for c in self.nodes.values() if c == cls)
                    label += f" - pass {self.samplers_seen.index(node) + 1} of {passes}"
                s.update(active=True, node=node, label=label, value=None, max=None)
            elif kind == "progress":
                s.update(value=data.get("value"), max=data.get("max"))
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                s.update(active=False, label="Idle" if kind == "execution_success" else kind.replace("_", " "),
                         node=None, value=None, max=None)

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
        if s.get("started") and s.get("active"):
            s["elapsed"] = int(time.time() - s["started"])
        s.pop("started", None)
        return s


comfy_monitor = None
_gpu_cache = {"at": 0.0, "value": None}


def gpu_stats():
    """Utilisation / VRAM / power from nvidia-smi, cached for a few seconds."""
    if time.time() - _gpu_cache["at"] < 3:
        return _gpu_cache["value"]
    _gpu_cache["at"] = time.time()
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            creationflags=NO_WINDOW,
        ).stdout.strip().splitlines()[0]
        util, used, total, power, temp = [x.strip() for x in out.split(",")]
        _gpu_cache["value"] = {"util": int(float(util)), "mem_used": round(float(used) / 1024, 1),
                               "mem_total": round(float(total) / 1024, 1),
                               "power": int(float(power)), "temp": int(float(temp))}
    except Exception:
        _gpu_cache["value"] = None
    return _gpu_cache["value"]


def comfyui_launcher():
    path = os.getenv("COMFYUI_LAUNCHER")
    return path if path and os.path.exists(path) else None


def cancel_pipeline_clips():
    """
    Cancels the avatar clips ComfyUI has for a stopped run, rendering or waiting.
    Only the pipeline's, by client id: a render you started in ComfyUI yourself
    carries on. ComfyUI itself keeps running, with its models loaded for the next run.
    """
    url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
    client_id = comfyui_client_id()
    try:
        queue = requests.get(f"{url}/queue", timeout=5).json()
        waiting = [entry[1] for entry in queue.get("queue_pending", []) if entry[3].get("client_id") == client_id]
        # Waiting ones first, or the next clip starts as soon as this one stops.
        if waiting:
            requests.post(f"{url}/queue", json={"delete": waiting}, timeout=5)
        for entry in queue.get("queue_running", []):
            if entry[3].get("client_id") == client_id:
                requests.post(f"{url}/interrupt", json={"prompt_id": entry[1]}, timeout=5)
    except requests.ConnectionError:
        pass  # ComfyUI isn't running, so it has nothing to cancel
    except (requests.RequestException, ValueError, LookupError, AttributeError) as e:
        job.log(f"⚠️ Could not cancel the ComfyUI clip, it may keep rendering: {e}")


def remove_partial_videos():
    """A stitch that gets killed leaves its unfinished file behind in the output folder."""
    try:
        for folder in (SHORTS_READY, VIDEO_OUTPUT):
            for name in os.listdir(folder) if os.path.isdir(folder) else ():
                if name.endswith(".part"):
                    os.remove(os.path.join(folder, name))
        # Each set of images has a folder of its own.
        for folder, _, names in os.walk(IMAGE_OUTPUT):
            for name in names:
                if name.endswith(".part"):
                    os.remove(os.path.join(folder, name))
    except OSError as e:
        job.log(f"⚠️ Could not remove the unfinished video: {e}")


def close_comfyui():
    """Closes ComfyUI if this server started it. One you started yourself isn't the panel's to close."""
    if _comfyui_process and _comfyui_process.poll() is None:
        kill_tree(_comfyui_process.pid)


def comfyui_command(launcher):
    """
    The ComfyUI command line from a start script such as the portable build's
    run_nvidia_gpu.bat, with --disable-auto-launch added so no browser tab
    opens: the panel only needs its API. None if the script has no main.py line.
    """
    try:
        with open(launcher, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if "main.py" in line and not line.lower().startswith(("rem", "::", "echo")):
                    return line if "--disable-auto-launch" in line else line + " --disable-auto-launch"
    except OSError:
        pass
    return None


def launch_comfyui(launcher):
    """
    Starts ComfyUI in its own minimised console window, so it can be watched
    (or closed to stop it) and outlives a server restarting for new code; Shut
    down closes it. (Not 'start': it only takes a quoted first argument as the
    window title and would open a folder named ComfyUI.)
    """
    global _comfyui_process
    command = comfyui_command(launcher)
    if os.name != "nt":
        _comfyui_process = subprocess.Popen(command or [launcher], shell=bool(command),
                                            cwd=os.path.dirname(launcher), start_new_session=True)
        return
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 7  # SW_SHOWMINNOACTIVE: minimised, without stealing focus
    _comfyui_process = subprocess.Popen(
        f"cmd /c {command}" if command else ["cmd", "/c", launcher],
        cwd=os.path.dirname(launcher),
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
        startupinfo=startup,
    )


def start_comfyui():
    """Starts ComfyUI if a launcher is configured. Returns True when it is (still) booting."""
    global _comfyui_start_requested_at
    with _comfyui_start_lock:
        _comfyui_cache["at"] = 0.0
        if comfyui_online():
            return False
        launcher = comfyui_launcher()
        if not launcher:
            return False
        if time.time() - _comfyui_start_requested_at < COMFYUI_STARTUP_WAIT:
            return True  # Launched moments ago and still loading.
        try:
            launch_comfyui(launcher)
        except OSError as e:
            raise RuntimeError(f"Could not start ComfyUI: {e}") from e
        _comfyui_start_requested_at = time.time()
        _comfyui_cache["at"] = 0.0
        return True


_quota_cache = {"at": 0.0, "value": None}


def cached_quota():
    # Cached briefly: every open browser polls status every 2s.
    if time.time() - _quota_cache["at"] > 5:
        _quota_cache["value"] = quota_summary()
        _quota_cache["at"] = time.time()
    return _quota_cache["value"]


def item_images(item):
    return ([item["image"]] if item.get("image") else []) + item.get("references", [])


def prepare_run(item):
    """ComfyUI starts when the run does, not when it is queued: it may not be needed for hours."""
    pipeline = item.get("pipeline", "shorts")
    if item.get("resume") and not PIPELINES[pipeline].load_run(item["resume"]):
        return "its saved run is gone"
    if not all(os.path.exists(path) for path in item_images(item)):
        return "its image is gone"
    if item.get("plan_only"):
        return None  # Writing the prompts needs no GPU
    try:
        always = pipeline in ("video", "image")
        needs_comfyui = always or (item["form"].get("avatar_source") or "comfyui") == "comfyui"
        if needs_comfyui and (always or (item["form"].get("avatar_clips") or "auto") != "0") and start_comfyui():
            item["env"] = {**(item.get("env") or {}), "COMFYUI_STARTUP_WAIT": str(COMFYUI_STARTUP_WAIT)}
    except RuntimeError as e:
        return str(e)
    return None


def prepare_upload(item):
    """Drops videos deleted or uploaded while this upload waited its turn."""
    gone = [name for name in item["videos"] if not os.path.exists(os.path.join(FOLDERS["output"], name))]
    if len(gone) == len(item["videos"]):
        return "no longer in Ready"
    item["argv"] = [arg for arg in item["argv"] if arg not in gone]
    item["videos"] = [name for name in item["videos"] if name not in gone]
    return None


run_queue = JobQueue(job, prepare_run, path=os.path.join(LOGS_DIR, "queue_runs.json"))
# A run's own upload step is waited out, when its video sits in Ready with no topic recorded yet.
upload_queue = JobQueue(uploads, prepare_upload, blocked=lambda: job.uploading,
                        path=os.path.join(LOGS_DIR, "queue_uploads.json"))
QUEUES = {"run": run_queue, "upload": upload_queue}


def work_left():
    return job.running or uploads.running or run_queue.pending() or upload_queue.pending()


_runs_cache = {"at": 0.0, "value": []}


def busy_runs():
    """The saved runs a job has in hand: the one going, and those queued to continue."""
    busy = {item.get("resume") for item in run_queue.pending()}
    if job.running:
        busy.add(job.progress.get("run_id"))
    return busy


def saved_runs():
    # Cached briefly: every open browser polls status every 2s.
    if time.time() - _runs_cache["at"] > 3:
        _runs_cache["at"] = time.time()
        try:
            _runs_cache["value"] = [run for pipeline in PIPELINES.values() for run in pipeline.describe_runs()]
        except Exception as e:  # This is polled for the whole page: one odd run must not freeze it
            job.log(f"⚠️ Could not list the saved runs: {e}")
    return _runs_cache["value"]


# --- Why a run made nothing: kept in its folder, for Output to show beside what it did make ---

OUTCOME_FILE = "outcome.json"
FIRST_ONLY = {"outcome": "first one made",
              "why": "You asked to try one first, so it stopped after the first. Open Prompts to see it and change the rest, "
                     "or continue to make them."}
NOT_RENDERED = {"outcome": "not rendered",
                "why": "You asked to read the prompts first (Let me read the prompts before rendering), so it stopped once "
                       "they were written. Open Prompts to change them, then render it."}
# The lines that say what went wrong, as the pipelines print them.
TROUBLE = re.compile(r"❌|⚠️|Error|Exception|Traceback|failed|refused|can't|could not", re.I)
MOST_WHY_LINES = 4


def why_it_failed(job, code):
    """The run's own last lines that say what went wrong, or that it didn't say."""
    lines = [line["text"].strip() for line in job.lines if line["n"] > getattr(job, "first_line", 0)]
    # A Generate turned down meanwhile is about another job, not this one.
    said = [line for line in lines if TROUBLE.search(line) and not line.startswith(("=====", REFUSED))]
    if said:
        return "\n".join(dict.fromkeys(said[-MOST_WHY_LINES:]))
    last = lines[-1] if lines else "nothing"
    return f"It ended (exit {code}) without saying why. Its last line: {last}"


def note_outcome(job, item, outcome):
    """Keeps why a run made nothing in its folder; None clears it, once it has made what it was asked for."""
    pipeline = item.get("pipeline")
    run_id = (job.progress or {}).get("run_id") or item.get("resume")
    if pipeline not in PIPELINES or not run_id:
        return
    try:
        state = PIPELINES[pipeline].load_run(run_id)
        if not state:
            return
        # Tried first, but there was only the one: it's done, there's nothing to say.
        if outcome is FIRST_ONLY and (getattr(state, "final", None) or (getattr(state, "finished", False) is True)):
            outcome = None
        path = os.path.join(state.folder, OUTCOME_FILE)
        if outcome:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({**outcome, "at": time.time()}, f, ensure_ascii=False)
        elif os.path.exists(path):
            os.remove(path)
    except (OSError, ValueError, AttributeError) as e:
        job.log(f"⚠️ Could not note how {item.get('label')} ended: {e}")
    _runs_cache["at"] = 0.0


def read_outcome(folder):
    """What note_outcome() kept for a run, or None."""
    try:
        with open(os.path.join(folder, OUTCOME_FILE), "r", encoding="utf-8") as f:
            found = json.load(f)
        return found if isinstance(found, dict) and found.get("outcome") else None
    except (OSError, ValueError):
        return None


def made_in_this_run():
    """
    What the job going now has already made: each finished image of a set, under Finished as it comes, rather than
    only once the whole set is done.
    """
    run_id = (job.progress or {}).get("run_id")
    if not job.running or job.pipeline != "image" or not run_id:
        return []
    try:
        state = PIPELINES["image"].load_run(run_id)
        started = job.started_at.timestamp() if job.started_at else 0
        made = []
        for shot in (state.done if state else []):
            made_at = os.path.getmtime(state.image_path(shot))
            # Only what this job made: a redo, or a set continued, kept the rest from before.
            if made_at < started:
                continue
            made.append({"label": f"{state.title or state.brief} · {shot.get('title') or shot['number']}", "outcome": "done",
                         "pipeline": "image", "at": datetime.fromtimestamp(made_at).strftime("%H:%M"), "t": made_at})
        return made
    except (OSError, ValueError, AttributeError, KeyError):
        return []  # Being written as we look: the next poll has it


def stopped_runs():
    """What can be continued, with why it stopped if it didn't finish on its own. A finished film is reached from its video instead."""
    busy = busy_runs()
    return [run for run in saved_runs() if run["id"] not in busy and not run.get("finished")]


def run_pipeline(run_id):
    """Shorts' runs are numbered within their batch: run_20261101_120000_1. Image's start image_."""
    if run_id.startswith("image_"):
        return PIPELINES["image"]
    return PIPELINES["shorts" if re.fullmatch(r"run_\d{8}_\d{6}_\d+", run_id) else "video"]


def stop_run(start_next=False):
    """Kills the run with its ffmpeg renders and ComfyUI clips. What is queued waits for Resume, unless start_next."""
    # Before the kill, or the next job could start as soon as this one dies.
    if job.running:
        run_queue.hold()
    stopped = job.stop()
    if stopped:
        cancel_pipeline_clips()
        remove_partial_videos()
        _runs_cache["at"] = 0.0
        # Only now: the clean-up above would take the next job's clips and files with it.
        if start_next:
            run_queue.resume()
    return stopped


@app.errorhandler(413)
def too_large(error):
    return {"ok": False, "error": f"That file is over {MAX_UPLOAD_BYTES // 2 ** 20} MB."}, 413


@app.route("/login", methods=["GET", "POST"])
def login():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        # Constant-time compare so the password cannot be guessed by timing.
        if secrets.compare_digest(request.form.get("password", ""), expected):
            session["authed"] = True
            session.permanent = True
            target = request.args.get("next") or ""
            # Only back into this site: a link can't send you elsewhere after signing in.
            return redirect(target if re.fullmatch(r"/(?![/\\]).*", target) else url_for("index"))
        error = "Wrong password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


_saved_workflows_cache = {"at": 0.0, "online": None, "value": []}
_saved_workflows_lock = threading.Lock()


def saved_workflows():
    # Cached briefly: it asks ComfyUI about every node it has. Not across ComfyUI starting, which is when they appear.
    online = comfyui_online()
    with _saved_workflows_lock:
        cache = _saved_workflows_cache
        if time.time() - cache["at"] > 30 or cache["online"] != online:
            url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
            cache.update(at=time.time(), online=online, value=list_saved(url, SAVED_WORKFLOWS_DIR))
        return cache["value"]


def on_this_pc():
    return request.remote_addr in ("127.0.0.1", "::1")


def can_open_folders():
    """Explorer opens on the PC running the server, which is no use to someone on their phone."""
    return os.name == "nt" and on_this_pc()


@app.route("/")
@app.route("/shorts", endpoint="shorts_page")
@app.route("/video", endpoint="video_page")
@app.route("/image", endpoint="image_page")
@password_required
def index():
    port = request.host.rsplit(":", 1)[-1] if ":" in request.host else "80"
    context = {}
    for pipeline in PIPELINES.values():
        context.update(pipeline.context())
    protected = bool(os.getenv("APP_PASSWORD"))
    tailscale_url, tailscale_problem = tailscale.reach(port) if protected else (None, None)
    return render_template(
        "index.html",
        protected=protected,
        phone_url=f"http://{lan_ip()}:{port}",
        tailscale_url=tailscale_url,
        tailscale_problem=tailscale_problem,
        can_open_folders=can_open_folders(),
        # The page you asked for by address, or else the one it last showed.
        initial_pipeline=request.path.strip("/"),
        **context,
    )


@app.route("/runs/<run_id>/continue", methods=["POST"])
@password_required
def continue_run(run_id):
    """Starts a stopped run again from where it got to, or queues that behind the run in progress."""
    return run_pipeline(run_id).continue_run(run_id)


@app.route("/runs/<run_id>/discard", methods=["POST"])
@password_required
def discard_run(run_id):
    return run_pipeline(run_id).discard_run(run_id)


@app.route("/stop", methods=["POST"])
@password_required
def stop():
    return {"ok": stop_run(), "error": STOP_FAILED if job.running else None}


@app.route("/skip", methods=["POST"])
@password_required
def skip():
    """Stops the run so the next in the queue can start. It waits under Stopped runs until you continue it."""
    # A film's plan is saved once written: stopped before that, there would be nothing to continue.
    _runs_cache["at"] = 0.0
    if job.progress.get("run_id") not in {run["id"] for run in saved_runs()}:
        return {"ok": False, "error": "This run has nothing saved to continue from yet. Try again once its plan is written."}, 409
    stopped = stop_run(start_next=True)
    # Once stopped, what's running is the next job.
    return {"ok": stopped, "error": STOP_FAILED if job.running and not stopped else None}


@app.route("/publish/stop", methods=["POST"])
@password_required
def stop_upload():
    if uploads.running:
        upload_queue.hold()
    return {"ok": uploads.stop(), "error": STOP_FAILED if uploads.running else None}


@app.route("/queue/<lane>/remove/<int:item_id>", methods=["POST"])
@password_required
def queue_remove(lane, item_id):
    if lane not in QUEUES:
        abort(404)
    return {"ok": QUEUES[lane].remove(item_id)}


@app.route("/queue/<lane>/start/<int:item_id>", methods=["POST"])
@password_required
def queue_start(lane, item_id):
    if lane not in QUEUES:
        abort(404)
    return {"ok": QUEUES[lane].start(item_id)}


@app.route("/queue/<lane>/resume", methods=["POST"])
@password_required
def queue_resume(lane):
    if lane not in QUEUES:
        abort(404)
    QUEUES[lane].resume()
    return {"ok": True}


@app.route("/shutdown", methods=["POST"])
@password_required
def shutdown():
    """
    Force-stops everything the panel is doing, then the server itself: started
    from start_webapp.pyw it has no window to close.
    """
    for queue in QUEUES.values():
        queue.close()
    stop_run()
    uploads.stop()
    # Exiting now would leave it running with nothing in the panel to stop it.
    if job.running or uploads.running:
        return {"ok": False, "error": f"{STOP_FAILED} Then shut down again."}, 500
    close_comfyui()
    # Reply first, then exit, so the browser gets its answer.
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"ok": True}


@app.route("/launcher/refresh", methods=["POST"])
def launcher_refresh():
    """
    Exits if the code changed after this server started, so start_webapp.pyw can
    start a fresh one. Only answers this PC, and never stops a run.
    """
    if not on_this_pc():
        abort(403)
    try:
        code_changed_at = float(request.form.get("code_changed_at", 0))
    except ValueError:
        abort(400)
    if code_changed_at <= STARTED_AT or work_left():
        return {"restarting": False}
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"restarting": True}


@app.route("/api/status")
@password_required
def status():
    payload, queue = run_queue.snapshot_with_job()
    payload["server_started"] = STARTED_AT
    payload["queue"] = queue
    payload["stopped_runs"] = stopped_runs()
    payload["made_now"] = made_in_this_run()
    payload["upload"] = uploads.snapshot()
    payload["upload"]["queue"] = upload_queue.snapshot()
    payload["videos"] = {folder: list_videos(folder) for folder in FOLDERS}
    payload["image_sets"] = [run for run in saved_runs() if run["pipeline"] == "image" and run["images"]]
    online = comfyui_online()
    payload["comfyui"] = {"online": online, "can_start": bool(comfyui_launcher()),
                          "starting": not online and time.time() - _comfyui_start_requested_at < COMFYUI_STARTUP_WAIT,
                          **(comfy_monitor.snapshot() if comfy_monitor else {})}
    payload["gpu"] = gpu_stats()
    payload["quota"] = cached_quota()
    return payload


PRESETS_PATH = os.path.join(LOGS_DIR, "presets.json")
PRESET_SIDES = ("shorts", "video", "image")
MOST_PRESET_NAME = 60


def load_presets():
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {side: [p for p in saved.get(side, []) if isinstance(p, dict) and p.get("name")] for side in PRESET_SIDES}
    except (OSError, ValueError, AttributeError):
        return {side: [] for side in PRESET_SIDES}


@app.route("/api/presets", methods=["GET", "POST"])
@password_required
def presets():
    """A side's settings under a name: kept here, so every device has the same ones. POST saves one, or deletes it."""
    if request.method == "GET":
        return {"ok": True, "presets": load_presets()}
    body = request.get_json(silent=True) or {}
    side, name = body.get("side"), str(body.get("name") or "").strip()[:MOST_PRESET_NAME]
    if side not in PRESET_SIDES or not name:
        return {"ok": False, "error": "A preset needs a side and a name."}, 400
    saved = load_presets()
    kept = [p for p in saved[side] if p["name"] != name]
    if not body.get("delete"):
        values = body.get("values")
        if not isinstance(values, dict):
            return {"ok": False, "error": "A preset needs its settings."}, 400
        kept.append({"name": name, "values": {str(k): v for k, v in values.items() if isinstance(v, (str, bool, int, float))}})
    saved[side] = sorted(kept, key=lambda p: p["name"].casefold())
    try:
        os.makedirs(os.path.dirname(PRESETS_PATH), exist_ok=True)
        with open(PRESETS_PATH + ".tmp", "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=1)
        os.replace(PRESETS_PATH + ".tmp", PRESETS_PATH)
    except OSError as e:
        return {"ok": False, "error": f"Could not save it: {e}"}, 500
    return {"ok": True, "presets": saved}


@app.route("/api/workflows")
@password_required
def workflows():
    """
    What runs on this pipeline. logs/workflows/not_listed.json says why anything else isn't here.
    Image also lists what can't run, with why: it has no built-in workflow to fall back on, so an empty list says nothing.
    """
    pipeline = request.args.get("pipeline") or "video"
    found = [{key: w[key] for key in ("name", "label", "takes", "problem") if key in w} for w in saved_workflows()
             if w.get("pipeline", pipeline_of(w["name"])) == pipeline and (pipeline == "image" or "problem" not in w)]
    if pipeline == "video":
        # How long its video model makes a clip, for the form to say.
        paths = {w["name"]: w.get("path") for w in saved_workflows()}
        for w in found:
            w["clip_seconds"] = video_planner.clip_seconds(workflow_model(paths.get(w["name"])))
    return {"online": comfyui_online(), "workflows": found}


@app.route("/comfyui/start", methods=["POST"])
@password_required
def comfyui_start():
    """Only ComfyUI can list what's saved in it, and a run would only start it once one is under way."""
    try:
        start_comfyui()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}


@app.route("/videos/<folder>/<path:name>")
@password_required
def video_file(folder, name):
    return send_file(safe_video_path(folder, name), mimetype="video/mp4", conditional=True)


@app.route("/posters/<folder>/<path:name>")
@password_required
def poster(folder, name):
    """The picture a Short shows before it plays, as videos only load once you press play."""
    stem = os.path.splitext(safe_video_path(folder, name))[0]
    for candidate in (f"{stem}.jpg", f"{stem}.still.jpg"):
        if os.path.exists(candidate):
            return send_file(candidate, mimetype="image/jpeg", max_age=3600)
    abort(404)


@app.route("/download/<folder>/<path:name>")
@password_required
def download(folder, name):
    return send_file(safe_video_path(folder, name), as_attachment=True, download_name=name)


@app.route("/folders/<folder>/open", methods=["POST"])
@password_required
def open_folder(folder):
    """Shows a tab's folder in File Explorer. Made here if need be: nothing has been uploaded at first."""
    if not can_open_folders():
        abort(403)
    if folder not in FOLDERS:
        abort(404)
    try:
        os.makedirs(FOLDERS[folder], exist_ok=True)
        os.startfile(FOLDERS[folder])
    except OSError as e:
        return {"ok": False, "error": f"Could not open the folder: {e}"}, 500
    return {"ok": True}


@app.route("/delete/<folder>/<path:name>", methods=["POST"])
@password_required
def delete(folder, name):
    return PIPELINES[FOLDER_PIPELINES.get(folder, "shorts")].delete_video(folder, safe_video_path(folder, name))


@app.after_request
def never_cache_status(response):
    if request.path == "/api/status":
        response.headers["Cache-Control"] = "no-store"
    return response


REFUSED = "🚫"


@app.after_request
def log_refused_generate(response):
    """A Generate the panel turned down says why in the log, where it can still be read once the page has moved on."""
    if request.path.startswith("/generate/") and response.status_code >= 400 and response.is_json:
        error = (response.get_json(silent=True) or {}).get("error")
        if error:
            job.log(f"{REFUSED} Not added ({request.path.rsplit('/', 1)[-1]}): {error}")
    return response


class _QuietPolling(logging.Filter):
    """Drops the access-log line for the page's 2-second status poll, which would bury everything else."""

    def filter(self, record):
        return "/api/status" not in record.getMessage()


def connect_tailscale(port):
    url, problem = tailscale.connect(port)
    if url:
        print(f"🌍 Anywhere:   {url}  (Tailscale)")
    elif problem:
        print(f"⚠️ {problem} Your phone can only reach the panel at home until it connects.")


def main():
    logging.getLogger("werkzeug").addFilter(_QuietPolling())

    parser = argparse.ArgumentParser(description="The ComfyUI auto-gen panel.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1; 0.0.0.0 for your phone).")
    parser.add_argument("--port", type=int, default=8800, help="Port (default: 8800).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.host != "127.0.0.1" and not os.getenv("APP_PASSWORD"):
        parser.error(
            "Refusing to listen on a public address without APP_PASSWORD set. "
            "Anyone who finds the URL could spend your API quota and GPU. "
            "Set APP_PASSWORD in your .env and try again."
        )

    if not os.getenv("FLASK_SECRET_KEY"):
        print("⚠️ FLASK_SECRET_KEY not set — using a random key, so logins drop on restart.")

    global comfy_monitor
    comfy_monitor = ComfyMonitor()

    print(f"🌐 This PC:    http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"📱 Your phone: http://{lan_ip()}:{args.port}  (or http://{socket.gethostname()}.local:{args.port})")
        # In the background: it can take a while, and the panel shouldn't wait for it.
        threading.Thread(target=connect_tailscale, args=(args.port,), daemon=True).start()
    print("   Stop it with 'Shut down' in the panel (or close this window if it has one).")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

"""
Web UI for auto-comfyui: the control panel for the whole pipeline.

Starts films from an uploaded image and a short prompt, queues more while one is
going, lets you read and change the prompts before anything renders, streams the
log, shows whether ComfyUI is up, and lists the finished films for playback,
download and upload, from the desktop or a phone on the same network.

    python webapp.py --host 0.0.0.0 --port 8801
    (or double-click start_webapp.pyw / the "AutoComfyUI" desktop shortcut)

Set APP_PASSWORD to expose it beyond localhost. Without it the server refuses
any host but 127.0.0.1: an open instance lets anyone spend your LLM quota and GPU.
"""
import argparse
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import wraps

import requests
import websocket
from dotenv import load_dotenv
from flask import (Flask, abort, redirect, render_template, request,
                   send_file, session, url_for)

from modules import media, paths, planner
from modules.comfy import ComfyClient, client_id
from modules.run_eta import RunEta
from modules.run_state import STAGES, RunState, folder_size, list_runs, run_id_in
from modules.saved_workflows import list_saved

load_dotenv(os.path.join(paths.PROJECT_ROOT, ".env"))

PROJECT_ROOT = paths.PROJECT_ROOT
FOLDERS = {"output": paths.OUTPUT_DIR, "uploaded": paths.UPLOADED_DIR}
STAGE_TIMINGS = os.path.join(paths.LOGS_DIR, "stage_timings.json")
# Keeps the pipeline and nvidia-smi from flashing a console window when the
# server itself runs windowless (start_webapp.pyw).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# How long a killed process gets to disappear before it counts as surviving.
STOP_WAIT = 10
STOP_FAILED = "It could not be stopped. End it in Task Manager (python.exe)."
# start_webapp.pyw compares this with the code on disk to replace a server that
# predates an update (templates are cached and Python only loads at start).
STARTED_AT = time.time()
# A phone photo is well under this; the limit only stops a mistake filling the disk.
MAX_UPLOAD_BYTES = 50 * 2 ** 20
DEFAULT_FILM_SECONDS = 120

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


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

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    @property
    def idle(self):
        """Past running: _drain is done with the last process too, so a new one can't get mixed up with it."""
        return self.process is None or self.finished_at is not None

    def start(self, argv, env=None, keep_log=False):
        with self.lock:
            if self.closed or not self.idle:
                return False
            eta = RunEta(argv, STAGE_TIMINGS)
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
            self.finished_at = None
            self.returncode = None
            self.command = " ".join(argv[1:])
            self.progress = {}
            self.eta = eta
            self.stopped = False
            self.process = process

        threading.Thread(target=self._drain, daemon=True).start()
        return True

    def _drain(self):
        process = self.process
        for line in process.stdout:
            line = line.rstrip()
            # ffmpeg's progress carriage-returns would flood the log.
            if not line or line.startswith("frame=") or line.startswith("["):
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
    PATTERNS = [
        ("run_id", re.compile(r"Run folder: (run_\w+)")),
        ("stage", re.compile(r"▶ Stage (\d+)/(\d+): (.+)")),
        ("clip", re.compile(r"Clip (\d+)/(\d+) · (.+) · (\d+)s")),
        ("retry", re.compile(r"(?:Rejected: |❌ )(.*) \(attempt (\d+)/(\d+)\)")),
        ("part", re.compile(r"Preparing clip (\d+)/(\d+)")),
    ]

    def _track(self, line):
        p = self.progress
        for kind, pattern in self.PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            if kind == "run_id":
                p["run_id"] = m.group(1)
            elif kind == "stage":
                p["stage"] = [int(m.group(1)), int(m.group(2)), m.group(3).strip()]
                p["stage_started"] = time.time()
                p.pop("detail", None)
                self.eta.stage(*p["stage"])
            elif kind == "clip":
                p["detail"] = f"Clip {m.group(1)} of {m.group(2)}: {m.group(3)}, {m.group(4)}s"
                self.eta.clip(int(m.group(1)), int(m.group(2)), int(m.group(4)))
            elif kind == "retry":
                self.eta.clip_retry(int(m.group(2)), int(m.group(3)), m.group(1))
            elif kind == "part":
                p["detail"] = f"Preparing clip {m.group(1)} of {m.group(2)}"
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
        """In a run's own Upload step, when its film sits in output/ and isn't to be claimed."""
        return self.running and self.progress.get("stage", [None, None, None])[2] == "Upload"

    def snapshot(self):
        return {
            "running": self.running,
            "lines": list(self.lines),
            "returncode": self.returncode,
            "stopped": self.stopped,
            "command": self.command,
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
    queued with. A failed job doesn't stop the ones after it. Stop does: pressing it
    means nothing should start by itself, so what is queued waits for Resume. Saved
    to `path`, so what was waiting comes back after a Shut down or crash, on hold for
    the same reason.
    """

    KEPT_RESULTS = 8

    def __init__(self, job, prepare, blocked=lambda: False, path=None):
        self.job = job
        self.prepare = prepare  # Last step before a job starts; returns why it can't run, or None.
        self.blocked = blocked
        self.path = path
        self.lock = threading.Lock()
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
        with self.lock:
            self._ensure_watcher()
            item["id"] = self.next_id
            self.next_id += 1
            self.items.append(item)
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
            item = self.items.pop(0)
            self._save()
            self._start(item)

    def _finish_current(self):
        item, self.current = self.current, None
        code = self.job.returncode
        if self.job.stopped:
            self._record(item, "stopped")
        elif code == 0:
            self._record(item, "done")
        else:
            self._record(item, "failed", f"exit {code}")
            if self.items:
                self.job.log(f"❌ {item['label']} failed (exit {code}).")

    def _start(self, item):
        if item.get("waited"):
            self.job.log(f"===== Next in the queue: {item['label']} =====")
        try:
            error = self.prepare(item)
            # A job that waited adds to the log, so the one before it can still be read.
            if not error and self.job.start(item["argv"], item["env"], keep_log=item.get("waited", False)):
                self.current = item
                return
            error = error or "Another job took its place."  # A setup check, started in the moment between two jobs
        except Exception as e:  # One bad job must not take the watcher thread, and the queue, down with it
            error = str(e)
        item["error"] = error
        self.job.log(f"⚠️ Skipped {item['label']}: {error}")
        self._record(item, "skipped", error)

    def _record(self, item, outcome, note=None):
        self.results.append({"label": item["label"], "outcome": outcome, "note": note,
                             "at": datetime.now().strftime("%H:%M")})
        del self.results[:-self.KEPT_RESULTS]

    def hold(self):
        with self.lock:
            self.held = bool(self.items)

    def resume(self):
        with self.lock:
            self._ensure_watcher()
            self.held = False
            self._advance()

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

    def snapshot(self):
        with self.lock:
            self._advance()
            public = lambda item: {key: item.get(key) for key in ("id", "label", "detail", "estimate", "videos")}
            return {"items": [public(item) for item in self.items],
                    "current": public(self.current) if self.current else None,
                    "results": list(self.results),
                    "held": self.held}


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

    titles = {run["id"]: run["label"] for run in saved_runs()}
    videos = []
    for name in os.listdir(directory):
        if not name.endswith(".mp4"):
            continue
        stat = os.stat(os.path.join(directory, name))
        run = run_id_in(name)
        videos.append({
            "name": name,
            "run": run if run in titles else None,
            "title": titles.get(run),
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
# Seconds a run waits for a ComfyUI the panel just launched to load its models.
COMFYUI_STARTUP_WAIT = 180


def comfyui_url():
    return (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")


def comfyui_online():
    # Cached briefly: every open browser polls status every 2s.
    if time.time() - _comfyui_cache["at"] > 10:
        try:
            _comfyui_cache["online"] = requests.get(f"{comfyui_url()}/system_stats", timeout=2).status_code == 200
        except requests.RequestException:
            _comfyui_cache["online"] = False
        _comfyui_cache["at"] = time.time()
    return _comfyui_cache["online"]


BUILT_IN_TAKES = {"min_images": 1, "max_images": 1}
_saved_workflows_cache = {"at": 0.0, "online": None, "value": []}
_saved_workflows_lock = threading.Lock()


def saved_workflows():
    # Cached briefly: it asks ComfyUI about every node it has. Not across ComfyUI starting, which is when they appear.
    online = comfyui_online()
    with _saved_workflows_lock:
        cache = _saved_workflows_cache
        if time.time() - cache["at"] > 30 or cache["online"] != online:
            cache.update(at=time.time(), online=online, value=list_saved(comfyui_url(), paths.SAVED_WORKFLOWS_DIR))
        return cache["value"]


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


# What each node of the LTX workflow is doing, in plain words.
COMFY_LABELS = {
    "UNETLoader": "Loading the video model", "UnetLoaderGGUF": "Loading the video model",
    "DualCLIPLoader": "Loading the text encoder", "DualCLIPLoaderGGUF": "Loading the text encoder",
    "VAELoader": "Loading the VAE", "VAELoaderKJ": "Loading the audio VAE",
    "LatentUpscaleModelLoader": "Loading the upscaler",
    "LoadImage": "Loading the start frame",
    "CLIPTextEncode": "Reading the prompt",
    "LTXVImgToVideoInplace": "Placing the start frame", "LTXVPreprocess": "Preparing the image",
    "SamplerCustomAdvanced": "Generating video", "LTXVLatentUpsampler": "Upscaling",
    "VAEDecodeTiled": "Decoding frames", "LTXVAudioVAEDecode": "Decoding audio",
    "VHS_VideoCombine": "Saving the video",
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

    def _loop(self):
        while True:
            url = comfyui_url().replace("http", "ws", 1) + f"/ws?clientId={client_id()}"
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
            queue = requests.get(f"{comfyui_url()}/queue", timeout=5).json()
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
    Cancels the clips ComfyUI has for a stopped run, rendering or waiting. Only the
    pipeline's, by client id: a render you started in ComfyUI yourself carries on.
    ComfyUI itself keeps running, with its models loaded for the next run.
    """
    try:
        ComfyClient().cancel()
    except (RuntimeError, ValueError) as e:  # A workflow file that won't load
        job.log(f"⚠️ Could not cancel the ComfyUI clip, it may keep rendering: {e}")


def remove_partial_videos():
    """A stitch that gets killed leaves its unfinished file behind in the output folder."""
    try:
        for name in os.listdir(paths.OUTPUT_DIR):
            if name.endswith(".part"):
                os.remove(os.path.join(paths.OUTPUT_DIR, name))
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            job.log(f"⚠️ Could not remove the unfinished film: {e}")


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


def start_comfyui_for_run():
    """Starts ComfyUI when a launcher is configured. Returns True when it is (still) booting, so the run should wait for it."""
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


def item_images(item):
    return ([item["image"]] if item.get("image") else []) + item.get("references", [])


def prepare_run(item):
    """ComfyUI starts when the run does, not when it is queued: it may not be needed for hours."""
    if item.get("resume") and not RunState.load(item["resume"]):
        return "its saved run is gone"
    if not all(os.path.exists(path) for path in item_images(item)):
        return "its image is gone"
    if item.get("plan_only"):
        return None  # Writing the prompts needs no GPU
    try:
        if start_comfyui_for_run():
            item["env"]["COMFYUI_STARTUP_WAIT"] = str(COMFYUI_STARTUP_WAIT)
    except RuntimeError as e:
        return str(e)
    return None


def prepare_upload(item):
    """Drops films deleted or uploaded while this upload waited its turn."""
    gone = [name for name in item["videos"] if not os.path.exists(os.path.join(paths.OUTPUT_DIR, name))]
    if len(gone) == len(item["videos"]):
        return "no longer in Ready"
    item["argv"] = [arg for arg in item["argv"] if arg not in gone]
    item["videos"] = [name for name in item["videos"] if name not in gone]
    return None


run_queue = JobQueue(job, prepare_run, path=os.path.join(paths.LOGS_DIR, "queue_runs.json"))
# A run's own upload step is waited out, when its film sits in output/.
upload_queue = JobQueue(uploads, prepare_upload, blocked=lambda: job.uploading,
                        path=os.path.join(paths.LOGS_DIR, "queue_uploads.json"))
QUEUES = {"run": run_queue, "upload": upload_queue}


def work_left():
    return job.running or uploads.running or run_queue.pending() or upload_queue.pending()


def resume_argv(state, redo=None):
    argv = [sys.executable, "main.py", "--resume", state.id]
    return argv + ["--redo", str(redo)] if redo else argv


_runs_cache = {"at": 0.0, "value": []}


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
    try:
        parts.append(datetime.fromtimestamp(os.path.getmtime(state.path)).strftime("%d %b %H:%M"))
    except OSError:
        pass
    parts.append(f"{folder_size(state.folder) / 2 ** 20:.0f} MB")
    return {"id": state.id, "label": state.title or state.brief, "detail": " · ".join(parts),
            "clips": len(state.clips), "finished": bool(state.final),
            "estimate": RunEta(resume_argv(state), STAGE_TIMINGS).plan()["total"]}


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
            _runs_cache["value"] = [describe_run(state) for state in list_runs() if state.shots]
        except Exception as e:  # This is polled for the whole page: one odd run must not freeze it
            job.log(f"⚠️ Could not list the saved runs: {e}")
    return _runs_cache["value"]


def stopped_runs():
    """What can be continued. A finished run is reached from its film instead."""
    busy = busy_runs()
    return [run for run in saved_runs() if run["id"] not in busy and not run["finished"]]


def stop_run():
    """Kills the run with its ffmpeg renders and ComfyUI clips. What is queued waits for Resume."""
    # Before the kill, or the next job could start as soon as this one dies.
    if job.running:
        run_queue.hold()
    stopped = job.stop()
    if stopped:
        cancel_pipeline_clips()
        remove_partial_videos()
        _runs_cache["at"] = 0.0
    return stopped


def clean_inputs():
    """Images whose job has finished or gone: the run keeps its own copy, so nothing needs these any more."""
    needed = {path for item in run_queue.pending() for path in item_images(item)}
    try:
        for name in os.listdir(paths.INPUTS_DIR):
            path = os.path.join(paths.INPUTS_DIR, name)
            if path not in needed:
                os.remove(path)
    except OSError:
        pass  # Only costs a little disk until the next one


@app.errorhandler(413)
def too_large(error):
    return {"ok": False, "error": f"That image is over {MAX_UPLOAD_BYTES // 2 ** 20} MB."}, 413


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


def on_this_pc():
    return request.remote_addr in ("127.0.0.1", "::1")


def can_open_folders():
    """Explorer opens on the PC running the server, which is no use to someone on their phone."""
    return os.name == "nt" and on_this_pc()


@app.route("/")
@password_required
def index():
    port = request.host.rsplit(":", 1)[-1] if ":" in request.host else "80"
    return render_template(
        "index.html",
        protected=bool(os.getenv("APP_PASSWORD")),
        phone_url=f"http://{lan_ip()}:{port}",
        stages=STAGES,
        llms=[{"name": name, "label": label, "ready": planner.llm_configured(name)} for name, label in planner.LLMS],
        default_llm=planner.default_llm(),
        film_limits=[planner.MIN_FILM_SECONDS, planner.MAX_FILM_SECONDS, DEFAULT_FILM_SECONDS],
        can_upload=youtube_configured(),
        can_open_folders=can_open_folders(),
    )


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

    image_2_role = form.get("image_2_role") or planner.EXTRA_CONTEXT
    if image_2_role not in planner.IMAGE_2_ROLES:
        return None, "Unknown role for image 2."
    flags += ["--image-2-role", image_2_role]

    llm = form.get("llm") or planner.default_llm()
    if llm not in dict(planner.LLMS):
        return None, "Unknown LLM."
    flags += ["--llm", llm]

    size = (form.get("size") or "").strip().lower()
    if size:
        if not re.fullmatch(r"\d{2,4}x\d{2,4}", size):
            return None, "Render size must look like 1280x736."
        flags += ["--size", size]

    if form.get("upload"):
        privacy = form.get("privacy") or "private"
        if privacy not in ("private", "unlisted", "public"):
            return None, "Unknown privacy setting."
        flags += ["--upload", "--privacy", privacy]

    if form.get("review"):
        flags.append("--plan-only")
    return flags, None


@app.route("/api/estimate", methods=["POST"])
@password_required
def estimate():
    """What a run with the form's current options should take. Starts nothing."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}
    # Reviewing first only writes the prompts now, but what you're deciding on is the whole film.
    return {"ok": True, **RunEta([arg for arg in flags if arg != "--plan-only"], STAGE_TIMINGS).plan()}


def queue_label(text):
    text = " ".join(text.split())
    return text if len(text) <= 80 else text[:77] + "..."


def form_uploads(files, takes):
    """The images chosen in the form's boxes (image, image_2...), in order. Returns (uploads, error)."""
    field = lambda number: "image" if number == 1 else f"image_{number}"
    filled = {name for name, upload in files.items() if upload.filename}
    uploads = [files[field(number)] for number in range(1, len(filled) + 1) if field(number) in filled]
    if len(uploads) < len(filled):
        return None, "Fill the image boxes in order: each one adds to those before it."
    low, high = takes["min_images"], takes["max_images"]
    # Clips with room for one image still leave image 2 for the LLM.
    room = 2 if high == 1 else high
    if len(uploads) < low:
        return None, "Choose the image the film opens on." if low == 1 else f"This workflow needs {low} images."
    if len(uploads) > room:
        return None, "This workflow makes its clips from text alone: leave the image out." if not room else f"This workflow takes at most {room} images."
    return uploads, None


@app.route("/generate", methods=["POST"])
@password_required
def generate():
    """Starts a film, or queues it behind the one in progress."""
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    env, takes, error = chosen_workflow(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "Say what should happen in the film."}, 400
    llm = flags[flags.index("--llm") + 1]
    if not planner.llm_configured(llm):
        return {"ok": False, "error": f"{llm} has no credentials. See .env.example."}, 400

    uploads, error = form_uploads(request.files, takes)
    if error:
        return {"ok": False, "error": error}, 400
    try:
        images = [media.load_image(upload.stream) for upload in uploads]
    except RuntimeError:
        return {"ok": False, "error": "That file isn't an image this can read."}, 400

    # Kept in a folder of its own, under a name of our own: queued, it may wait for hours.
    clean_inputs()
    image_paths = [os.path.join(paths.INPUTS_DIR, f"{secrets.token_hex(8)}.png") for _ in images]
    for image, path in zip(images, image_paths):
        media.save_input(image, path)

    references = [flag for path in image_paths[1:] for flag in ("--reference", path)]
    argv = [sys.executable, "main.py", *flags, *references, *([] if images else ["--no-image"]), "--", *image_paths[:1], prompt]
    plan_only = "--plan-only" in flags
    seconds = int(flags[flags.index("--duration") + 1])
    length = f"{seconds}s" if seconds < 60 else f"{seconds // 60}:{seconds % 60:02d}"
    detail = [f"{length} film", llm, "review the prompts first" if plan_only else "renders straight away"]
    if env:
        detail.append(request.form["workflow"].removesuffix(".json"))
    item = {"label": queue_label(prompt), "detail": " · ".join(detail), "argv": argv, "env": env,
            "image": image_paths[0] if image_paths else None, "references": image_paths[1:], "plan_only": plan_only, "estimate": RunEta(argv, STAGE_TIMINGS).plan()["total"]}
    queued = run_queue.add(item)
    if item.get("error"):
        return {"ok": False, "error": item["error"]}, 500
    return {"ok": True, "queued": queued}


def queue_saved_run(state, redo=None):
    argv = resume_argv(state, redo)
    what = f"Redo from shot {redo}" if redo else "Continue"
    item = {"label": queue_label(state.title or state.brief), "detail": what, "argv": argv, "env": {},
            "resume": state.id, "estimate": RunEta(argv, STAGE_TIMINGS).plan()["total"]}
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
    prompts = (request.get_json(silent=True) or {}).get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(state.shots) or not all(isinstance(p, str) and p.strip() for p in prompts):
        return {"ok": False, "error": "Every shot needs a prompt."}, 400
    for shot, prompt in zip(state.shots, prompts):
        shot["prompt"] = prompt.strip()
    state.save()
    return {"ok": True}


@app.route("/runs/<run_id>/continue", methods=["POST"])
@password_required
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


@app.route("/runs/<run_id>/discard", methods=["POST"])
@password_required
def discard_run(run_id):
    """Deletes a run's plan and clips. A film already finished from it stays."""
    state, refused = load_idle_run(run_id)
    if refused:
        return refused
    state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


@app.route("/check", methods=["POST"])
@password_required
def check():
    """Runs the preflight with the same LLM as a real run would use."""
    if job.running:
        return {"ok": False, "error": "A run is already in progress."}, 409
    flags, error = build_options(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    env, _, error = chosen_workflow(request.form)
    if error:
        return {"ok": False, "error": error}, 400
    argv = [sys.executable, "main.py", "--check", "--llm", flags[flags.index("--llm") + 1]]
    if not job.start(argv, env):
        return {"ok": False, "error": "A run is already in progress."}, 409
    return {"ok": True}


@app.route("/stop", methods=["POST"])
@password_required
def stop():
    return {"ok": stop_run(), "error": STOP_FAILED if job.running else None}


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
    payload = job.snapshot()
    payload["queue"] = run_queue.snapshot()
    payload["stopped_runs"] = stopped_runs()
    payload["upload"] = uploads.snapshot()
    payload["upload"]["queue"] = upload_queue.snapshot()
    payload["videos"] = {folder: list_videos(folder) for folder in FOLDERS}
    online = comfyui_online()
    payload["comfyui"] = {"online": online, "can_start": bool(comfyui_launcher()),
                          "starting": not online and time.time() - _comfyui_start_requested_at < COMFYUI_STARTUP_WAIT,
                          **(comfy_monitor.snapshot() if comfy_monitor else {})}
    payload["gpu"] = gpu_stats()
    return payload


@app.route("/api/workflows")
@password_required
def workflows():
    found = [{key: w[key] for key in ("name", "label", "takes", "problem") if key in w} for w in saved_workflows()]
    return {"online": comfyui_online(), "workflows": found}


@app.route("/videos/<folder>/<path:name>")
@password_required
def video(folder, name):
    return send_file(safe_video_path(folder, name), mimetype="video/mp4", conditional=True)


@app.route("/download/<folder>/<path:name>")
@password_required
def download(folder, name):
    return send_file(safe_video_path(folder, name), as_attachment=True, download_name=name)


@app.route("/folders/<folder>/open", methods=["POST"])
@password_required
def open_folder(folder):
    """Shows a tab's folder in File Explorer. Made here if need be: uploaded/ doesn't exist until the first upload."""
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


def youtube_configured():
    return all(os.getenv(k) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"))


@app.route("/publish", methods=["POST"])
@password_required
def publish():
    """
    Uploads one film from output/ (form 'video'), or all of them, as a run, or
    queues that behind the upload in progress. All means the films there now: what
    the page asked about, not whatever has been made by the time it starts.
    """
    privacy = request.form.get("privacy") or "private"
    if privacy not in ("private", "unlisted", "public"):
        return {"ok": False, "error": "Unknown privacy setting."}, 400
    if not youtube_configured():
        return {"ok": False, "error": "Set the YouTube keys in .env first (run scripts/authorize_youtube.py once)."}, 400

    claimed = {video for other in upload_queue.pending() for video in other["videos"]}
    ready = list_videos("output")
    argv = [sys.executable, "publish.py", "--privacy", privacy]
    name = request.form.get("video")
    if name:
        name = os.path.basename(safe_video_path("output", name))
        if name in claimed:
            return {"ok": False, "error": "That film is already uploading or queued."}, 409
        videos = [name]
        label = next((v["title"] for v in ready if v["name"] == name and v["title"]), name)
    else:
        videos = sorted(v["name"] for v in ready if v["name"] not in claimed)
        if not videos:
            return {"ok": False, "error": "Every ready film is already uploading or queued."}, 409
        label = f"{len(videos)} {'film' if len(videos) == 1 else 'films'}"
    argv += videos

    queued = upload_queue.add({"label": label, "detail": privacy, "argv": argv, "env": None, "videos": videos})
    return {"ok": True, "queued": queued}


@app.route("/delete/<folder>/<path:name>", methods=["POST"])
@password_required
def delete(folder, name):
    """Deletes a film and the run that made it, whose clips would otherwise sit on the disk unseen."""
    path = safe_video_path(folder, name)
    run_id = run_id_in(os.path.basename(path))
    if run_id and run_id in busy_runs():
        return {"ok": False, "error": "That film's run is going or queued. Stop it or take it out of the queue first."}, 409
    os.remove(path)
    state = RunState.load(run_id)
    if state:
        state.remove()
    _runs_cache["at"] = 0.0
    return {"ok": True}


class _QuietPolling(logging.Filter):
    """Drops the access-log line for the page's 2-second status poll, which would bury everything else."""

    def filter(self, record):
        return "/api/status" not in record.getMessage()


def main():
    logging.getLogger("werkzeug").addFilter(_QuietPolling())

    parser = argparse.ArgumentParser(description="Web UI for auto-comfyui.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1; 0.0.0.0 for your phone).")
    parser.add_argument("--port", type=int, default=8801, help="Port (default: 8801).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.host != "127.0.0.1" and not os.getenv("APP_PASSWORD"):
        parser.error(
            "Refusing to listen on a public address without APP_PASSWORD set. "
            "Anyone who finds the URL could spend your LLM quota and GPU. "
            "Set APP_PASSWORD in your .env and try again."
        )

    if not os.getenv("FLASK_SECRET_KEY"):
        print("⚠️ FLASK_SECRET_KEY not set — using a random key, so logins drop on restart.")

    global comfy_monitor
    comfy_monitor = ComfyMonitor()

    print(f"🌐 This PC:    http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"📱 Your phone: http://{lan_ip()}:{args.port}  (or http://{socket.gethostname()}.local:{args.port})")
    print("   Stop it with 'Shut down' in the panel (or close this window if it has one).")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

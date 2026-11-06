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
import math
import os
import re
import shutil
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from urllib.parse import urlparse
from functools import wraps

import requests
import websocket
from dotenv import load_dotenv
from flask import (Flask, abort, has_request_context, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.exceptions import NotFound

from core import batch_stop, eta_keys, workflow_options
from core.paths import LOGS_DIR, PROJECT_ROOT, SAVED_WORKFLOWS_DIR, comfyui_client_id
from core.quota import quota_summary
from core.saved_workflows import display_label, list_saved, pipeline_of
from panel import settings, tailscale
from pipelines.image.paths import OUTPUT_DIR as IMAGE_OUTPUT
from pipelines.image.run_eta import RunEta as ImageEta
from pipelines.shorts.paths import OUTPUT_DIR as SHORTS_READY, UPLOADED_DIR as SHORTS_UPLOADED
from pipelines.shorts.run_eta import RunEta as ShortsEta
from pipelines.video.paths import OUTPUT_DIR as VIDEO_OUTPUT
from pipelines.video import media as video_media
from pipelines.video import planner as video_planner
from pipelines.video import comfy as video_comfy
from pipelines.video.comfy import workflow_model
from pipelines.video.run_state import UPSCALED, films_of_runs, run_of_film
from pipelines.video.run_eta import RunEta as VideoEta

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

FOLDERS = {"output": SHORTS_READY, "uploaded": SHORTS_UPLOADED, "video": VIDEO_OUTPUT, "image": IMAGE_OUTPUT}
FOLDER_PIPELINES = {"output": "shorts", "uploaded": "shorts", "video": "video", "image": "image"}
SHORTS_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_shorts.json")
VIDEO_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_video.json")
IMAGE_TIMINGS = os.path.join(LOGS_DIR, "stage_timings_image.json")
PROMPT_WRITER_TIMINGS = os.path.join(LOGS_DIR, "prompt_writer_timings.json")
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
# The Run generation at lower priority setting: what the pipeline, and ComfyUI when this launches it, start with.
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def priority_flags():
    return BELOW_NORMAL if settings.load()["low_priority"] else 0
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


# What a log line is, for the page to colour and filter by: a stage, a problem, a warning, a success, ComfyUI's own, or dim.
LINE_KINDS = (("stage", re.compile(r"^(▶ Stage|=====)")), ("err", re.compile(r"^(❌|🚨)|\bError\b|Traceback")),
              ("warn", re.compile(r"^(⚠️|🛑)")), ("ok", re.compile(r"^(✅|✨|📦)")),
              ("comfy", re.compile(r"^🎛️ ComfyUI")), ("dim", re.compile(r"^(Deleted:|🧹)")))


def line_kind(text):
    text = text.strip()
    return next((kind for kind, pattern in LINE_KINDS if pattern.search(text)), "")


class Job:
    """
    Tracks one process at a time. Rendering saturates the CPU and GPU, so a second
    concurrent run would only make both slower. Uploads hardly touch either, so
    they get a Job of their own and can run during a render.
    """

    def __init__(self, batches=False):
        self.lock = threading.Lock()
        # The runs' job, not the uploads': a Stop after this one it was asked for goes with the job it was asked of.
        self.batches = batches
        self.process = None
        # {"n": sequence number, "t": "HH:MM:SS", "text": ...}; the page appends by n.
        self.lines = []
        self.seq = 0
        # Lines come from the pipeline's output and from ComfyUI's websocket at once.
        self._log_lock = threading.Lock()
        # The workflow file and options this run was started with, kept after it ends so it can still be looked at.
        self.workflow = None
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
            if self.batches:
                batch_stop.clear()
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
                creationflags=NO_WINDOW | priority_flags(),
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
            self.workflow = {"path": (env or {}).get("COMFYUI_WORKFLOW"), "pipeline": pipeline,
                             "options": eta_keys.options_in(argv) if pipeline == "image" else eta_keys.options_in_env(env)}
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
        with self._log_lock:
            self.seq += 1
            self.lines.append({"n": self.seq, "t": datetime.now().strftime("%H:%M:%S"), "k": line_kind(text), "text": text})

    # Lines main.py and its modules print, turned into structured progress.
    PATTERNS = {
        "shorts": [
            ("run", re.compile(r"===== RUN (\d+)/(\d+) =====")),
            ("run_id", re.compile(r"Run folder: (run_\w+)")),
            ("stage", re.compile(r"\u25b6 Stage (\d+)/(\d+): (.+)")),
            ("scenes", re.compile(r"narration for (\d+) scenes|Generation for (\d+) scenes")),
            ("clip", re.compile(r"Clip (\d+)/(\d+) \u00b7 scene (\d+): (\d+)s ([\w-]+)")),
            ("rejected", re.compile(r"Rejected: (.*) \(attempt (\d+)/(\d+)\)")),
            ("clip_failed", re.compile(r"Scene .+ avatar clip not made; using stock footage\.")),
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
        if self.pipeline == "video" and line.startswith("✅ Upscaled: "):
            p["upscaled_file"] = os.path.basename(line.removeprefix("✅ Upscaled: ").strip())
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
            elif kind == "clip_failed":
                self.eta.clip_failed()
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


job = Job(batches=True)
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
        edit_id = request.form.get("queue_edit_id", type=int) if has_request_context() else None
        edit_token = request.form.get("queue_edit_token") if edit_id else None
        # Generate's "Start later": it waits in the queue until it's started.
        if has_request_context() and request.path.startswith("/generate/") and request.form.get("start_later"):
            item["later"] = True
        with self.lock:
            self._ensure_watcher()
            if edit_id is not None:
                old = next((entry for entry in self.items if entry["id"] == edit_id), None)
                if not old or not edit_token or old.get("edit_token") != edit_token or old.get("pipeline") != item.get("pipeline"):
                    item["error"] = "That queued job changed or is no longer being edited. Open its setup again."
                    return True
                self.items.remove(old)
                item["id"] = edit_id
                item["waited"] = True
                item.pop("later", None)
                self.items.insert(0, item)
                self._save()
                self._advance()
                return item in self.items
            if self.current and self.job.idle:
                self._finish_current()
            item["id"] = self.next_id
            self.next_id += 1
            # On hold and nothing running: this one starts now, and what was waiting stays on hold.
            if self.held and self.job.idle and not self.current and not item.get("later") and not self.blocked():
                self.held = bool(self.items)
                item["waited"] = False
                self._start(item)
                return False
            # Ahead of those waiting to be started, which stay at the end.
            at = len(self.items) if item.get("later") else next((i for i, waiting in enumerate(self.items) if waiting.get("later")), len(self.items))
            self.items.insert(at, item)
            if self.job is job and self.job.running and batch_stop.requested():
                self.held = True
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
        # Before the next job can start: what this one loaded would otherwise stay in memory under it.
        if self.job.batches:
            ahead = None if self.held else next((waiting for waiting in self.items if not waiting.get("later")), None)
            if frees_comfyui(item, ahead):
                # Not under the queue's lock: a busy ComfyUI can take seconds to answer, and the page's status waits on it.
                threading.Thread(target=free_comfyui, daemon=True).start()
        code = self.job.returncode
        if self.job.stopped:
            self._record(item, "stopped")
            note_outcome(self.job, item, None)
        elif code == 0:
            self._record(item, "done")
            note_outcome(self.job, item, NOT_RENDERED if item.get("plan_only") else FIRST_ONLY if item.get("first_only") else None)
        else:
            # In plain words, with what to do, when it's a failure we know; the run's own lines under it.
            plain, said = explain_failure(self.job, code)
            self._record(item, "failed", plain or f"exit {code}")
            note_outcome(self.job, item, {"outcome": "failed", "why": f"{plain}\n\nFrom its log:\n{said}" if plain else said})
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
        run_id = ((self.job.progress or {}).get("run_id") or item.get("resume")) if outcome != "skipped" and self is run_queue else None
        state = PIPELINES[item["pipeline"]].load_run(run_id) if run_id and item.get("pipeline") in PIPELINES else None
        film = os.path.basename(state.final) if state and getattr(state, "final", None) else None
        if not state or (item.get("pipeline") in ("image", "video") and not state.shots):
            run_id = None
        argv = item.get("argv") or []
        if outcome == "done" and "pipelines.video.upscale" in argv:
            film = (self.job.progress or {}).get("upscaled_file")
        position = None
        if outcome == "done" and "pipelines.image.upscale" in argv and "--image" in argv[:-1] and argv[argv.index("--image") + 1].isdigit():
            position = int(argv[argv.index("--image") + 1]) + 1
        self.results.append({"label": item["label"], "outcome": outcome, "note": note, "pipeline": item.get("pipeline"),
                             "run_id": run_id, "film": film, "position": position, "job_id": item.get("id"),
                             "at": datetime.now().strftime("%H:%M"), "t": time.time()})
        del self.results[:-self.KEPT_RESULTS]

    def dismiss_failure(self, job_id):
        """Forgets a failure that has no saved run to retry. Returns whether there was one."""
        with self.lock:
            kept = [result for result in self.results
                    if not (result["job_id"] == job_id and result["outcome"] == "failed" and not result["run_id"])]
            dismissed = len(kept) != len(self.results)
            self.results = kept
            return dismissed

    def hold(self):
        with self.lock:
            self.held = bool(self.items)

    def resume(self):
        with self.lock:
            self._ensure_watcher()
            self.held = False
        # Starting the next job can take seconds (ComfyUI checks): not while the page's request waits on it.
        threading.Thread(target=self._advance_locked, daemon=True).start()

    def _advance_locked(self):
        with self.lock:
            self._advance()

    def start(self, item_id):
        """Starts a pending job: at the end of the queue, or right away when nothing is queued ahead of it."""
        with self.lock:
            item = next((item for item in self.items if item["id"] == item_id), None)
            if not item:
                return False
            if item.get("edit_token"):
                return False
            item.pop("later", None)
            self.items.remove(item)
            ahead = any(not waiting.get("later") for waiting in self.items)
            # Ahead of those still pending, which stay at the end.
            at = next((i for i, waiting in enumerate(self.items) if waiting.get("later")), len(self.items))
            self.items.insert(at, item)
            # On hold with nothing ahead of it, this is what starts; the pending ones stay on hold.
            if self.held and not ahead and self.job.idle and not self.current and not self.blocked():
                self.held = len(self.items) > 1
                self.items.remove(item)
                item["waited"] = False
                self._save()
                self._start(item)
                return True
            self._advance()
            self._save()
            return True

    def pause(self, item_id):
        """Takes a waiting job out of the active queue: it goes to the end, waiting to be started."""
        with self.lock:
            item = next((item for item in self.items if item["id"] == item_id), None)
            if not item:
                return False
            item["later"] = True
            self.items.remove(item)
            self.items.append(item)
            self._advance()
            self._save()
            return True

    def run_now(self, item_id):
        """Runs one waiting job from a paused queue, leaving the rest paused."""
        with self.lock:
            if self.current and self.job.idle:
                self._finish_current()
            if not self.held or not self.job.idle or self.current or self.blocked():
                return "not_ready", None
            item = next((item for item in self.items if item["id"] == item_id), None)
            if not item:
                return "not_found", None
            if item.get("edit_token"):
                return "not_ready", None
            self.items.remove(item)
            item.pop("later", None)
            self.held = bool(self.items)
            self._save()
            self._start(item)
            if self.current is item:
                return "started", None
            return "failed", item.get("error", "Could not start the queued job.")

    def reorder(self, ids):
        """
        Puts the waiting jobs in the order of these ids, as dragged on the page. One added since the page was drawn
        isn't in it, and stays where it was among the rest. Returns whether the order changed.
        """
        with self.lock:
            by_id = {item["id"]: item for item in self.items}
            wanted = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
            slots = iter(wanted)
            placed = {id(item) for item in wanted}
            order = [next(slots) if id(item) in placed else item for item in self.items]
            changed = order != self.items
            self.items = order
            self._advance()
            self._save()
            return changed

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
            public = lambda item: {**{key: item.get(key) for key in ("id", "label", "detail", "estimate", "videos", "pipeline", "later", "resume")},
                                   "editing": bool(item.get("edit_token")),
                                   "continuation": bool(item.get("resume") and "--redo" not in (item.get("argv") or [])
                                                        and not upscale_queue_settings(item)),
                                   "facts": run_facts(item)}
            return {"items": [public(item) for item in self.items],
                    "current": public(self.current) if self.current else None,
                    "results": list(self.results),
                    "held": self.held}


def run_facts(item):
    """What a run's title opens on: the workflow it renders with and the size it makes, as its options say."""
    argv, env = item.get("argv") or [], item.get("env") or {}
    flag = lambda name: argv[argv.index(name) + 1] if name in argv[:-1] else None
    workflow = env.get("COMFYUI_WORKFLOW")
    size = item.get("resolution") or flag("--size") or env.get("AVATAR_SIZE")
    if not size and item.get("resume") and item.get("pipeline") in ("image", "video") and not upscale_queue_settings(item):
        state = PIPELINES[item["pipeline"]].load_run(item["resume"])
        if state and getattr(state, "width", None) and getattr(state, "height", None):
            size = f"{state.width}x{state.height}"
    if not size and upscale_queue_settings(item):
        size = f"{flag('--resolution')}p" if flag("--resolution") else f"{flag('--scale') or '2'}× upscale"
    if not size and flag("--aspect"):
        aspect, megapixels = flag("--aspect"), flag("--megapixels") or "1"
        size = f"{'Source aspect' if aspect == 'image' else aspect}" + (f" at {megapixels} MP" if megapixels != "image" else " · source size")
    if not size and item.get("pipeline", "shorts") == "shorts":
        size = (os.getenv("AVATAR_SIZE") or "736x1280").lower()
    return {"workflow": display_label(os.path.basename(workflow)) if workflow else "Built-in",
            "resolution": size or ("As the run was set up" if "--resume" in argv else "The workflow's own"),
            **({"prompt": item["prompt"]} if item.get("prompt") else {})}


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
    chosen = {key: value for key, value in chosen.items() if value != next(option["default"] for option in options if option["id"] == key).strip()}
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


NSFW_PATH = os.path.join(LOGS_DIR, "nsfw.json")
_nsfw_lock = threading.Lock()


def _read_nsfw():
    try:
        with open(NSFW_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"image": [], "video": [], "video_file": [], "video_sources": {}}
    if not isinstance(data, dict) or any(not isinstance(data.get(key), list) or
                                         not all(isinstance(value, str) for value in data[key])
                                         for key in ("image", "video", "video_file")):
        raise ValueError("Invalid NSFW tags file.")
    sources = data.setdefault("video_sources", {})
    if not isinstance(sources, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in sources.items()):
        raise ValueError("Invalid NSFW video sources.")
    return data


def _video_nsfw_key(name, films=None):
    run = run_of_film(name, films)
    return ("video", run) if run else ("video_file", name)


def _apply_nsfw(payload, tags, film_runs):
    image_tags, video_tags, video_files = (set(tags[key]) for key in ("image", "video", "video_file"))
    tagged_films = set()
    for film in payload["videos"]["video"]:
        kind, value = _video_nsfw_key(film["name"], film_runs)
        source = UPSCALED.sub(".mp4", film["name"])
        film["nsfw"] = (kind == "video" and value in video_tags) or source in video_files or tags["video_sources"].get(source) in video_tags
        if film["nsfw"]:
            tagged_films.add(film["name"])
    for image_set in payload["image_sets"]:
        image_set["nsfw"] = image_set["id"] in image_tags

    def visible_result(result):
        if result.get("pipeline") == "image":
            return result.get("run_id") not in image_tags
        if result.get("pipeline") == "video":
            return result.get("run_id") not in video_tags and result.get("film") not in tagged_films
        return True

    payload["finished_results"] = [result for result in payload["queue"]["results"] if visible_result(result)]
    payload["made_now"] = [result for result in payload["made_now"] if visible_result(result)]


@app.route("/api/output/nsfw", methods=["POST"])
@password_required
def tag_nsfw():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("kind") not in ("image", "video") or not isinstance(body.get("id"), str) or not isinstance(body.get("nsfw"), bool):
        return {"ok": False, "error": "Invalid output tag."}, 400
    kind, identifier = body["kind"], body["id"]
    if kind == "image":
        if not PIPELINES["image"].load_run(identifier):
            return {"ok": False, "error": "That image batch no longer exists."}, 404
        key, value = "image", identifier
    else:
        safe_video_path("video", identifier)
        key, value = _video_nsfw_key(identifier)
    try:
        with _nsfw_lock:
            tags = _read_nsfw()
            if kind == "video":
                source = UPSCALED.sub(".mp4", identifier)
                if key == "video_file":
                    value = source
                    if not body["nsfw"] and source in tags["video_sources"]:
                        key, value = "video", tags["video_sources"][source]
                if key == "video":
                    if body["nsfw"]:
                        tags["video_sources"][source] = value
                    else:
                        tags["video_sources"] = {name: run for name, run in tags["video_sources"].items() if run != value}
                    tags["video_file"] = [name for name in tags["video_file"] if name != source]
            values = set(tags[key])
            if body["nsfw"]:
                values.add(value)
            else:
                values.discard(value)
            tags[key] = sorted(values)
            os.makedirs(os.path.dirname(NSFW_PATH), exist_ok=True)
            temp = NSFW_PATH + ".tmp"
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(tags, f, ensure_ascii=False, indent=2)
            os.replace(temp, NSFW_PATH)
    except (OSError, ValueError) as e:
        app.logger.error("Could not save NSFW tags: %s", e)
        return {"ok": False, "error": "Could not save NSFW tags. Check the panel log."}, 500
    return {"ok": True}


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


def _check_comfyui():
    url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
    try:
        online = requests.get(f"{url}/system_stats", timeout=2).status_code == 200
    except requests.RequestException:
        online = False
    # Not answering, yet its port is open: running, but too busy to reply, as when it pages for want of memory.
    _comfyui_cache.update(online=online, listening=online or _port_open(url), at=time.time())


_comfyui_checking = threading.Lock()


def _check_comfyui_in_background():
    try:
        _check_comfyui()
    finally:
        _comfyui_checking.release()


def comfyui_online(wait=True):
    """
    Whether ComfyUI answers, checked at most every 10 seconds. wait=False is for the page's status: it answers with
    what was last found and checks again in the background, so a ComfyUI too busy to reply can't hold the page up.
    """
    if time.time() - _comfyui_cache["at"] > 10:
        if wait or not _comfyui_cache["at"]:
            _check_comfyui()
        elif _comfyui_checking.acquire(blocking=False):
            threading.Thread(target=_check_comfyui_in_background, daemon=True).start()
    return _comfyui_cache["online"]


def _port_open(url):
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


def comfyui_listening():
    comfyui_online(wait=False)
    return _comfyui_cache.get("listening", False)


def ram_stats():
    """This PC's memory, in GB, or None without psutil: what ComfyUI runs short of first."""
    try:
        import psutil
        memory = psutil.virtual_memory()
    except Exception:
        return None
    return {"free": round(memory.available / 2 ** 30, 1), "total": round(memory.total / 2 ** 30, 1)}


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

    def _note(self, text, kind="🎛️ ComfyUI · "):
        """A line in the run's log, from ComfyUI's side: what it did, and how long it took."""
        if job.running:
            job.log(f"{kind}{text}")

    def _close(self):
        """Says what the node that just ended took, when it was worth saying: a second or more, or sampling steps."""
        s = self.state
        started, steps = getattr(self, "node_at", None), getattr(self, "node_steps", None)
        self.node_at = self.node_steps = None
        if started is None or not s.get("label") or s.get("node") is None:
            return
        seconds = time.time() - started
        if seconds >= 1 or steps:
            self._note(f"{s['label']}{f' · {steps} steps' if steps else ''} · {int(seconds)}s")

    def _handle(self, msg):
        kind, data = msg.get("type"), msg.get("data") or {}
        with self.lock:
            s = self.state
            if kind == "status":
                s["queue"] = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
            elif kind == "execution_start":
                self.nodes = self._load_nodes(data.get("prompt_id"))
                self.samplers_seen = []
                self.node_at = self.node_steps = None
                s.update(active=True, started=time.time(), node=None, label="Starting", value=None, max=None)
                self._note(f"started a workflow of {len(self.nodes)} nodes" if self.nodes else "started a workflow")
            elif kind == "executing":
                node = data.get("node")
                self._close()
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
                self.node_at = time.time()
            elif kind == "progress":
                s.update(value=data.get("value"), max=data.get("max"))
                self.node_steps = data.get("max")
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                self._close()
                if kind == "execution_error":
                    where = data.get("node_type") or "a node"
                    self._note(f"failed in {where}: {str(data.get('exception_message') or 'no reason given').strip()[:300]}", "❌ ComfyUI ")
                elif kind == "execution_interrupted":
                    self._note("stopped", "⚠️ ComfyUI ")
                else:
                    total = int(time.time() - s["started"]) if s.get("started") else None
                    self._note(f"finished{f' in {total}s' if total is not None else ''}")
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


_gpu_checking = threading.Lock()


def gpu_stats():
    """
    Utilisation / VRAM / power from nvidia-smi, cached for a few seconds. Once it has a reading, a new one is taken in
    the background: under load nvidia-smi can take seconds, and the page's status shouldn't wait for it.
    """
    if time.time() - _gpu_cache["at"] < 3:
        return _gpu_cache["value"]
    if _gpu_cache["value"] is None:
        return _read_gpu()
    if _gpu_checking.acquire(blocking=False):
        def read():
            try:
                _read_gpu()
            finally:
                _gpu_checking.release()
        threading.Thread(target=read, daemon=True).start()
    return _gpu_cache["value"]


def _read_gpu():
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


def _comfyui_workflow(item):
    """The ComfyUI workflow a queued job runs, or None if it uses none: Shorts with stock footage only."""
    env = (item or {}).get("env") or {}
    if env.get("COMFYUI_WORKFLOW"):
        return env["COMFYUI_WORKFLOW"]
    return None if (item or {}).get("pipeline", "shorts") == "shorts" and env.get("AVATAR_SOURCE") != "comfyui" else "built-in"


def frees_comfyui(item, ahead):
    """
    Whether ComfyUI should let go of its models once this job ends: when it used ComfyUI, unless the job that starts
    next runs the same workflow, which would only load them again.
    """
    used = _comfyui_workflow(item)
    return bool(used) and used != _comfyui_workflow(ahead)


def free_comfyui():
    """
    Has ComfyUI unload every model and clear its node cache, as its own Free model and node cache does. It keeps
    what it loaded otherwise: after a morning of MiniMax H3 and Krea 2 runs it held over 57 GB on a 32 GB PC, and
    paged so hard it stopped answering mid-render. Idle, it does this at once; it cancels nothing.
    """
    url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
    try:
        requests.post(f"{url}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
        job.log("🧹 ComfyUI let go of its models, so the next job starts with the memory free.")
    except requests.RequestException:
        pass  # Not running, or too busy to answer: nothing more to do about it here


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
                    if "--disable-auto-launch" not in line:
                        line += " --disable-auto-launch"
                    reserve = settings.load()["reserve_ram_gb"]
                    if reserve and "--cache-ram" not in line:
                        line += f" --cache-ram {reserve:g}"
                    return line
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
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP | priority_flags(),
        startupinfo=startup,
    )


def comfyui_processes():
    """The ComfyUI processes running on this PC, started by this server or not."""
    try:
        import psutil
    except ImportError as e:
        raise RuntimeError("Process inspection requires psutil; install the panel's requirements.") from e
    found = []
    for process in psutil.process_iter(["name", "cmdline"]):
        if process.pid == os.getpid() or not (process.info.get("name") or "").lower().startswith("python"):
            continue
        args = process.info.get("cmdline")
        if args is None:
            raise RuntimeError(f"Cannot inspect Python process {process.pid}; ComfyUI may still be running.")
        for arg in args[1:]:
            path = arg.replace("\\", "/").lower()
            if path.endswith("/main.py") and "comfyui" in path:
                found.append(process)
                break
            if path == "main.py":
                try:
                    if "comfyui" in process.cwd().lower():
                        found.append(process)
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as e:
                    raise RuntimeError(f"Cannot inspect Python process {process.pid}; ComfyUI may still be running.") from e
                break
            if path.endswith(".py") or path in ("-m", "-c"):
                break
    return found


def start_fresh_comfyui():
    """Stops local ComfyUI processes before a manual start, without touching unrelated Python processes."""
    try:
        import psutil
    except ImportError as e:
        raise RuntimeError("Process inspection requires psutil; install the panel's requirements.") from e
    global _comfyui_start_requested_at
    with _comfyui_start_lock:
        launcher = comfyui_launcher()
        if not launcher:
            raise RuntimeError("COMFYUI_LAUNCHER isn't set in .env, so the panel can't start ComfyUI.")
        try:
            processes = comfyui_processes()
            for process in processes:
                kill_tree(process.pid)
            _, alive = psutil.wait_procs(processes, timeout=10)
        except (psutil.Error, OSError) as e:
            raise RuntimeError(f"Could not inspect or stop local ComfyUI processes: {e}") from e
        if alive:
            raise RuntimeError(f"Could not stop ComfyUI process(es): {', '.join(str(p.pid) for p in alive)}.")
        url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
        if _port_open(url):
            raise RuntimeError("The ComfyUI address is still in use after stopping local processes; no new instance was started.")
        _comfyui_start_requested_at = 0.0
        _comfyui_cache["at"] = 0.0
        try:
            launch_comfyui(launcher)
        except OSError as e:
            raise RuntimeError(f"Could not start ComfyUI: {e}") from e
        _comfyui_start_requested_at = time.time()
        _comfyui_cache["at"] = 0.0


def apply_comfyui_priority():
    """Sets ComfyUI running now to the priority the setting asks for. One it starts itself starts at it."""
    try:
        import psutil
        low = settings.load()["low_priority"]
        for process in comfyui_processes():
            try:
                if os.name == "nt":
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if low else psutil.NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(10 if low else 0)
            except psutil.Error:
                pass  # Gone already, or not ours to change
    except Exception as e:
        job.log(f"⚠️ Could not change ComfyUI's priority: {e}")


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


_writing_prompt = False
_panel_closing = False
_comfy_restarting = False
run_queue = JobQueue(job, prepare_run, blocked=lambda: _writing_prompt or _panel_closing, path=os.path.join(LOGS_DIR, "queue_runs.json"))
# A run's own upload step is waited out, when its video sits in Ready with no topic recorded yet.
upload_queue = JobQueue(uploads, prepare_upload, blocked=lambda: job.uploading,
                        path=os.path.join(LOGS_DIR, "queue_uploads.json"))
QUEUES = {"run": run_queue, "upload": upload_queue}


def work_left():
    return _writing_prompt or job.running or uploads.running or run_queue.pending() or upload_queue.pending()


def restart_blocked():
    """
    Something a restart would cut short: a run or upload going, or a queue about to start one. A queue on hold
    isn't: it's saved, and comes back on hold.
    """
    return _writing_prompt or job.running or uploads.running or any(
        not queue.held and any(not item.get("later") for item in queue.pending())
        for queue in (run_queue, upload_queue)
    )


_runs_cache = {"at": 0.0, "value": []}


def busy_runs():
    """The saved runs a job has in hand: the one going, and those queued to continue."""
    busy = {item.get("resume") for item in run_queue.pending()}
    if job.running:
        busy.add(job.progress.get("run_id"))
    return busy


def active_runs():
    """Runs going or in the active queue; pending continuations reserve their run but are not active."""
    busy = {item.get("resume") for item in run_queue.pending() if not item.get("later")}
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
              "why": "A preview run: it stopped after the first. Open Prompts to see it and change the rest, "
                     "or continue to make them."}
NOT_RENDERED = {"outcome": "not rendered",
                "why": "You asked to read the prompts first (Let me read the prompts before rendering), so it stopped once "
                       "they were written. Open Prompts to change them, then render it."}
# The lines that say what went wrong, as the pipelines print them.
TROUBLE = re.compile(r"❌|⚠️|Error|Exception|Traceback|failed|refused|can't|could not", re.I)
MOST_WHY_LINES = 4


# The failures that come up, in plain words with what to do: the first whose pattern is in the run's log.
FAILURES = (
    (r"out of memory|OutOfMemoryError|CUDA error: out of memory|not enough (GPU )?memory|allocation on device",
     "Out of GPU memory. Try a smaller size (1 MP), fewer references, or turn off SeedVR2 or the 2nd pass under Workflow options."),
    (r"Needs the node [^,]+, which isn't installed|missing_node_type|node type .* not found|does not exist in the node",
     "A node this workflow needs isn't installed in ComfyUI. Install it with ComfyUI Manager, or turn off the option that uses it."),
    (r"is not installed|Value not in list: (ckpt|unet|vae|lora|clip|model)_name|No such file or directory: .*models",
     "A model file this workflow needs isn't in ComfyUI's models folder. Download it there, then continue."),
    (r"ComfyUI isn't running|Connection refused|Max retries exceeded|Failed to establish a new connection|ConnectionError",
     "ComfyUI wasn't running, or stopped answering. Start it (Generate starts it when COMFYUI_LAUNCHER is set), then continue."),
    (r"ComfyUI rejected the workflow|prompt_outputs_failed_validation|Prompt outputs failed validation",
     "ComfyUI refused the workflow as sent: something it expects is missing or out of range. The lines below say what."),
    (r"\b429\b|RESOURCE_EXHAUSTED|exceeded your current quota|rate limit",
     "The LLM's quota is used up for now (Gemini's free tier allows 20 a day per model). Choose another available provider, or try again later."),
    (r"\b503\b|UNAVAILABLE|overloaded|high demand",
     "The LLM was too busy to answer. Continue to try again: what was done is kept."),
    (r"has no credentials|invalid api.?key|Unauthorized|\b401\b",
     "The LLM has no working API key. Set it in .env (see .env.example)."),
    (r"rejected on all \d+ tries|failed on all \d+ tries",
     "Every try was rejected by the check (a broken frame, or lip-sync that didn't match). Try another seed, or a shorter line."),
    (r"No space left on device|disk is full",
     "The disk is full. Delete old output, or clear finished runs, then continue."),
    (r"timed out|Timeout|took longer than",
     "It took longer than allowed (COMFYUI_TIMEOUT in .env). A bigger size or more steps take longer: continue, or raise the limit."),
)


def explain_failure(job, code):
    """
    (In plain words, or None) and the run's own lines that say what went wrong. The plain words are for the failures
    that come up again and again; anything else shows only its lines.
    """
    said = why_it_failed(job, code)
    # The run's last lines that say what went wrong: a warning it got past earlier isn't why it stopped.
    plain = next((say for pattern, say in FAILURES if re.search(pattern, said, re.I)), None)
    return plain, said


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
        for position, shot in enumerate(state.shots if state else [], 1):
            if not shot.get("file"):
                continue
            made_at = os.path.getmtime(state.image_path(shot))
            # Only what this job made: a redo, or a set continued, kept the rest from before.
            if made_at < started:
                continue
            made.append({"label": f"{state.title or state.brief} · {shot.get('title') or shot['number']}", "outcome": "done",
                         "pipeline": "image", "run_id": state.id, "position": position,
                         "at": datetime.fromtimestamp(made_at).strftime("%H:%M"), "t": made_at})
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
        settings=settings.load(),
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


# Held by Stop after this one, so taking it back lets the queue go on again; held before stays held.
_held_for_stop_after = False


@app.route("/stop-after", methods=["POST"])
@password_required
def stop_after():
    """
    Stop after this one: a batch ends once the video or image it's making now is done, and what is queued after it
    waits for Resume, as with Stop. Asked again, it takes that back. A film is one video, so it has nothing to stop.
    """
    global _held_for_stop_after
    if not job.running:
        return {"ok": False, "error": "Nothing is running."}, 409
    if job.pipeline == "video":
        return {"ok": False, "error": "A film is one video, so there is no batch to stop after it: use Stop or Skip."}, 409
    if batch_stop.requested():
        batch_stop.clear()
        job.log("▶️ The batch goes on: Stop after this one taken back.")
        if _held_for_stop_after:
            run_queue.resume()
        _held_for_stop_after = False
        return {"ok": True, "stop_after": False}
    _held_for_stop_after = not run_queue.held and bool(run_queue.items)
    batch_stop.request()
    run_queue.hold()
    job.log("⏸️ The batch stops once the one being made now is done.")
    return {"ok": True, "stop_after": True}


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


@app.route("/queue/<lane>/failed/<int:job_id>/dismiss", methods=["POST"])
@password_required
def queue_dismiss_failure(lane, job_id):
    if lane not in QUEUES:
        abort(404)
    return {"ok": QUEUES[lane].dismiss_failure(job_id)}


@app.route("/queue/<lane>/start/<int:item_id>", methods=["POST"])
@password_required
def queue_start(lane, item_id):
    if lane not in QUEUES:
        abort(404)
    return {"ok": QUEUES[lane].start(item_id)}


@app.route("/queue/<lane>/pause/<int:item_id>", methods=["POST"])
@password_required
def queue_pause(lane, item_id):
    if lane not in QUEUES:
        abort(404)
    return {"ok": QUEUES[lane].pause(item_id)}


def legacy_queue_form(item):
    """Reconstruct forms for generation jobs saved before form snapshots were kept."""
    argv, env = item.get("argv") or [], item.get("env") or {}
    pipeline = item.get("pipeline")
    if pipeline not in ("video", "image") or "--resume" in argv or not any(
        name in argv for name in ("pipelines.video.main", "pipelines.image.main")
    ):
        return None
    flag = lambda name, default="": argv[argv.index(name) + 1] if name in argv[:-1] else default
    workflow = next((w["name"] for w in saved_workflows() if env.get("COMFYUI_WORKFLOW") and w.get("path")
                     and os.path.normcase(os.path.abspath(w["path"])) == os.path.normcase(os.path.abspath(env["COMFYUI_WORKFLOW"]))), "")
    if env.get("COMFYUI_WORKFLOW") and not workflow:
        return None
    form = {"workflow": workflow, "prompt": item.get("prompt") or "", "seed": flag("--seed"),
            "prompt_mode": "direct" if "--direct" in argv else "llm", "llm": flag("--llm"),
            "size": flag("--size"), "review": "1" if "--plan-only" in argv or "--enhance-only" in argv else "",
            "try_first": "1" if "--first-only" in argv else ""}
    files = {}
    if pipeline == "video":
        form.update(duration=flag("--duration", "120"), clip_seconds=flag("--clip-seconds"), start=flag("--start", "chain"),
                    h3_enhance=flag("--h3-enhance", "off"), h3_creativity=flag("--h3-creativity", "balanced"),
                    audio_mode="file" if "--audio" in argv else "text" if "--narration" in argv else "none",
                    narration=flag("--narration"), voice=flag("--voice"))
        if "--fixed-clips" in argv:
            form["clip_fixed"] = "1"
        images = ([item["image"]] if item.get("image") else []) + [
            argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--reference"]
        for i, path in enumerate(images):
            files["image" if not i else f"image_{i + 1}"] = path
        image_roles = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--image-role"]
        form.update({f"image_{i + 2}_role": role for i, role in enumerate(image_roles)})
        if flag("--audio"):
            files["audio_file"] = flag("--audio")
        for kind in ("video", "audio"):
            for i, value in enumerate(argv[:-1]):
                if value == f"--ref-{kind}":
                    number = sum(name.startswith("ref_media") for name in files) + 1
                    name = "ref_media" if number == 1 else f"ref_media_{number}"
                    files[name] = argv[i + 1]
                    if i + 3 < len(argv) and argv[i + 2] == f"--ref-{kind}-role":
                        form[name + "_role"] = argv[i + 3]
        options = json.loads(env.get("COMFYUI_OPTIONS") or "{}")
    else:
        if "--areas" in argv:
            return None
        form.update(count=flag("--count", "1"), image_mode="compose" if "--compose" in argv else "expand" if "--expand" in argv else "edit",
                    aspect=flag("--aspect"), megapixels=flag("--megapixels"), detail=flag("--reference-detail"),
                    transparent="1" if "--transparent" in argv else "")
        for i, path in enumerate(argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--image"):
            files["image" if not i else f"image_{i + 1}"] = path
        options = dict(argv[i + 1].split("=", 1) for i, value in enumerate(argv[:-1]) if value == "--option" and "=" in argv[i + 1])
        if "--expand" in argv:
            for side, value in zip(("left", "top", "right", "bottom"), flag("--expand").split(",")):
                form[f"expand_{side}"] = value
            form["expand_units"] = "pixels"
    return form, files, options


def upscale_queue_settings(item):
    argv = item.get("argv") or []
    flag = lambda name, default="": argv[argv.index(name) + 1] if name in argv[:-1] else default
    if "pipelines.image.upscale" in argv and item.get("pipeline") == "image":
        return {"kind": "image_upscale", "method": flag("--method", "seedvr2"),
                "strength": flag("--strength", "1"), "resolution": flag("--resolution", "0")}
    if "pipelines.video.upscale" in argv and item.get("pipeline") == "video":
        return {"kind": "video_upscale", "scale": flag("--scale", "2"), "quality": flag("--quality")}
    return None


@app.route("/queue/run/edit/<int:item_id>", methods=["POST"])
@password_required
def queue_edit(item_id):
    with run_queue.lock:
        item = next((entry for entry in run_queue.items if entry["id"] == item_id), None)
        if not item:
            return {"ok": False, "error": "That job has already started or is no longer queued."}, 409
        upscale = upscale_queue_settings(item)
        if item.get("resume") and not upscale:
            return {"ok": False, "error": "Setup for a continued run cannot be edited here. The job was not changed."}, 409
        if upscale:
            item["later"] = True
            item["edit_token"] = secrets.token_urlsafe(24)
            run_queue._save()
            return {"ok": True, "id": item_id, "token": item["edit_token"], "pipeline": item.get("pipeline"),
                    "upscale": upscale}
        restored = (item["form"], item.get("form_files", {}), {}) if item.get("form") else legacy_queue_form(item)
        if not restored:
            return {"ok": False, "error": "Setup is unavailable for this job. It remains in the queue unchanged."}, 409
        form, files, options = restored
        missing = [name for name, path in files.items() if not os.path.isfile(path)]
        if missing:
            return {"ok": False, "error": "A queued input file is missing. The job was not changed."}, 409
        item["later"] = True
        item["edit_token"] = secrets.token_urlsafe(24)
        run_queue._save()
        return {"ok": True, "id": item_id, "token": item["edit_token"], "pipeline": item.get("pipeline"),
                "form": form, "files": list(files), "filenames": {name: os.path.basename(path) for name, path in files.items()}, "options": options}


@app.route("/queue/run/edit/<int:item_id>/upscale", methods=["POST"])
@password_required
def queue_edit_upscale(item_id):
    from panel import image as image_panel, video as video_panel
    from PIL import Image

    data = request.get_json(silent=True) or {}
    token = data.get("token")
    if not isinstance(token, str):
        return {"ok": False, "error": "That editing session expired. Open the queued job again."}, 409
    with run_queue.lock:
        old = next((entry for entry in run_queue.items if entry["id"] == item_id), None)
        if not old or not old.get("edit_token") or not secrets.compare_digest(old["edit_token"], token):
            return {"ok": False, "error": "That editing session expired. Open the queued job again."}, 409
        settings = upscale_queue_settings(old)
        if not settings:
            return {"ok": False, "error": "This job is not an upscale."}, 400
        argv = old["argv"]
        flag = lambda name: argv[argv.index(name) + 1] if name in argv[:-1] else None
        if settings["kind"] == "image_upscale":
            state = image_panel.RunState.load(old.get("resume"))
            position = flag("--image")
            if not state or not position or not position.isdigit() or not 1 <= int(position) <= len(state.shots or []):
                return {"ok": False, "error": "The original image is no longer available."}, 409
            shot = state.shots[int(position) - 1]
            if not shot.get("file"):
                return {"ok": False, "error": "The original image is no longer available."}, 409
            try:
                with Image.open(state.image_path(shot)) as source:
                    shorter = min(source.size)
            except OSError:
                return {"ok": False, "error": "The original image could not be read."}, 409
            method = data.get("method")
            try:
                strength = float(data.get("strength", 1))
                resolution = int(data.get("resolution", 0))
            except (TypeError, ValueError):
                return {"ok": False, "error": "Choose a valid strength and resolution."}, 400
            if method not in ("fast", "seedvr2") or not math.isfinite(strength) or not 0 <= strength <= 1:
                return {"ok": False, "error": "Choose a valid upscale method and strength."}, 400
            if shorter > image_panel.UPSCALE_MOST_SIDE or (method == "seedvr2" and resolution and
                    (resolution not in image_panel.UPSCALE_RESOLUTIONS or resolution <= shorter)):
                return {"ok": False, "error": "Choose a supported resolution larger than the original image."}, 400
            if method == "fast":
                resolution = 0
            how = (f"SeedVR2, {resolution}p" if resolution else "SeedVR2, twice the size") if method == "seedvr2" else f"Superscale-SP, twice the size, strength {round(strength * 100)}%"
            new_argv = [*image_panel.UPSCALE, "--resume", state.id, "--image", position, "--method", method,
                        "--strength", f"{strength:g}", *(["--resolution", str(resolution)] if resolution else [])]
            estimate = image_panel.UPSCALE_SECONDS if method == "seedvr2" else image_panel.FAST_UPSCALE_SECONDS
        else:
            name = flag("--film")
            if not name:
                return {"ok": False, "error": "The original film is no longer available."}, 409
            try:
                video_panel.safe_video_path("video", name)
            except NotFound:
                return {"ok": False, "error": "The original film is no longer available."}, 409
            try:
                scale = float(data.get("scale"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "Choose a valid scale."}, 400
            quality = data.get("quality")
            if not math.isfinite(scale) or not 1 <= scale <= 4 or quality not in video_panel.film_upscale.QUALITIES:
                return {"ok": False, "error": "Choose a valid scale and quality."}, 400
            how = f"RTX VSR, {scale:g}x, {quality}"
            new_argv = [*video_panel.UPSCALE, "--film", name, "--scale", f"{scale:g}", "--quality", quality]
            estimate = old.get("estimate")
        replacement = {**old, "argv": new_argv, "detail": how, "estimate": estimate, "waited": True}
        replacement.pop("later", None)
        replacement.pop("edit_token", None)
        run_queue.items.remove(old)
        run_queue.items.insert(0, replacement)
        run_queue._save()
        run_queue._advance()
        return {"ok": True, "queued": replacement in run_queue.items}


def queue_form_snapshot():
    return {name: values for name, values in request.form.to_dict(flat=False).items()
            if name not in ("queue_edit_id", "queue_edit_token", "start_later")}


@app.before_request
def validate_queue_edit():
    if not request.path.startswith("/generate/") or not request.form.get("queue_edit_id"):
        return None
    item_id = request.form.get("queue_edit_id", type=int)
    token = request.form.get("queue_edit_token", "")
    with run_queue.lock:
        item = next((entry for entry in run_queue.items if entry["id"] == item_id), None)
        if not item or item.get("pipeline") != request.path.rsplit("/", 1)[-1] or not item.get("edit_token") or not secrets.compare_digest(item["edit_token"], token):
            return {"ok": False, "error": "That editing session expired. Open the queued job again."}, 409
    return None


@app.route("/queue/run/edit/<int:item_id>/file/<name>")
@password_required
def queue_edit_file(item_id, name):
    with run_queue.lock:
        item = next((entry for entry in run_queue.items if entry["id"] == item_id), None)
        if not item or not item.get("edit_token") or not secrets.compare_digest(item["edit_token"], request.headers.get("X-Queue-Edit-Token", "")):
            abort(404)
        restored = (item["form"], item.get("form_files", {}), {}) if item.get("form") else legacy_queue_form(item)
        path = restored[1].get(name) if restored else None
        if not path or not os.path.isfile(path):
            abort(404)
        return send_file(path, download_name=os.path.basename(path))


@app.route("/queue/run/edit/<int:item_id>/cancel", methods=["POST"])
@password_required
def queue_edit_cancel(item_id):
    token = (request.get_json(silent=True) or {}).get("token", "")
    with run_queue.lock:
        item = next((entry for entry in run_queue.items if entry["id"] == item_id), None)
        if not item or not item.get("edit_token") or not secrets.compare_digest(item["edit_token"], token):
            return {"ok": False, "error": "That editing session is no longer active."}, 409
        item.pop("edit_token")
        run_queue._save()
        return {"ok": True}


@app.route("/queue/<lane>/run-now/<int:item_id>", methods=["POST"])
@password_required
def queue_run_now(lane, item_id):
    if lane not in QUEUES:
        abort(404)
    if lane != "run":
        abort(404)
    result, error = QUEUES[lane].run_now(item_id)
    if result == "not_ready":
        return {"ok": False, "error": "Run this is available only when the generation queue is paused and idle."}, 409
    if result == "not_found":
        return {"ok": False, "error": "That job is no longer in the queue."}, 404
    if result == "failed":
        return {"ok": False, "error": error}, 500
    return {"ok": True}


@app.route("/queue/<lane>/order", methods=["POST"])
@password_required
def queue_order(lane):
    """The waiting jobs in a new order: {"ids": [...]}, first to start first."""
    if lane not in QUEUES:
        abort(404)
    ids = (request.get_json(silent=True) or {}).get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        return {"ok": False, "error": "The order is a list of the queued jobs' ids."}, 400
    return {"ok": True, "changed": QUEUES[lane].reorder(ids)}


@app.route("/queue/<lane>/resume", methods=["POST"])
@password_required
def queue_resume(lane):
    if lane not in QUEUES:
        abort(404)
    QUEUES[lane].resume()
    return {"ok": True}


@app.route("/queue/<lane>/hold", methods=["POST"])
@password_required
def queue_hold(lane):
    if lane not in QUEUES:
        abort(404)
    QUEUES[lane].hold()
    return {"ok": True}


@app.route("/api/settings", methods=["GET", "POST"])
@password_required
def api_settings():
    if request.method == "POST":
        changes = request.get_json(silent=True)
        if not isinstance(changes, dict):
            return {"ok": False, "error": "Settings are a JSON object."}, 400
        try:
            saved = settings.update(changes)
        except ValueError as e:
            return {"ok": False, "error": str(e)}, 400
        except OSError as e:
            return {"ok": False, "error": f"Could not save it: {e}"}, 500
        if "low_priority" in changes:
            threading.Thread(target=apply_comfyui_priority, daemon=True).start()
        return {"ok": True, "settings": saved}
    return {"ok": True, "settings": settings.load(), "disk": disk_usage()}


@app.route("/api/write-prompt", methods=["GET", "POST"])
@password_required
def write_prompt():
    from core import h3_refiner

    if request.method == "GET":
        side = request.args.get("pipeline")
        if side not in ("image", "video"):
            return {"ok": False, "error": "Choose an image or video prompt."}, 400
        try:
            model = h3_refiner.selected_model()
        except ValueError as e:
            return {"ok": False, "error": str(e)}, 500
        timings = eta_keys.Timings(PROMPT_WRITER_TIMINGS, {})
        estimate = timings.step(f"{model} | {side}")
        if estimate is not None and (not math.isfinite(estimate) or estimate <= 0):
            app.logger.warning("Invalid prompt writer timing for %s / %s: %s", model, side, estimate)
            estimate = None
        return {"ok": True, "model": model, "seconds": estimate if estimate is not None else
                (300 if model == "qwen3-14b" else 180)}
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("pipeline") not in ("image", "video"):
        return {"ok": False, "error": "Choose an image or video prompt."}, 400
    text = data.get("prompt")
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        return {"ok": False, "error": "Enter a prompt of up to 16,000 characters."}, 400
    global _writing_prompt
    with run_queue.lock:
        if _writing_prompt or _panel_closing or _comfy_restarting or job.running or (run_queue.pending() and not run_queue.held):
            return {"ok": False, "error": "Wait for the current generation or prompt writer to finish."}, 409
        _writing_prompt = True
    try:
        url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
        started = time.monotonic()
        refiner = h3_refiner.Refiner(url, require_compiler=False)
        if data.get("model") is not None and data["model"] != refiner.model_id:
            return {"ok": False, "error": "The selected model changed. Try writing the prompt again."}, 409
        with refiner:
            prompt = refiner.write_prompt(text.strip(), data["pipeline"])
        timings = eta_keys.Timings(PROMPT_WRITER_TIMINGS, {})
        key = f"{refiner.model_id} | {data['pipeline']}"
        old = timings.step(key)
        if old is not None and (not math.isfinite(old) or old <= 0):
            app.logger.warning("Discarding invalid prompt writer timing for %s: %s", key, old)
            timings.timed_steps.discard(key)
        timings.learn_step(key, time.monotonic() - started)
        try:
            os.makedirs(os.path.dirname(PROMPT_WRITER_TIMINGS), exist_ok=True)
            temporary = PROMPT_WRITER_TIMINGS + ".tmp"
            with open(temporary, "w", encoding="utf-8") as file:
                json.dump({"steps": {key: timings.steps[key] for key in sorted(timings.timed_steps)}, "renders": timings.renders}, file, indent=2)
            os.replace(temporary, PROMPT_WRITER_TIMINGS)
        except OSError as e:
            app.logger.warning("Could not save local prompt writer timings: %s", e)
        return {"ok": True, "prompt": prompt}
    except (RuntimeError, OSError, ValueError) as e:
        return {"ok": False, "error": str(e)}, 500
    finally:
        with run_queue.lock:
            _writing_prompt = False
        run_queue._advance_locked()


_disk_cache = {"at": 0.0, "value": None}


def disk_usage():
    """The disk the videos and images go to: what's free, and what the three output folders take."""
    if _disk_cache["value"] and time.time() - _disk_cache["at"] < 30:
        return _disk_cache["value"]
    folder = FOLDERS["output"]
    while folder and not os.path.exists(folder) and os.path.dirname(folder) != folder:
        folder = os.path.dirname(folder)
    try:
        usage = shutil.disk_usage(folder)
    except OSError:
        return None
    used = 0
    for root in {os.path.abspath(path) for path in FOLDERS.values()}:
        for where, _, names in os.walk(root):
            for name in names:
                try:
                    used += os.path.getsize(os.path.join(where, name))
                except OSError:
                    pass
    value = {"free": usage.free, "total": usage.total, "outputs": used,
             "drive": os.path.splitdrive(os.path.abspath(folder))[0] or "/"}
    _disk_cache.update(at=time.time(), value=value)
    return value


POWEROFF_IDLE = 15   # Seconds with nothing to do before the PC is told to shut down: a gap between two jobs isn't the end.
POWEROFF_POLL = 3
POWEROFF_GRACE = 60  # Seconds Windows waits, in which turning the option off (or new work) cancels it.
_poweroff = {"armed": False, "at": None, "idle_since": None}
_poweroff_lock = threading.Lock()


def _shutdown_command(cancel=False):
    if os.name == "nt":
        return ["shutdown", "/a"] if cancel else ["shutdown", "/s", "/t", str(POWEROFF_GRACE)]
    return ["shutdown", "-c"] if cancel else ["shutdown", "-h", f"+{max(1, POWEROFF_GRACE // 60)}"]


def _run_shutdown(cancel=False):
    try:
        subprocess.run(_shutdown_command(cancel), creationflags=NO_WINDOW, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass


def poweroff_state():
    with _poweroff_lock:
        left = max(0, round(_poweroff["at"] - time.time())) if _poweroff["at"] else None
        return {"armed": _poweroff["armed"], "seconds_left": left}


def _poweroff_loop():
    """While armed: once everything has been done for a moment, shuts the PC down. Not saved: a new server starts unarmed."""
    while True:
        time.sleep(POWEROFF_POLL)
        with _poweroff_lock:
            if not _poweroff["armed"]:
                _poweroff["watching"] = False
                return
            busy = bool(work_left())
            if busy:
                _poweroff["idle_since"] = None
                if _poweroff["at"]:
                    _poweroff["at"] = None
                    _run_shutdown(cancel=True)
            elif _poweroff["at"] is None:
                _poweroff["idle_since"] = _poweroff["idle_since"] or time.time()
                if time.time() - _poweroff["idle_since"] >= POWEROFF_IDLE:
                    _poweroff["at"] = time.time() + POWEROFF_GRACE
                    _run_shutdown()


@app.route("/api/poweroff", methods=["GET", "POST"])
@password_required
def api_poweroff():
    """Shut the PC down once the queue is empty: for this session only, so it is never saved."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        arm = bool(body.get("armed"))
        with _poweroff_lock:
            was_counting = _poweroff["at"] is not None
            _poweroff.update(armed=arm, idle_since=None, at=None if not arm else _poweroff["at"])
            if arm and not _poweroff.get("watching"):
                _poweroff["watching"] = True
                threading.Thread(target=_poweroff_loop, daemon=True).start()
            if not arm:
                _poweroff["at"] = None
        if not arm and was_counting:
            _run_shutdown(cancel=True)
    return {"ok": True, **poweroff_state()}


def respawn_command():
    """What starts this server again as it was started: the helper waits for the port to be let go first."""
    code = ("import json,subprocess,sys,time;time.sleep(2);a=json.loads(sys.argv[1]);"
            "subprocess.Popen(a['argv'],cwd=a['cwd'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
            "stderr=subprocess.DEVNULL,creationflags=a['flags'])")
    flags = NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return [sys.executable, "-c", code, json.dumps({"argv": [sys.executable, *sys.argv], "cwd": PROJECT_ROOT, "flags": flags})], flags


@app.route("/restart/panel", methods=["POST"])
@password_required
def restart_panel():
    """Starts the panel again, to load new code. Only when nothing is running that it would cut short."""
    global _panel_closing
    with run_queue.lock:
        if restart_blocked():
            return {"ok": False, "error": "A run, upload or prompt writer is going, or a queue is about to start one. Wait for it first."}, 409
        command, flags = respawn_command()
        subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags)
        _panel_closing = True
        threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"ok": True}


def restart_comfyui_now():
    for process in comfyui_processes():
        kill_tree(process.pid)
    close_comfyui()
    for _ in range(20):
        _comfyui_cache["at"] = 0.0
        if not comfyui_online():
            break
        time.sleep(0.5)
    global _comfyui_start_requested_at
    _comfyui_start_requested_at = 0.0
    try:
        start_comfyui()
    except RuntimeError as e:
        job.log(f"⚠️ {e}")


@app.route("/restart/comfyui", methods=["POST"])
@password_required
def restart_comfyui():
    global _comfy_restarting
    with run_queue.lock:
        if job.running or _writing_prompt or _comfy_restarting:
            return {"ok": False, "error": "A run or prompt writer is going, or ComfyUI is restarting. Wait for it first."}, 409
        if not comfyui_launcher():
            return {"ok": False, "error": "COMFYUI_LAUNCHER isn't set in .env, so the panel can't start ComfyUI again."}, 400
        _comfy_restarting = True
        def restart():
            global _comfy_restarting
            try:
                restart_comfyui_now()
            finally:
                with run_queue.lock:
                    _comfy_restarting = False
        threading.Thread(target=restart, daemon=True).start()
    return {"ok": True}


@app.route("/shutdown", methods=["POST"])
@password_required
def shutdown():
    """
    Force-stops everything the panel is doing, then the server itself: started
    from start_webapp.pyw it has no window to close.
    """
    global _panel_closing
    with run_queue.lock:
        if _writing_prompt:
            return {"ok": False, "error": "Wait for the prompt writer to finish before shutting down."}, 409
        for queue in QUEUES.values():
            queue.close()
        _panel_closing = True
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
    start a fresh one. Only answers this PC, and never stops a run or a queue that's going.
    """
    if not on_this_pc():
        abort(403)
    try:
        code_changed_at = float(request.form.get("code_changed_at", 0))
    except ValueError:
        abort(400)
    global _panel_closing
    with run_queue.lock:
        if code_changed_at <= STARTED_AT or restart_blocked():
            return {"restarting": False}
        _panel_closing = True
        threading.Timer(0.5, os._exit, args=(0,)).start()
    return {"restarting": True}


@app.route("/api/status")
@password_required
def status():
    payload, queue = run_queue.snapshot_with_job()
    run_id = (payload.get("progress") or {}).get("run_id")
    if queue.get("current") and run_id and payload.get("pipeline") in ("image", "video"):
        state = PIPELINES[payload["pipeline"]].load_run(run_id)
        if state and getattr(state, "width", None) and getattr(state, "height", None):
            queue["current"]["facts"]["resolution"] = f"{state.width}x{state.height}"
    payload["server_started"] = STARTED_AT
    payload["stop_after"] = payload["running"] and batch_stop.requested()
    payload["queue"] = queue
    payload["stopped_runs"] = stopped_runs()
    payload["made_now"] = made_in_this_run()
    payload["upload"] = uploads.snapshot()
    payload["upload"]["queue"] = upload_queue.snapshot()
    payload["videos"] = {folder: list_videos(folder) for folder in FOLDERS}
    payload["image_sets"] = [dict(run) for run in saved_runs() if run["pipeline"] == "image" and run["images"]]
    with _nsfw_lock:
        tags = _read_nsfw()
    _apply_nsfw(payload, tags, films_of_runs())
    online = comfyui_online(wait=False)
    payload["comfyui"] = {"online": online, "listening": comfyui_listening(), "can_start": bool(comfyui_launcher()), "ram": ram_stats(),
                          "starting": not online and time.time() - _comfyui_start_requested_at < COMFYUI_STARTUP_WAIT,
                          **(comfy_monitor.snapshot() if comfy_monitor else {})}
    payload["gpu"] = gpu_stats()
    payload["poweroff"] = poweroff_state()
    payload["quota"] = cached_quota()
    return payload


def _shown(value):
    """An input as the workflow viewer says it: a link to another node's output, or the value itself, cut if long."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
        return f"→ node {value[0]}" + (f" (output {value[1]})" if value[1] else "")
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 400 else text[:400] + "…"


@app.route("/api/run-workflow")
@password_required
def run_workflow():
    """
    The workflow the current or last run renders with, as it is sent to ComfyUI with that run's options: which options
    were set, and every node with what it is given. For the page's Workflow view.
    """
    started = job.workflow
    if not started:
        return {"error": "Nothing has run yet."}, 404
    path, chosen, pipeline = started.get("path"), started.get("options") or {}, started.get("pipeline")
    if not path and pipeline == "video":
        path = video_comfy.DEFAULT_WORKFLOW
    if not path:
        return {"error": "This run uses the built-in workflow, which isn't a saved one."}, 404
    api = None
    if chosen:
        try:
            api = workflow_options.remake(path, chosen, pipeline)[0]
        except RuntimeError:
            api = None
    if api is None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                api = json.load(f)
        except (OSError, ValueError):
            return {"error": "That workflow's file is gone."}, 404
    try:
        with open(os.path.splitext(path)[0] + ".map.json", "r", encoding="utf-8") as f:
            offered = json.load(f).get("_options") or []
    except (OSError, ValueError):
        offered = []
    options = [{"label": option["label"], "value": chosen.get(option["id"], option["default"]), "changed": option["id"] in chosen}
               for option in offered if not option.get("role")]
    nodes = [{"id": node_id, "type": node.get("class_type", ""), "inputs": {name: _shown(value) for name, value in node.get("inputs", {}).items()}}
             for node_id, node in sorted(api.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else 0, item[0]))]
    return {"name": display_label(os.path.basename(path)), "pipeline": pipeline, "options": options, "nodes": nodes,
            "running": job.running, "api": api}


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


@app.route("/api/page-error", methods=["POST"])
@password_required
def page_error():
    """A part of the page that threw while updating, from a phone's browser: into the panel's own log, to be found."""
    error = str((request.get_json(silent=True) or {}).get("error") or "")[:1500]
    logging.getLogger("werkzeug").warning("Page error from %s: %s", request.remote_addr, error)
    return {"ok": True}


@app.route("/api/presets", methods=["GET", "POST"])
@password_required
def presets():
    """A side's settings under a name: kept here, so every device has the same ones. POST saves one, deletes it, or makes it the default."""
    if request.method == "GET":
        return {"ok": True, "presets": load_presets()}
    body = request.get_json(silent=True) or {}
    side, name = body.get("side"), str(body.get("name") or "").strip()[:MOST_PRESET_NAME]
    if side not in PRESET_SIDES or not name:
        return {"ok": False, "error": "A preset needs a side and a name."}, 400
    saved = load_presets()
    if "default" in body and "values" not in body and not body.get("delete"):
        # The side's default, set as the page opens: one at most, so making one the default unmakes the last.
        if not any(p["name"] == name for p in saved[side]):
            return {"ok": False, "error": "There is no preset of that name."}, 404
        for p in saved[side]:
            p.pop("default", None)
            if body["default"] and p["name"] == name:
                p["default"] = True
    else:
        old = next((p for p in saved[side] if p["name"] == name), {})
        kept = [p for p in saved[side] if p["name"] != name]
        if not body.get("delete"):
            values = body.get("values")
            if not isinstance(values, dict):
                return {"ok": False, "error": "A preset needs its settings."}, 400
            # Saved over, it stays the default if it was one.
            kept.append({"name": name, "values": {str(k): v for k, v in values.items() if isinstance(v, (str, bool, int, float))},
                         **({"default": True} if old.get("default") else {})})
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
    # The image_, video_ or shorts_ in front only sorts them in ComfyUI: here the side already says it.
    for w in found:
        w["label"] = display_label(w["label"])
    if pipeline == "video":
        # How long its video model makes a clip, for the form to say.
        paths = {w["name"]: w.get("path") for w in saved_workflows()}
        for w in found:
            model = workflow_model(paths.get(w["name"]))
            w["clip_seconds"] = video_planner.clip_seconds(model)
            from core import h3_refiner
            w["openh3"] = h3_refiner.supports_path(paths.get(w["name"]))
            if video_media.most_pixels(model):
                w["most_pixels"] = video_media.most_pixels(model)
    elif pipeline == "shorts":
        paths = {w["name"]: w.get("path") for w in saved_workflows()}
        for w in found:
            w["openh3"] = workflow_model(paths.get(w["name"])) == "minimax_h3"
    return {"online": comfyui_online(), "workflows": found}


@app.route("/comfyui/start", methods=["POST"])
@password_required
def comfyui_start():
    """A manual start replaces any local ComfyUI instance; automatic starts don't."""
    with run_queue.lock:
        if restart_blocked():
            return {"ok": False, "error": "A run or upload is going, or a queue is about to start one. Stop it or let it finish first."}, 409
        try:
            start_fresh_comfyui()
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}, 500
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
    # Started with its output thrown away (the restart does), Python writes in the console's code page, which has no emoji.
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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

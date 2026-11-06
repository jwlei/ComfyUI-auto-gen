"""Optional OpenH3-IR compilation without changing an H3 render graph."""
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time

import ffmpeg
import requests

from core import director
from core.paths import PROJECT_ROOT

CREATIVITY = ("restrained", "balanced", "bold", "extreme")
TIMINGS = ("off", "review", "render")
ASPECTS = {"21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3,
           "1:1": 1, "3:4": 3 / 4, "9:16": 9 / 16}
TIMEOUT = 3600


def settings(mode, creativity):
    mode, creativity = mode or "off", creativity or "balanced"
    if mode not in TIMINGS:
        raise ValueError("Unknown OpenH3-IR timing.")
    if creativity not in CREATIVITY:
        raise ValueError("Unknown OpenH3-IR creativity.")
    return mode, creativity


def supports(workflow):
    directors = [node for node in workflow.values() if node.get("class_type") == director.DIRECTOR]
    if directors:
        inputs = workflow.get(director.INPUTS_ID, {})
        assets = workflow.get(director.ASSETS_ID, {})
        return (len(directors) == 1
                and directors[0].get("inputs", {}).get("director_inputs") == [director.INPUTS_ID, 0]
                and inputs.get("class_type") == director.INPUTS
                and inputs.get("inputs", {}).get("ref_assets_1") == [director.ASSETS_ID, 0]
                and assets.get("class_type") == director.ASSETS)
    return any(node.get("class_type") == "MiniMaxH3ReferenceToVideo" for node in workflow.values())


def supports_path(path):
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as source:
            return supports(json.load(source))
    except (OSError, ValueError, AttributeError):
        return False


def first_frame(workflow, images):
    if len(images) != 1 or not any(n.get("class_type") == director.DIRECTOR for n in workflow.values()):
        return False
    inputs = workflow.get(director.INPUTS_ID, {})
    return inputs.get("_meta", {}).get(director.IMAGES_AS, "start") == "start"


def _paths():
    launcher = os.getenv("COMFYUI_LAUNCHER") or ""
    root = os.getenv("H3IR_COMFY_ROOT") or os.path.join(os.path.dirname(launcher), "ComfyUI")
    python = os.getenv("H3IR_PYTHON") or os.path.join(os.path.dirname(launcher), "python_embeded", "python.exe")
    base = os.path.dirname(os.path.dirname(launcher))
    server = os.getenv("H3IR_LLAMA_SERVER") or os.path.join(base, "llama.cpp", "b12177-vulkan", "llama-server.exe")
    models = os.path.join(base, "models", "starnodes-qwen3.8-vl-27b-promptrefiner-abliterated")
    model = os.getenv("H3IR_MODEL_FILE") or os.path.join(models, "model.gguf")
    projector = os.getenv("H3IR_PROJECTOR_FILE") or os.path.join(models, "mmproj.gguf")
    for label, path in (("ComfyUI Python", python), ("OpenH3-IR node pack", os.path.join(root, "custom_nodes", "ComfyUI-OpenH3-IR", "compiler.py")),
                        ("llama.cpp server", server), ("Qwen model", model), ("vision projector", projector)):
        if not os.path.isfile(path):
            raise RuntimeError(f"OpenH3-IR needs {label}: {path}. See docs/video.md.")
    return root, python, server, model, projector


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Refiner:
    def __init__(self, comfy_url=None):
        self.root, self.python, server, model, projector = _paths()
        self.process = None
        self.server = server
        self.model = model
        self.projector = projector
        self.comfy_url = comfy_url

    def __enter__(self):
        if self.comfy_url:
            try:
                queue = requests.get(f"{self.comfy_url}/queue", timeout=10)
                queue.raise_for_status()
                state = queue.json()
                if not isinstance(state, dict) or not all(isinstance(state.get(key), list) for key in ("queue_running", "queue_pending")):
                    raise RuntimeError("ComfyUI returned an invalid queue status; cannot safely release its models.")
                if state.get("queue_running") or state.get("queue_pending"):
                    raise RuntimeError("ComfyUI has work running or queued; wait for it before using OpenH3-IR.")
                reply = requests.post(f"{self.comfy_url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
                reply.raise_for_status()
            except (requests.RequestException, ValueError) as e:
                raise RuntimeError(f"Could not release ComfyUI models before OpenH3-IR: {e}") from e
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"
        args = [self.server, "--model", self.model, "--mmproj", self.projector, "--host", "127.0.0.1",
                "--port", str(port), "--ctx-size", "16384", "--parallel", "1", "--threads", "8",
                "--gpu-layers", os.getenv("H3IR_GPU_LAYERS") or "16", "--alias", "h3-refiner"]
        self.log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self.process = subprocess.Popen(args, stdout=self.log, stderr=self.log,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            print("🧠 Starting the local OpenH3-IR model...", flush=True)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError(f"OpenH3-IR model server stopped: {self.log.read()[-3000:]}")
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                    if response.ok and any(m.get("id") == "h3-refiner" for m in response.json().get("data", [])):
                        return self
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(1)
            raise RuntimeError("OpenH3-IR model server did not start within 180 seconds.")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "log"):
            self.log.close()

    def compile(self, prompt, seconds, width, height, creativity, images=(), image_roles=(), videos=(), audios=(), seed=7,
                first_frame=False):
        _, creativity = settings("render", creativity)
        ratio = width / height
        aspect = min(ASPECTS, key=lambda a: abs(ASPECTS[a] - ratio))
        assets = []
        with tempfile.TemporaryDirectory(prefix="openh3-") as temp:
            for index, path in enumerate(images):
                role = image_roles[index - 1] if index and index - 1 < len(image_roles) else "context"
                assets.append({"kind": "image", "role": "frame_anchor_first" if first_frame else "environment" if role == "place" else "subject",
                               "path": path, "sizing": "match",
                               "note": "Another view of the first subject." if role == "view" else ""})
            for index, video in enumerate(videos):
                path = video["path"]
                length = video["seconds"]
                assets.append({"kind": "video", "role": "structure" if video["role"] == "motion" else "subject",
                               "path": path, "seconds": length, "frames": round(length * 24),
                               "note": "The environment shown in the video." if video["role"] == "place" else ""})
                if video.get("sound"):
                    sound = os.path.join(temp, f"soundtrack-{index}.wav")
                    try:
                        ffmpeg.input(path).output(sound, vn=None, ac=2, ar=48000).run(overwrite_output=True, quiet=True)
                    except ffmpeg.Error as e:
                        raise RuntimeError(f"Could not extract reference video soundtrack: {e}") from e
                    assets.append({"kind": "audio", "role": "bgm", "path": sound, "seconds": length,
                                   "paired_video_path": path})
            for audio in audios:
                assets.append({"kind": "audio", "role": {"voice": "voice_timbre", "music": "bgm", "sound": "sfx"}[audio["role"]],
                               "path": audio["path"], "seconds": audio["seconds"]})
            hashes = [digest(asset["path"]) for asset in assets]
            if len(set(hashes)) != len(hashes):
                raise RuntimeError("OpenH3-IR cannot safely distinguish identical reference files; use distinct files.")
            brief = {"intent": prompt, "seconds": seconds, "aspect": aspect, "creativity": creativity,
                     "effort": "fast", "seed": int(seed if seed is not None else 7) % (2 ** 32),
                     "silent": False, "shots": "auto", "assets": assets, "transcripts": {}}
            request = {"comfy_root": self.root, "brief": brief, "url": self.url, "model": "h3-refiner", "timeout": TIMEOUT}
            worker = os.path.join(PROJECT_ROOT, "core", "h3_refiner_worker.py")
            print(f"🧠 OpenH3-IR: writing a brief ({creativity})...", flush=True)
            try:
                result = subprocess.run([self.python, worker], input=json.dumps(request), encoding="utf-8",
                                        capture_output=True, timeout=TIMEOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("OpenH3-IR timed out while writing the brief.") from e
            if result.returncode:
                raise RuntimeError(f"OpenH3-IR failed: {result.stderr[-4000:]}")
            try:
                body = json.loads(result.stdout)
                prompt_out, wiring, frames = body["prompt"], body["wiring"], body["frames"]
            except (ValueError, KeyError, TypeError) as e:
                raise RuntimeError(f"OpenH3-IR returned an invalid response: {result.stdout[-1000:]}") from e
            expected_frames = max(5, round(seconds * 24))
            expected_frames += (5 - expected_frames % 17) % 17
            if frames != expected_frames:
                raise RuntimeError(f"OpenH3-IR duration mismatch: {frames} frames, expected {expected_frames}.")
            expected_mode = "i2va" if first_frame else "ref2va" if assets else "t2va"
            if body.get("mode") != expected_mode:
                raise RuntimeError(f"OpenH3-IR returned {body.get('mode')!r} instead of {expected_mode}.")
            expected = []
            for kind, prefix in (("image", "ref_image"), ("video", "ref_video")):
                expected += [(f"{prefix}_{i}", hashes[assets.index(a)]) for i, a in enumerate(
                    (a for a in assets if a["kind"] == kind), 1)]
            standalone = [a for a in assets if a["kind"] == "audio" and not a.get("paired_video_path")]
            expected += [(f"ref_audio_{i}", hashes[assets.index(a)]) for i, a in enumerate(standalone, 1)]
            for i, video in enumerate(videos, 1):
                if video.get("sound"):
                    pair = next(a for a in assets if a.get("paired_video_path") == video["path"])
                    expected.append((f"ref_video_audio_{i}", hashes[assets.index(pair)]))
            actual = [(entry["wiring"], entry["sha256"]) for entry in wiring]
            if sorted(expected) != sorted(actual) or not isinstance(prompt_out, str) or not prompt_out.strip():
                raise RuntimeError("OpenH3-IR returned a brief whose reference bindings do not match the render.")
            if body.get("degraded"):
                print(f"⚠️ OpenH3-IR used its fallback: {body.get('fallback_reason')}", flush=True)
            return prompt_out

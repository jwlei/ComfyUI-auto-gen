"""Optional OpenH3-IR compilation without changing an H3 render graph."""
import hashlib
import base64
import json
import os
import re
import socket
import subprocess
import tempfile
import time

import ffmpeg
import requests

from core import director, h3_refiner_worker, prompt_writer_instructions
from core.paths import PROJECT_ROOT
from panel import settings as panel_settings

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


def selected_model():
    return panel_settings.load()["h3_model"]


def _paths(model_id, require_compiler=True):
    launcher = os.getenv("COMFYUI_LAUNCHER") or ""
    root = os.getenv("H3IR_COMFY_ROOT") or os.path.join(os.path.dirname(launcher), "ComfyUI")
    python = os.getenv("H3IR_PYTHON") or os.path.join(os.path.dirname(launcher), "python_embeded", "python.exe")
    base = os.path.dirname(os.path.dirname(launcher))
    server = os.getenv("H3IR_LLAMA_SERVER") or os.path.join(base, "1_Misc", "llama.cpp", "b12177-vulkan", "llama-server.exe")
    folder = {"gemma-4-12b": "gemma-4-12b-heretic", "qwen3-14b": "qwen3-14b-abliterated"}[model_id]
    models = os.path.join(PROJECT_ROOT, "models", folder)
    model = os.getenv("H3IR_MODEL_FILE") or os.path.join(models, "model.gguf")
    projector = (os.getenv("H3IR_PROJECTOR_FILE") or os.path.join(models, "mmproj.gguf")) if model_id == "gemma-4-12b" else None
    needed = [("llama.cpp server", server), ("OpenH3-IR model", model)]
    if require_compiler:
        needed += [("ComfyUI Python", python), ("OpenH3-IR node pack", os.path.join(root, "custom_nodes", "ComfyUI-OpenH3-IR", "compiler.py"))]
    if projector:
        needed.append(("vision projector", projector))
    for label, path in needed:
        if not os.path.isfile(path):
            raise RuntimeError(f"OpenH3-IR needs {label}: {path}. See docs/video.md.")
    return root, python, server, model, projector


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def video_prompt_format(text):
    text = re.sub(r"^T2VA\s*\n+", "", text.strip(), count=1)
    if text.startswith("[Shot 1]"):
        text = "integrated_multimodal_description: " + text
    names = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
             "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
    for name in names:
        text = re.sub(rf"(?m)^{name}[ \t]*$", name + ":", text)
    expected = (("subject_definitions", "summary", "retention_analysis", "detailed_description",
                 "overall_soundscape", "non_diegetic_music") if text.startswith("subject_definitions:")
                else ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"))
    positions = [re.search(rf"(?m)^{name}:", text) for name in expected]
    if (any(position is None for position in positions)
            or [position.start() for position in positions if position] != sorted(position.start() for position in positions if position)
            or "[Shot 1]" not in text):
        raise RuntimeError("The model did not return the required video prompt sections. Try again.")
    matches = [position for position in positions if position]
    for index, position in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if not text[position.end():end].strip():
            raise RuntimeError("The model returned an empty video prompt section. Try again.")
    prefix = text[:matches[0].start()].strip()
    if prefix:
        raise RuntimeError("The model returned unexpected text before the video prompt. Try again.")
    return text


class Refiner:
    def __init__(self, comfy_url=None, require_compiler=True):
        self.model_id = selected_model()
        self.root, self.python, server, model, projector = _paths(self.model_id, require_compiler)
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
        args = [self.server, "--model", self.model]
        if self.projector:
            args += ["--mmproj", self.projector]
        args += ["--host", "127.0.0.1",
                "--port", str(port), "--ctx-size", "16384", "--parallel", "1", "--threads", "8",
                "--gpu-layers", os.getenv("H3IR_GPU_LAYERS") or "16", "--alias", "h3-refiner"]
        if self.model_id == "qwen3-14b":
            args += ["--reasoning-format", "deepseek"]
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

    def write_prompt(self, text, pipeline):
        instruction = prompt_writer_instructions.INSTRUCTIONS[pipeline]
        try:
            response = requests.post(f"{self.url}/chat/completions", json={
                "model": "h3-refiner", "messages": [{"role": "system", "content": instruction},
                                                      {"role": "user", "content": text}],
                "max_tokens": 4096 if self.model_id == "qwen3-14b" else 2048, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": self.model_id == "qwen3-14b"},
            }, timeout=900)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            result = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("The generated prompt was cut off; try a shorter input.")
            if not isinstance(result, str) or not result.strip():
                raise RuntimeError("The model returned an empty prompt.")
            if self.model_id == "qwen3-14b":
                _, result = h3_refiner_worker.split_qwen_reasoning(result)
            if "<think>" in result or "</think>" in result:
                raise RuntimeError("The model returned its reasoning instead of a clean prompt. Try again.")
            if not result.strip():
                raise RuntimeError("The model returned an empty prompt.")
            return video_prompt_format(result) if pipeline == "video" else result.strip()
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Could not write the prompt: {e}") from e

    def plan(self, prompt, schema, images=()):
        if self.model_id == "qwen3-14b" and images:
            raise RuntimeError("Qwen3 14B is text-only and cannot analyze attached images. Choose Gemma 4 in Settings.")
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                    base64.b64encode(image).decode("ascii")}} for image in images]
        content.append({"type": "text", "text": prompt})
        try:
            response = requests.post(f"{self.url}/chat/completions", json={
                "model": "h3-refiner", "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "prompt_plan", "strict": True, "schema": schema}},
                "max_tokens": 8192, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=TIMEOUT)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("The local model's plan was cut off. Try fewer images or a shorter video.")
            result = choice["message"]["content"]
            if not isinstance(result, str) or not result.strip():
                raise RuntimeError("The local model returned an empty plan.")
            if self.model_id == "qwen3-14b":
                _, result = h3_refiner_worker.split_qwen_reasoning(result)
            return result.strip()
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Could not get a plan from the local model: {e}") from e

    def compile(self, prompt, seconds, width, height, creativity, images=(), image_roles=(), videos=(), audios=(), seed=7,
                first_frame=False):
        _, creativity = settings("render", creativity)
        if self.model_id == "qwen3-14b" and (images or videos):
            raise RuntimeError("Qwen3 14B is text-only and cannot analyze image or video references. Choose Gemma 4 in Settings.")
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
            request = {"comfy_root": self.root, "brief": brief, "url": self.url, "model": "h3-refiner",
                       "model_id": self.model_id, "timeout": TIMEOUT}
            worker = os.path.join(PROJECT_ROOT, "core", "h3_refiner_worker.py")
            print(f"🧠 OpenH3-IR: writing a brief ({creativity})...", flush=True)
            try:
                result = subprocess.run([self.python, worker], input=json.dumps(request), encoding="utf-8",
                                        capture_output=True, timeout=TIMEOUT,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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
            if self.model_id == "qwen3-14b" and expected_mode == "t2va":
                start = prompt_out.find("integrated_multimodal_description:")
                if start > 0:
                    print("⚠️ Removed unexpected text before the OpenH3-IR prompt sections.", flush=True)
                    prompt_out = prompt_out[start:]
            return prompt_out

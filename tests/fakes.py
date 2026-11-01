"""
Stand-ins for ComfyUI and the Claude API on localhost: real HTTP, no GPU, no cost.

Imported before anything else in the tests, so every file they make goes to a
throwaway folder, including from the pipeline processes the panel starts.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import ffmpeg

# main.py does this for a real run. The pipeline's emoji don't fit a Windows console's default.
sys.stdout.reconfigure(encoding="utf-8")

DATA = tempfile.mkdtemp(prefix="autocomfyui_tests_")
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ["AUTOCOMFYUI_DATA"] = DATA
# load_dotenv never replaces what is already set, so the real .env can't reach the tests.
for name in ("APP_PASSWORD", "COMFYUI_LAUNCHER", "GEMINI_MODEL", "CLAUDE_MODEL", "LLM_PROVIDER", "COMFYUI_WORKFLOW",
             "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
    os.environ[name] = ""


def make_clip(path, seconds, color="red", audio=True):
    video = ffmpeg.input(f"color=c={color}:s=64x64:r=24:d={seconds}", f="lavfi")
    streams = [video]
    if audio:
        streams.append(ffmpeg.input(f"sine=f=440:d={seconds}", f="lavfi"))
    ffmpeg.output(*streams, path, vcodec="libx264", pix_fmt="yuv420p", acodec="aac", preset="ultrafast") \
        .run(overwrite_output=True, quiet=True)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, status=200, kind="application/json"):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _clip_seconds(workflow):
    """Wherever this workflow says how long a clip is: a node of seconds, or a latent's frames."""
    for node in workflow.values():
        if "SECONDS" in node["_meta"]["title"].upper() or "DURATION" in node["_meta"]["title"].upper():
            return node["inputs"]["value"]
    return next(node["inputs"]["length"] / 24 for node in workflow.values() if isinstance(node["inputs"].get("length"), int))


class _ComfyHandler(_Handler):
    def do_GET(self):
        url = urlparse(self.path)
        state = self.server
        if url.path == "/system_stats":
            self._send({})
        elif url.path == "/object_info":
            self._send(state.object_info)
        elif url.path == "/api/userdata":
            self._send(list(state.saved))
        elif url.path.startswith("/api/userdata/"):
            name = unquote(url.path.rsplit("/", 1)[1]).removeprefix("workflows/")
            self._send(state.saved[name] if name in state.saved else {}, 200 if name in state.saved else 404)
        elif url.path == "/queue":
            self._send(state.queue)
        elif url.path.startswith("/history/"):
            prompt_id = url.path.rsplit("/", 1)[1]
            if prompt_id in state.hanging:
                self._send({})
            elif prompt_id in state.errors:
                self._send({prompt_id: {"status": {"status_str": "error", "messages": [
                    ["execution_error", {"node_type": "FakeNode", "exception_message": "out of memory"}]]}}})
            else:
                self._send({prompt_id: {"status": {"completed": True}, "outputs": {"140": {"gifs": [
                    {"filename": f"{prompt_id}.png", "type": "temp"},
                    {"filename": f"{prompt_id}.mp4", "subfolder": "", "type": "temp"}]}}}})
        elif url.path == "/view":
            with open(os.path.join(state.folder, parse_qs(url.query)["filename"][0]), "rb") as f:
                self._send(f.read(), kind="video/mp4")
        else:
            self._send({}, 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        state = self.server
        if self.path == "/upload/image":
            # The image alone: the multipart boundary around it differs on every request.
            boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
            part = next(p for p in body.split(b"--" + boundary) if b'name="image"' in p)
            state.uploads.append(part.split(b"\r\n\r\n", 1)[1])
            self._send({"name": "uploaded.png", "subfolder": ""})
        elif self.path == "/prompt":
            data = json.loads(body)
            with state.lock:
                state.prompts.append(data)
                number = len(state.prompts)
                step = state.script.pop(0) if state.script else "ok"
            prompt_id = f"p{number}"
            seconds = _clip_seconds(data["prompt"])
            if step == "error":
                state.errors.add(prompt_id)
            elif step == "hang":
                state.hanging.add(prompt_id)
                state.queue = {"queue_running": [[0, prompt_id, {}, {"client_id": data["client_id"]}]], "queue_pending": []}
            else:
                make_clip(os.path.join(state.folder, f"{prompt_id}.mp4"), 0.5 if step == "short" else seconds,
                          color=f"0x{(number * 40) % 256:02x}2040", audio=step != "silent")
            self._send({"prompt_id": prompt_id})
        elif self.path in ("/queue", "/interrupt"):
            state.cancels.append((self.path, json.loads(body)))
            if self.path == "/interrupt":
                state.queue = {"queue_pending": [], "queue_running": []}
            self._send({})
        else:
            self._send({}, 404)


class _ClaudeHandler(_Handler):
    def do_POST(self):
        state = self.server
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        state.requests.append({"body": request, "beta": self.headers.get("anthropic-beta"), "path": self.path})
        if state.status != 200:
            self._send({"type": "error", "error": {"type": "api_error", "message": "fake failure"}}, state.status)
            return
        reply = state.reply(request) if callable(state.reply) else state.reply
        events = [{"type": "message_start", "message": {
            "id": "msg_fake", "type": "message", "role": "assistant", "content": [], "model": request["model"],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}}]
        for index, text in enumerate(reply):
            events += [{"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
                       {"type": "content_block_stop", "index": index}]
        events += [{"type": "message_delta", "delta": {"stop_reason": state.stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 5}},
                   {"type": "message_stop"}]
        stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        self._send(stream.encode(), kind="text/event-stream")


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def fake_comfy(folder):
    server = _serve(_ComfyHandler)
    server.folder, server.lock = folder, threading.Lock()
    server.prompts, server.uploads, server.script, server.cancels = [], [], [], []
    server.errors, server.hanging = set(), set()
    server.queue = {"queue_pending": [], "queue_running": []}
    server.saved, server.object_info = {}, {}
    server.url = f"http://127.0.0.1:{server.server_port}"
    return server


def _ui_node(node_id, kind, widgets=None, links=None, outputs=(), values=None, title=None, mode=0):
    inputs = [{"name": name, "type": name.upper(), "link": link} for name, link in (links or {}).items()]
    inputs += [{"name": name, "widget": {"name": name}, "link": None} for name in widgets or {}]
    return {"id": node_id, "type": kind, "title": title, "mode": mode, "inputs": inputs,
            "outputs": [{"type": kind} for kind in outputs], "widgets_values": values or list((widgets or {}).values())}


# What ComfyUI says about the nodes in saved_workflow().
OBJECT_INFO = {kind: {"display_name": kind, "input": {"required": required}} for kind, required in {
    "CheckpointLoader": {"ckpt_name": [["model.safetensors"]]},
    "LoraLoaderModelOnly": {"model": ["MODEL"], "lora_name": [["lora.safetensors"]]},
    "CLIPTextEncode": {"text": ["STRING", {}], "clip": ["CLIP"]},
    "LoadImage": {"image": [["example.png"], {"image_upload": True}]},
    "LatentVideo": {"width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}]},
    "PrimitiveStringMultiline": {"value": ["STRING", {}]},
    "PrimitiveFloat": {"value": ["FLOAT", {}]},
    "SizePicker": {"megapixels": ["FLOAT", {}]},
    "ReferenceToVideo": {"prompt": ["STRING", {}], "width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}]},
    "KSampler": {"seed": ["INT", {}], "steps": ["INT", {}], "model": ["MODEL"], "positive": ["CONDITIONING"],
                 "negative": ["CONDITIONING"], "latent_image": ["LATENT"]},
    "VAEDecode": {"samples": ["LATENT"]},
    "PreviewImage": {"images": ["IMAGE"]},
    "CreateVideo": {"images": ["IMAGE"], "fps": ["FLOAT", {}]},
    "SaveVideo": {"video": ["VIDEO"], "filename_prefix": ["STRING", {}]},
    "VHS_VideoCombine": {"images": ["IMAGE"], "frame_rate": ["FLOAT", {}], "save_output": ["BOOLEAN", {}]},
}.items()}
# Inputs a workflow runs fine without.
OBJECT_INFO["LatentVideo"]["input"]["optional"] = {"start_image": ["IMAGE"]}
OBJECT_INFO["ReferenceToVideo"]["input"]["optional"] = {"ref_images": ["COMFY_AUTOGROW_V3", {}]}


def saved_workflow():
    """As ComfyUI saves one, with what only its editor knows: Set/Get, a note, a bypass, a mute, and a seed's dropdown."""
    return {"nodes": [
        _ui_node(1, "CheckpointLoader", {"ckpt_name": "model.safetensors"}),
        _ui_node(9, "LoraLoaderModelOnly", {"lora_name": "not installed, but bypassed"}, {"model": 1}, ["MODEL"], mode=4),
        _ui_node(2, "CLIPTextEncode", {"text": "saved positive"}, {"clip": 2}),
        _ui_node(3, "CLIPTextEncode", {"text": "saved negative"}, {"clip": 7}),
        _ui_node(20, "SetNode", links={"CONDITIONING": 4}, values=["positive"]),
        _ui_node(21, "GetNode", values=["positive"]),
        _ui_node(4, "LoadImage", {"image": "example.png"}, values=["example.png", "image"], mode=4),
        _ui_node(5, "LatentVideo", {"width": 1280, "height": 704, "length": 121}, {"start_image": 9}),
        _ui_node(6, "KSampler", {"seed": 0, "steps": 0}, {"model": 3, "positive": 6, "negative": 8, "latent_image": 10},
                 values=[123, "randomize", 20]),
        _ui_node(7, "VAEDecode", links={"samples": 11}),
        _ui_node(8, "PreviewImage", links={"images": 13}),
        _ui_node(10, "CreateVideo", {"fps": 24}, {"images": 12}),
        _ui_node(11, "SaveVideo", {"filename_prefix": "video/ComfyUI"}, {"video": 14}),
        _ui_node(12, "MarkdownNote", values=["notes"]),
        _ui_node(13, "LoraLoaderModelOnly", {"lora_name": "not installed, but muted"}, mode=2),
    ], "links": [
        [1, 1, 0, 9, 0, "MODEL"], [2, 1, 1, 2, 0, "CLIP"], [3, 9, 0, 6, 2, "MODEL"], [4, 2, 0, 20, 0, "CONDITIONING"],
        [6, 21, 0, 6, 3, "CONDITIONING"], [7, 1, 1, 3, 0, "CLIP"], [8, 3, 0, 6, 4, "CONDITIONING"], [9, 4, 0, 5, 0, "IMAGE"],
        [10, 5, 0, 6, 5, "LATENT"], [11, 6, 0, 7, 0, "LATENT"], [12, 7, 0, 10, 0, "IMAGE"], [13, 7, 0, 8, 0, "IMAGE"],
        [14, 10, 0, 11, 0, "VIDEO"]]}


def reference_workflow():
    """Like MiniMax's R2V: two optional reference images, and a node that takes the prompt itself."""
    return {"nodes": [
        _ui_node(1, "LoadImage", {"image": "b.png"}, values=["b.png", "image"]),
        _ui_node(2, "LoadImage", {"image": "a.png"}, values=["a.png", "image"]),
        _ui_node(3, "PrimitiveStringMultiline", {"value": "saved prompt"}, title="Input Text (Prompt)"),
        _ui_node(4, "PrimitiveFloat", {"value": 5}, title="Float (Duration)"),
        _ui_node(5, "SizePicker", {"megapixels": 0.4}),
        _ui_node(6, "ReferenceToVideo", links={"ref_images.ref_image_0": 1, "ref_images.ref_image_1": 2, "prompt": 3,
                                               "width": 5, "height": 6, "length": 7}),
        _ui_node(7, "CreateVideo", {"fps": 24}, {"images": 8}),
        _ui_node(8, "SaveVideo", {"filename_prefix": "video/R2V"}, {"video": 9}),
    ], "links": [
        [1, 2, 0, 6, 0, "IMAGE"], [2, 1, 0, 6, 1, "IMAGE"], [3, 3, 0, 6, 2, "STRING"], [5, 5, 0, 6, 3, "INT"],
        [6, 5, 1, 6, 4, "INT"], [7, 4, 0, 6, 5, "FLOAT"], [8, 6, 0, 7, 0, "IMAGE"], [9, 7, 0, 8, 0, "VIDEO"]]}


def text_only_workflow():
    """A plain T2V: saved_workflow() without its image."""
    ui = saved_workflow()
    ui["nodes"] = [node for node in ui["nodes"] if node["id"] != 4]
    next(node for node in ui["nodes"] if node["id"] == 5)["inputs"] = [
        entry for entry in next(node for node in ui["nodes"] if node["id"] == 5)["inputs"] if entry["name"] != "start_image"]
    ui["links"] = [link for link in ui["links"] if link[0] != 9]
    return ui


def plan_reply(shots=8, seconds=8, title="The Harbour Storm"):
    return [json.dumps({"title": title, "shots": [
        {"title": f"Shot {i + 1}", "seconds": seconds, "prompt": f"Prompt number {i + 1}."} for i in range(shots)]})]


def fake_claude():
    server = _serve(_ClaudeHandler)
    server.requests, server.reply, server.status, server.stop_reason = [], plan_reply(), 200, "end_turn"
    server.url = f"http://127.0.0.1:{server.server_port}"
    return server


def close(server):
    server.shutdown()
    server.server_close()

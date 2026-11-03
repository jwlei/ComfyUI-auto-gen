"""
Stand-ins for ComfyUI and the Claude API on localhost: real HTTP, no GPU, no cost.

Imported before anything else in the tests, so every file they make goes to a
throwaway folder, including from the pipeline processes the panel starts.
"""
import atexit
import io
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import ffmpeg
from PIL import Image

# main.py does this for a real run. The pipeline's emoji don't fit a Windows console's default.
sys.stdout.reconfigure(encoding="utf-8")

DATA = tempfile.mkdtemp(prefix="comfyui_auto_gen_tests_")
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
os.environ["AUTOGEN_DATA"] = DATA
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
    """Wherever this workflow says how long a clip is: a node of seconds, or a latent's frames. None if it makes images."""
    for node in workflow.values():
        if "SECONDS" in node["_meta"]["title"].upper() or "DURATION" in node["_meta"]["title"].upper():
            return node["inputs"]["value"]
    return next((node["inputs"]["length"] / 24 for node in workflow.values() if isinstance(node["inputs"].get("length"), int)), None)


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
            elif seconds is None:
                path = os.path.join(state.folder, f"{prompt_id}.png")
                if step == "corrupt":
                    with open(path, "wb") as f:
                        f.write(b"not an image")
                else:
                    # What a real one does with an empty latent: the size it was told.
                    latent = next((node["inputs"] for node in data["prompt"].values()
                                   if isinstance(node["inputs"].get("width"), int) and isinstance(node["inputs"].get("height"), int)), {})
                    made = (latent["width"], latent["height"]) if latent else Image.open(io.BytesIO(state.uploads[-1])).size if state.uploads else (32, 32)
                    Image.new("RGB", made, (number * 40 % 256, 20, 40)).save(path)
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
    "SaveImage": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]},
    "SaveImageAdvanced": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}], "format": ["COMFY_DYNAMICCOMBO_V3", {}]},
    "EmptyLatentImage": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]},
    "ConditioningZeroOut": {"conditioning": ["CONDITIONING"]},
    "PrimitiveBoolean": {"value": ["BOOLEAN", {}]},
    "ComfySwitchNode": {"switch": ["BOOLEAN", {}]},
    "TextGenerate": {"prompt": ["STRING", {}], "clip": ["CLIP"]},
    "TextEncodeQwenImageEditPlus": {"prompt": ["STRING", {}], "clip": ["CLIP"]},
    "FluxKontextImageScale": {"image": ["IMAGE"]},
    "VAEEncode": {"pixels": ["IMAGE"]},
    # As ComfyUI has it: its images are a growing list that's required, and can hold none.
    "TextEncodeFibo2": {"clip": ["CLIP"], "prompt": ["STRING", {}], "negative_prompt": ["STRING", {}], "width": ["INT", {}], "height": ["INT", {}],
                        "batch_size": ["INT", {}], "images": ["COMFY_AUTOGROW_V3", {"template": {
                            "input": {"required": {"image": ["IMAGE", {}]}}, "names": [f"image_{n}" for n in range(1, 17)], "min": 0}}]},
}.items()}
# Inputs a workflow runs fine without.
OBJECT_INFO["LatentVideo"]["input"]["optional"] = {"start_image": ["IMAGE"]}
OBJECT_INFO["ReferenceToVideo"]["input"]["optional"] = {"ref_images": ["COMFY_AUTOGROW_V3", {}]}
# MiniMax H3's: up to 9 images, and 3 each of videos, their sound and audio clips.
OBJECT_INFO["MiniMaxH3ReferenceToVideo"] = {"display_name": "MiniMax H3 Reference to Video", "input": {
    "required": dict(OBJECT_INFO["ReferenceToVideo"]["input"]["required"]),
    "optional": {group: ["COMFY_AUTOGROW_V3", {"template": {"prefix": f"{group[:-1]}_", "min": 0, "max": most}}]
                 for group, most in (("ref_images", 9), ("ref_videos", 3), ("ref_video_audios", 3), ("ref_audios", 3))}}}
OBJECT_INFO["ComfySwitchNode"]["input"]["optional"] = {"on_false": ["COMFY_MATCHTYPE_V3", {}], "on_true": ["COMFY_MATCHTYPE_V3", {}]}
OBJECT_INFO["TextEncodeQwenImageEditPlus"]["input"]["optional"] = {"image1": ["IMAGE"], "image2": ["IMAGE"]}


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


def h3_workflow():
    """reference_workflow() on MiniMax H3's node, which also takes videos and audio clips that nothing is wired to."""
    ui = reference_workflow()
    next(node for node in ui["nodes"] if node["id"] == 6)["type"] = "MiniMaxH3ReferenceToVideo"
    return ui


def text_only_workflow():
    """A plain T2V: saved_workflow() without its image."""
    ui = saved_workflow()
    ui["nodes"] = [node for node in ui["nodes"] if node["id"] != 4]
    next(node for node in ui["nodes"] if node["id"] == 5)["inputs"] = [
        entry for entry in next(node for node in ui["nodes"] if node["id"] == 5)["inputs"] if entry["name"] != "start_image"]
    ui["links"] = [link for link in ui["links"] if link[0] != 9]
    return ui


def _inside(link_id, origin, slot, target, target_slot, kind):
    return {"id": link_id, "origin_id": origin, "origin_slot": slot, "target_id": target, "target_slot": target_slot, "type": kind}


def text_to_image_workflow():
    """
    Like ComfyUI's Krea-2 template: everything in a subgraph, set on its face, with its size from a node outside.
    A prompt refiner that's on and a style LoRA that's off, whose file isn't installed.
    """
    return {"nodes": [
        _ui_node(29, "SaveImage", {"filename_prefix": "Krea2_turbo"}, {"images": 44}),
        _ui_node(30, "sub-text", {"value": "saved prompt", "value_1": True, "seed_1": 99, "value_3": False, "lora_name_1": "krea2_darkbrush.safetensors"},
                 {"width_1": 84, "height_1": 85}, ["IMAGE"]),
        _ui_node(49, "SizePicker", {"megapixels": 1}, outputs=["INT", "INT"]),
        _ui_node(47, "MarkdownNote", values=["notes"]),
    ], "links": [[44, 30, 0, 29, 0, "IMAGE"], [84, 49, 0, 30, 3, "INT"], [85, 49, 1, 30, 4, "INT"]],
       "last_node_id": 50, "last_link_id": 90,
       "definitions": {"subgraphs": [{"id": "sub-text", "name": "Text to Image", "inputs": [
           {"name": "value", "type": "STRING"}, {"name": "value_1", "type": "BOOLEAN"}, {"name": "seed_1", "type": "INT"},
           {"name": "width_1", "type": "INT"}, {"name": "height_1", "type": "INT"}, {"name": "value_3", "type": "BOOLEAN"},
           {"name": "lora_name_1", "type": "COMBO"}],
           "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
           "nodes": [
               _ui_node(10, "CheckpointLoader", {"ckpt_name": "model.safetensors"}, outputs=["MODEL", "CLIP"]),
               _ui_node(15, "LoraLoaderModelOnly", links={"model": 16, "lora_name": 78}, outputs=["MODEL"]),
               _ui_node(19, "PrimitiveStringMultiline", links={"value": 45}, values=["what was typed inside"], outputs=["STRING"], title="Text String (User Prompt)"),
               _ui_node(24, "PrimitiveBoolean", links={"value": 46}, outputs=["BOOLEAN"], title="Boolean (Refine Prompt?)"),
               _ui_node(23, "PrimitiveBoolean", links={"value": 77}, outputs=["BOOLEAN"], title="Boolean (Enable LoRA?)"),
               _ui_node(16, "TextGenerate", {"prompt": ""}, {"clip": 18}, ["STRING"]),
               _ui_node(21, "ComfySwitchNode", links={"on_false": 34, "on_true": 33, "switch": 32}, outputs=["STRING"]),
               _ui_node(22, "ComfySwitchNode", links={"on_false": 28, "on_true": 29, "switch": 31}, outputs=["MODEL"]),
               _ui_node(6, "CLIPTextEncode", links={"clip": 10, "text": 72}, outputs=["CONDITIONING"]),
               _ui_node(13, "ConditioningZeroOut", links={"conditioning": 13}, outputs=["CONDITIONING"]),
               _ui_node(5, "EmptyLatentImage", {"batch_size": 1}, {"width": 74, "height": 75}, ["LATENT"]),
               _ui_node(3, "KSampler", {"steps": 8}, {"model": 30, "positive": 4, "negative": 14, "latent_image": 2, "seed": 76}, ["LATENT"]),
               _ui_node(8, "VAEDecode", links={"samples": 7}, outputs=["IMAGE"]),
               _ui_node(50, "Note", values=["notes inside"]),
           ],
           "links": [
               _inside(45, -10, 0, 19, 0, "STRING"), _inside(46, -10, 1, 24, 0, "BOOLEAN"), _inside(76, -10, 2, 3, 4, "INT"),
               _inside(74, -10, 3, 5, 0, "INT"), _inside(75, -10, 4, 5, 1, "INT"), _inside(77, -10, 5, 23, 0, "BOOLEAN"),
               _inside(78, -10, 6, 15, 1, "COMBO"), _inside(16, 10, 0, 15, 0, "MODEL"), _inside(18, 10, 1, 16, 0, "CLIP"),
               _inside(34, 19, 0, 21, 0, "STRING"), _inside(33, 16, 0, 21, 1, "STRING"), _inside(32, 24, 0, 21, 2, "BOOLEAN"),
               _inside(28, 10, 0, 22, 0, "MODEL"), _inside(29, 15, 0, 22, 1, "MODEL"), _inside(31, 23, 0, 22, 2, "BOOLEAN"),
               _inside(10, 10, 1, 6, 0, "CLIP"), _inside(72, 21, 0, 6, 1, "STRING"), _inside(13, 6, 0, 13, 0, "CONDITIONING"),
               _inside(30, 22, 0, 3, 0, "MODEL"), _inside(4, 6, 0, 3, 1, "CONDITIONING"), _inside(14, 13, 0, 3, 2, "CONDITIONING"),
               _inside(2, 5, 0, 3, 3, "LATENT"), _inside(7, 3, 0, 8, 0, "LATENT"), _inside(43, 8, 0, -20, 0, "IMAGE"),
           ]}]}}


def edit_image_workflow():
    """Like ComfyUI's Qwen-Image-Edit template: one Load Image into a subgraph, whose second image input is left empty, and a scaler."""
    return {"nodes": [
        _ui_node(41, "LoadImage", {"image": "example.png"}, values=["example.png", "image"], outputs=["IMAGE", "MASK"]),
        _ui_node(170, "sub-edit", {"prompt": "saved instruction", "prompt_1": "", "seed": 7}, {"image": 376, "image2": None}, ["IMAGE"]),
        _ui_node(195, "SaveImageAdvanced", {"filename_prefix": "Edit", "format": "png"}, {"images": 388}),
    ], "links": [[376, 41, 0, 170, 0, "IMAGE"], [388, 170, 0, 195, 0, "IMAGE"]],
       "definitions": {"subgraphs": [{"id": "sub-edit", "name": "Image Edit", "inputs": [
           {"name": "image", "type": "IMAGE"}, {"name": "image2", "type": "IMAGE"}, {"name": "prompt", "type": "STRING"},
           {"name": "prompt_1", "type": "STRING"}, {"name": "seed", "type": "INT"}],
           "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
           "nodes": [
               _ui_node(1, "CheckpointLoader", {"ckpt_name": "model.safetensors"}, outputs=["MODEL", "CLIP"]),
               _ui_node(160, "FluxKontextImageScale", links={"image": 362}, outputs=["IMAGE"]),
               _ui_node(149, "TextEncodeQwenImageEditPlus", links={"clip": 1, "image1": 322, "image2": 366, "prompt": 378}, outputs=["CONDITIONING"]),
               _ui_node(151, "TextEncodeQwenImageEditPlus", links={"clip": 2, "image1": 325, "image2": 363, "prompt": 370}, outputs=["CONDITIONING"]),
               _ui_node(156, "VAEEncode", links={"pixels": 334}, outputs=["LATENT"]),
               _ui_node(169, "KSampler", {"steps": 40}, {"model": 3, "positive": 4, "negative": 5, "latent_image": 6, "seed": 387}, ["LATENT"]),
               _ui_node(158, "VAEDecode", links={"samples": 7}, outputs=["IMAGE"]),
           ],
           "links": [
               _inside(362, -10, 0, 160, 0, "IMAGE"), _inside(363, -10, 1, 151, 2, "IMAGE"), _inside(366, -10, 1, 149, 2, "IMAGE"),
               _inside(370, -10, 2, 151, 3, "STRING"), _inside(378, -10, 3, 149, 3, "STRING"), _inside(387, -10, 4, 169, 4, "INT"),
               _inside(1, 1, 1, 149, 0, "CLIP"), _inside(2, 1, 1, 151, 0, "CLIP"), _inside(322, 160, 0, 149, 1, "IMAGE"),
               _inside(325, 160, 0, 151, 1, "IMAGE"), _inside(334, 160, 0, 156, 0, "IMAGE"), _inside(3, 1, 0, 169, 0, "MODEL"),
               _inside(4, 151, 0, 169, 1, "CONDITIONING"), _inside(5, 149, 0, 169, 2, "CONDITIONING"), _inside(6, 156, 0, 169, 3, "LATENT"),
               _inside(7, 169, 0, 158, 0, "LATENT"), _inside(344, 158, 0, -20, 0, "IMAGE"),
           ]}]}}


def fibo2_workflow():
    """Like the Fibo 2.0 edit turbo test you'd save: no image_ in its name, its one Load Image bypassed, so it starts from text."""
    return {"nodes": [
        _ui_node(1, "CheckpointLoader", {"ckpt_name": "model.safetensors"}, outputs=["MODEL", "CLIP"]),
        _ui_node(14, "LoadImage", {"image": "example.png"}, values=["example.png", "image"], outputs=["IMAGE", "MASK"], mode=4),
        _ui_node(4, "TextEncodeFibo2", {"prompt": "saved caption", "negative_prompt": "", "width": 0, "height": 0, "batch_size": 1},
                 {"clip": 1, "images.image_1": 23}, ["CONDITIONING", "CONDITIONING", "LATENT"]),
        _ui_node(6, "KSampler", {"seed": 64, "steps": 4}, {"model": 20, "positive": 3, "negative": 4, "latent_image": 22}, ["LATENT"],
                 values=[64, "fixed", 4]),
        _ui_node(7, "VAEDecode", links={"samples": 6}, outputs=["IMAGE"]),
        _ui_node(8, "SaveImage", {"filename_prefix": "Fibo2_dev"}, {"images": 8}),
    ], "links": [[1, 1, 1, 4, 0, "CLIP"], [23, 14, 0, 4, 1, "IMAGE"], [20, 1, 0, 6, 0, "MODEL"], [3, 4, 0, 6, 1, "CONDITIONING"],
                 [4, 4, 1, 6, 2, "CONDITIONING"], [22, 4, 2, 6, 3, "LATENT"], [6, 6, 0, 7, 0, "LATENT"], [8, 7, 0, 8, 0, "IMAGE"]]}


def caption(number, edit=True):
    """A Fibo 2.0 prompt as the LLM writes it: a JSON caption, as text, with an edit_instruction when it edits."""
    found = {"short_description": f"A woman in look number {number}.", "objects": [{"description": "a woman", "clothing": f"look {number}"}]}
    return json.dumps({**found, "edit_instruction": f"Change her outfit to look number {number}."} if edit else found)


def captions_reply(count=3, edit=True, title="Three Looks"):
    return [json.dumps({"title": title, "versions": [{"title": f"Look {i + 1}", "prompt": caption(i + 1, edit)} for i in range(count)]})]


def versions_reply(count=3, title="Three Looks"):
    return [json.dumps({"title": title, "versions": [{"title": f"Look {i + 1}", "prompt": f"Change to look number {i + 1}."} for i in range(count)]})]


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

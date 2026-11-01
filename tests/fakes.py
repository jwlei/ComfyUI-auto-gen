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
from urllib.parse import parse_qs, urlparse

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


class _ComfyHandler(_Handler):
    def do_GET(self):
        url = urlparse(self.path)
        state = self.server
        if url.path == "/system_stats":
            self._send({})
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
            seconds = data["prompt"]["291"]["inputs"]["value"]
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
    server.url = f"http://127.0.0.1:{server.server_port}"
    return server


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

import copy
import json
import os
import random
import time

import requests

from modules.paths import PROJECT_ROOT

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = os.path.join(PROJECT_ROOT, "comfyui", "video_workflow.json")
# Counts time spent waiting in ComfyUI's queue too, which other work may be ahead in.
DEFAULT_TIMEOUT = 3600
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".gif")
SEED_INPUTS = ("seed", "noise_seed")
POLL_INTERVAL = 5
CLIENT_ID = "autocomfyui"


def client_id():
    return os.getenv("COMFYUI_CLIENT_ID") or CLIENT_ID


class ComfyClient:
    """
    Renders clips with a ComfyUI workflow exported in API format.

    The workflow sits next to a '.map.json' that says which node inputs to fill,
    as "<node id>.<input name>": prompt and image (required), text_to_video,
    seconds, width, height. Any 'seed' / 'noise_seed' input is randomised on every
    run.
    """

    def __init__(self):
        self.url = (os.getenv("COMFYUI_URL") or DEFAULT_COMFYUI_URL).rstrip("/")
        self.workflow_path = os.getenv("COMFYUI_WORKFLOW") or DEFAULT_WORKFLOW
        self.timeout = int(os.getenv("COMFYUI_TIMEOUT") or DEFAULT_TIMEOUT)
        # Fixed, so the panel can listen on ComfyUI's websocket for this run's progress
        # (ComfyUI only sends progress to the client that queued the job), and so
        # cancelling never touches anyone else's renders.
        self.client_id = client_id()
        self.workflow, self.fields = self._load_workflow()

    def _load_workflow(self):
        map_path = os.path.splitext(self.workflow_path)[0] + ".map.json"
        for path in (self.workflow_path, map_path):
            if not os.path.exists(path):
                raise RuntimeError(f"ComfyUI workflow file missing: {path}")

        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        with open(map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)

        if "nodes" in workflow:
            raise RuntimeError(f"{self.workflow_path} is a UI workflow. Export it with 'Export (API)' instead.")

        fields = {}
        for field, path in raw_map.items():
            if field.startswith("_"):
                continue  # Comments
            node_id, _, input_name = str(path).partition(".")
            if input_name not in workflow.get(node_id, {}).get("inputs", {}):
                raise RuntimeError(f"{map_path}: '{field}' points at '{path}', which is not in the workflow.")
            fields[field] = (node_id, input_name)

        for required in ("prompt", "image"):
            if required not in fields:
                raise RuntimeError(f"{map_path} must map '{required}'.")
        return workflow, fields

    def is_reachable(self):
        try:
            return requests.get(f"{self.url}/system_stats", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def wait_until_reachable(self):
        """
        The panel starts ComfyUI when a render starts and sets COMFYUI_STARTUP_WAIT,
        since loading its models takes a while. Without it, an offline ComfyUI fails
        straight away.
        """
        deadline = time.time() + int(os.getenv("COMFYUI_STARTUP_WAIT") or 0)
        if self.is_reachable():
            return True
        print(f"   ⏳ Waiting for ComfyUI to start at {self.url}...")
        while time.time() < deadline:
            time.sleep(5)
            if self.is_reachable():
                return True
        return False

    def _set(self, workflow, field, value):
        if field in self.fields:
            node_id, input_name = self.fields[field]
            workflow[node_id]["inputs"][input_name] = value

    def render(self, prompt, image, seconds, width, height, save_as):
        """Renders one clip and saves it as save_as plus its extension. Returns the path, or raises."""
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._set(workflow, "image", image)
        self._set(workflow, "text_to_video", False)
        self._set(workflow, "seconds", seconds)
        self._set(workflow, "width", width)
        self._set(workflow, "height", height)

        # Fresh seeds every run: ComfyUI would otherwise return its cached result.
        for node in workflow.values():
            for key in SEED_INPUTS:
                if isinstance(node.get("inputs", {}).get(key), int):
                    node["inputs"][key] = random.randint(0, 2**32 - 1)
            # Keeps our clips out of ComfyUI's output folder: they go to its temp
            # folder, which it clears itself, and are downloaded from there below.
            if "save_output" in node.get("inputs", {}):
                node["inputs"]["save_output"] = False

        response = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {response.text[:800]}")

        try:
            entry = self._wait_for(response.json()["prompt_id"])
        except KeyboardInterrupt:
            print("   🛑 Cancelling this clip in ComfyUI...")
            self.cancel()
            raise
        return self._download(entry, save_as)

    def _download(self, entry, save_as):
        outputs = [item
                   for node_output in entry.get("outputs", {}).values()
                   for items in node_output.values() if isinstance(items, list)
                   for item in items
                   if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(VIDEO_EXTENSIONS)]
        if not outputs:
            raise RuntimeError("The workflow finished without saving a video.")

        video = outputs[-1]
        save_path = save_as + os.path.splitext(video["filename"])[1]
        params = {"filename": video["filename"], "subfolder": video.get("subfolder", ""), "type": video.get("type", "output")}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Under another name until complete: a cut-off download must not pass as a clip.
        part_path = save_path + ".part"
        with requests.get(f"{self.url}/view", params=params, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.replace(part_path, save_path)
        return save_path

    def _wait_for(self, prompt_id):
        """Polls until the prompt finishes. Returns its history entry, or raises on error or timeout."""
        started = time.time()
        while time.time() - started < self.timeout:
            entry = requests.get(f"{self.url}/history/{prompt_id}", timeout=30).json().get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    details = [f"{data.get('node_type')}: {data.get('exception_message', '').strip()}"
                               for kind, data in status.get("messages", []) if kind == "execution_error"]
                    raise RuntimeError("; ".join(details) or "ComfyUI stopped this clip.")
                if status.get("completed", True):
                    return entry
            time.sleep(POLL_INTERVAL)

        self.cancel()
        raise RuntimeError(f"Timed out after {self.timeout}s.")

    def cancel(self):
        """Cancels this run's clips, rendering or waiting. A render you started in ComfyUI yourself carries on."""
        try:
            queue = requests.get(f"{self.url}/queue", timeout=5).json()
            waiting = [entry[1] for entry in queue.get("queue_pending", []) if entry[3].get("client_id") == self.client_id]
            # Waiting ones first, or the next clip starts as soon as this one stops.
            if waiting:
                requests.post(f"{self.url}/queue", json={"delete": waiting}, timeout=5)
            for entry in queue.get("queue_running", []):
                if entry[3].get("client_id") == self.client_id:
                    requests.post(f"{self.url}/interrupt", json={"prompt_id": entry[1]}, timeout=5)
        except requests.ConnectionError:
            pass  # ComfyUI isn't running, so it has nothing to cancel
        except (requests.RequestException, ValueError, LookupError, AttributeError) as e:
            print(f"   ⚠️ Could not cancel the ComfyUI clip, it may keep rendering: {e}")

    def upload(self, path, name):
        """Uploads to ComfyUI's input folder and returns the value LoadImage expects."""
        with open(path, "rb") as f:
            response = requests.post(f"{self.url}/upload/image",
                                     files={"image": (name, f, "application/octet-stream")},
                                     data={"overwrite": "true"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]

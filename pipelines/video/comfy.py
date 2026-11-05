import copy
import io
import json
import os
import time

import requests
from PIL import Image

from pipelines.video import h3_prompt, wan_prompt
from core import seeds, workflow_options
from core.paths import WORKFLOWS_DIR, comfyui_client_id as client_id
from core.saved_workflows import MOST_REFERENCES_FIELD, frames_for, image_slots, media_slots, takes

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = os.path.join(WORKFLOWS_DIR, "video_workflow.json")
# Counts time spent waiting in ComfyUI's queue too, which other work may be ahead in.
DEFAULT_TIMEOUT = 3600
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".gif")
POLL_INTERVAL = 5
# ComfyUI can stop answering for minutes while it works, with no harm to the render: loading SeedVR2's 7B model to
# the CPU holds up its web server too, and a long video holds it up longer. A slow reply is waited out; only this long
# without any answer gives up (the render's own timeout still caps it).
MOST_SILENCE = 1800
POLL_TIMEOUT = (10, 60)
# Queueing is answered as soon as ComfyUI has a moment, and one sent again could make the image twice.
QUEUE_TIMEOUT = (10, 180)


class ComfyClient:
    """
    Renders clips with a ComfyUI workflow exported in API format.

    The workflow sits next to a '.map.json' that says which node inputs to fill,
    as "<node id>.<input name>": prompt (required), image (image_2... for more, and
    image_link where one is plugged in if it can be left out), audio, text_to_video, seconds
    (or frames, with fps saying where its frame rate is), width, height. Reference videos and
    audio clips are ref_video and ref_audio (_2... for more), each with a _link to take out
    when it's left empty, and a video with the _split node making frames of it and its _sound.
    Every 'seed' / 'noise_seed' input gets the render's seed: the one asked for, or a random one.
    """

    def __init__(self, options=None):
        self.url = (os.getenv("COMFYUI_URL") or DEFAULT_COMFYUI_URL).rstrip("/")
        self.workflow_path = os.getenv("COMFYUI_WORKFLOW") or DEFAULT_WORKFLOW
        self.timeout = int(os.getenv("COMFYUI_TIMEOUT") or DEFAULT_TIMEOUT)
        # Fixed, so the panel can listen on ComfyUI's websocket for this run's progress
        # (ComfyUI only sends progress to the client that queued the job), and so
        # cancelling never touches anyone else's renders.
        self.client_id = client_id()
        self.workflow, self.fields, self.limits = self._load_workflow()
        self.options = self._offered()
        # The workflow's own options, as the panel chose them for this run.
        chosen = workflow_options.chosen_in_env() if options is None else options
        if chosen:
            self.workflow, self.fields = workflow_options.remake(self.workflow_path, chosen)

    def _offered(self):
        """What a run can change in this workflow, as the panel listed it."""
        with open(os.path.splitext(self.workflow_path)[0] + ".map.json", "r", encoding="utf-8") as f:
            return json.load(f).get("_options") or []

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

        if "prompt" not in fields:
            raise RuntimeError(f"{map_path} must map 'prompt'.")
        return workflow, fields, {key: raw_map[key] for key in (MOST_REFERENCES_FIELD,) if key in raw_map}

    def takes(self):
        return takes({**self.fields, **self.limits})

    def video_model(self):
        """The model, if its prompts have to be written a way of their own."""
        return video_model_of(self.workflow)

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

    def _fill_images(self, workflow, images):
        """Empty inputs come out of the workflow, or get a stand-in if its text switch needs them to exist."""
        self._set(workflow, "text_to_video", not images)
        for position, slot in enumerate(image_slots(self.fields)):
            if position < len(images):
                self._set(workflow, slot, images[position])
            elif f"{slot}_link" in self.fields:
                for node_id, input_name in [self.fields[f"{slot}_link"], *self.fields.get(f"{slot}_also", [])]:
                    workflow.get(node_id, {}).get("inputs", {}).pop(input_name, None)
                # Nothing uses it now, and an image it can't find would still stop ComfyUI.
                workflow.pop(self.fields[slot][0], None)
            else:
                self._set(workflow, slot, self._stand_in_image())

    def _fill_media(self, workflow, kind, uploaded):
        """Reference videos or audio clips into their slots. An empty slot comes out, with the nodes that only served it."""
        for position, slot in enumerate(media_slots(self.fields, kind)):
            if position < len(uploaded):
                self._set(workflow, slot, uploaded[position])
                continue
            for part in ("link", "sound"):
                if f"{slot}_{part}" in self.fields:
                    node_id, input_name = self.fields[f"{slot}_{part}"]
                    workflow.get(node_id, {}).get("inputs", {}).pop(input_name, None)
            for part in (slot, f"{slot}_split"):
                if part in self.fields:
                    workflow.pop(self.fields[part][0], None)

    def _stand_in_image(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buffer, format="PNG")
        buffer.seek(0)
        return self._send_image(buffer, "pipeline_no_image.png")

    def takes_audio(self):
        return "audio" in self.fields

    def render(self, prompt, images, seconds, width, height, save_as, audio=None, ref_videos=(), ref_audios=(), seed=None):
        """
        Renders one clip from these uploaded images, and reference videos and audio clips, and saves it as
        save_as plus its extension. Returns the path, or raises.
        """
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._fill_images(workflow, images)
        self._fill_media(workflow, "video", ref_videos)
        self._fill_media(workflow, "audio", ref_audios)
        self._set(workflow, "seconds", seconds)
        if "frames" in self.fields:
            node_id, input_name = self.fields["fps"]
            self._set(workflow, "frames", frames_for(seconds, workflow[node_id]["inputs"][input_name]))
        self._set(workflow, "width", width)
        self._set(workflow, "height", height)
        if audio:
            self._set(workflow, "audio", audio)
        return self._download(self._run(workflow, seed), save_as)

    def _run(self, workflow, seed=None):
        """Queues the filled-in workflow with this seed, or a random one, and waits. Returns its history entry, or raises."""
        # A random seed when none is asked for: ComfyUI would otherwise return its cached result.
        seeds.apply(workflow, seeds.pick(seed))
        for node in workflow.values():
            # Keeps our clips out of ComfyUI's output folder: they go to its temp
            # folder, which it clears itself, and are downloaded from there below.
            if "save_output" in node.get("inputs", {}):
                node["inputs"]["save_output"] = False

        response = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=QUEUE_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {response.text[:800]}")

        try:
            entry = self._wait_for(response.json()["prompt_id"])
        except KeyboardInterrupt:
            print("   🛑 Cancelling this clip in ComfyUI...")
            self.cancel()
            raise
        return entry

    def _download(self, entry, save_as, extensions=VIDEO_EXTENSIONS, what="a video"):
        outputs = [item
                   for node_output in entry.get("outputs", {}).values()
                   for items in node_output.values() if isinstance(items, list)
                   for item in items
                   if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(extensions)]
        if not outputs:
            raise RuntimeError(f"The workflow finished without saving {what}.")

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
        started = answered = time.time()
        quiet = False
        while time.time() - started < self.timeout:
            # Busy is not broken: before, one reply slower than 30s gave the image up while ComfyUI went on making it,
            # and the try after it waited behind, so neither was saved.
            try:
                entry = requests.get(f"{self.url}/history/{prompt_id}", timeout=POLL_TIMEOUT).json().get(prompt_id)
            except (requests.Timeout, requests.ConnectionError) as e:
                if time.time() - answered > MOST_SILENCE:
                    raise RuntimeError(f"ComfyUI has not answered for {MOST_SILENCE // 60} minutes ({e}).") from e
                if not quiet:
                    print("      ⏳ ComfyUI is too busy to answer right now: still waiting for it.")
                    quiet = True
                time.sleep(POLL_INTERVAL)
                continue
            answered, quiet = time.time(), False
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
        """Uploads to ComfyUI's input folder and returns the value LoadImage expects. Load Video and Load Audio take the same."""
        with open(path, "rb") as f:
            return self._send_image(f, name)

    def _send_image(self, file, name):
        response = requests.post(f"{self.url}/upload/image",
                                 files={"image": (name, file, "application/octet-stream")},
                                 data={"overwrite": "true"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]


def video_model_of(workflow):
    """The model a workflow in API format renders with, if its prompts have to be written a way of their own."""
    if any(node.get("class_type") == h3_prompt.REFERENCE_NODE for node in workflow.values()):
        return h3_prompt.MODEL
    return wan_prompt.MODEL if wan_prompt.uses_wan(workflow) else None


_models = {}


def workflow_model(path=None):
    """video_model_of() the workflow file at path, or the built-in one. None if it can't be read."""
    path = path or DEFAULT_WORKFLOW
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key not in _models:
            with open(path, "r", encoding="utf-8") as f:
                _models[key] = video_model_of(json.load(f))
        return _models[key]
    except (OSError, ValueError, AttributeError):
        return None

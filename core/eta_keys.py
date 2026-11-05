"""
What a render's time is learned under, for every side alike.

A render is timed under the model with its workflow, the options chosen and the size; under the same at any size; at
any options; and under the model in any workflow. An estimate takes the most particular one timed yet. So a new
workflow, model or setting needs nothing set up: it starts from its nearest relative and learns its own time from its
first render.

Times are kept in `logs/stage_timings_<side>.json` as {"steps": {...}, "renders": {key: seconds}}.
"""
import hashlib
import json
import os
import re

# The first real measurement replaces any guess; after that each one moves the stored value this much.
LEARNING_RATE = 0.3
# The inputs that name the model a workflow runs. An upscaler's model_name isn't one.
MODEL_INPUTS = ("unet_name", "ckpt_name", "model_name", "model")
MODEL_FILE = re.compile(r"\.(safetensors|gguf|ckpt|pt|pth|bin|sft)$", re.I)
NOT_THE_MODEL = re.compile(r"upscale|vae|clip|lora|seedvr|enhancer|textgenerate", re.I)
SEPARATOR = " | "

_models = {}


def models_in(api):
    """The model files a workflow in API format renders with, by name, without their folder and extension."""
    names = set()
    for node in api.values():
        if not isinstance(node, dict) or NOT_THE_MODEL.search(str(node.get("class_type", ""))):
            continue
        for key in MODEL_INPUTS:
            value = (node.get("inputs") or {}).get(key)
            if isinstance(value, str) and MODEL_FILE.search(value):
                names.add(MODEL_FILE.sub("", re.split(r"[\\/]", value)[-1]))
    return "+".join(sorted(names)) or None


def workflow_models(path):
    """models_in() the workflow file at path, read once per change to it. None if it can't be read."""
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key not in _models:
            with open(path, "r", encoding="utf-8") as f:
                _models[key] = models_in(json.load(f))
        return _models[key]
    except (OSError, TypeError, ValueError, AttributeError):
        return None


def render_keys(workflow, options=None, size=None, fallback="built-in"):
    """The keys a render is timed under, from the most particular to the least."""
    name = os.path.splitext(os.path.basename(workflow))[0] if workflow else fallback
    model = (workflow_models(workflow) if workflow else None) or name
    # A long text, such as a negative prompt, would make a key of paragraphs: it is told apart by a short digest.
    shown = lambda value: str(value) if len(str(value)) <= 24 else "#" + hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    chosen = ",".join(f"{key}={shown(value)}" for key, value in sorted((options or {}).items())) or "as saved"
    keys = [model, SEPARATOR.join((model, name)), SEPARATOR.join((model, name, chosen))]
    if size:
        keys.append(SEPARATOR.join((model, name, chosen, size)))
    return keys[::-1]


def options_in(argv):
    """The --option NAME=VALUE flags of an image run."""
    return dict(value.split("=", 1) for flag, value in zip(argv, argv[1:]) if flag == "--option" and "=" in value)


def options_in_env(env):
    """The workflow options a video or shorts run carries in COMFYUI_OPTIONS."""
    try:
        chosen = json.loads((env or {}).get("COMFYUI_OPTIONS") or "{}")
        return chosen if isinstance(chosen, dict) else {}
    except ValueError:
        return {}


class Timings:
    """The saved times: steps by name (some per LLM too) and renders by render_keys()."""

    def __init__(self, path, default_steps):
        self.path = path
        self.steps = dict(default_steps)
        self.renders = {}
        self.timed_steps = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
            steps = {key: value for key, value in (saved.get("steps") or {}).items() if number(value)}
            self.steps.update(steps)
            self.timed_steps = set(steps)
            self.renders = {key: value for key, value in (saved.get("renders") or {}).items() if number(value)}
        except (OSError, ValueError, AttributeError):
            pass  # A broken or old file only costs accuracy

    def step(self, keys, default=None):
        """The first of keys timed yet, else the first with a default, else default."""
        keys = [keys] if isinstance(keys, str) else keys
        for key in keys:
            if key in self.timed_steps:
                return self.steps[key]
        return next((self.steps[key] for key in keys if key in self.steps), default)

    def learn_step(self, keys, value):
        for key in [keys] if isinstance(keys, str) else keys:
            self.steps[key] = self._moved(self.steps.get(key) if key in self.timed_steps else None, value)
            self.timed_steps.add(key)

    def render(self, keys, default):
        return next((self.renders[key] for key in keys if key in self.renders), default)

    def timed(self, keys):
        """Whether this very render, settings and all, has been timed yet."""
        return keys[0] in self.renders

    def learn_render(self, keys, value):
        for key in keys:
            self.renders[key] = self._moved(self.renders.get(key), value)

    @staticmethod
    def _moved(old, value):
        return value if old is None else old + LEARNING_RATE * (value - old)

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"steps": {key: self.steps[key] for key in sorted(self.timed_steps)}, "renders": self.renders},
                          f, indent=2)
        except OSError:
            pass  # Next run just starts from older numbers

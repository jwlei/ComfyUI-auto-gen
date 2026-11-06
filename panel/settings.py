"""The few settings the gear in the header holds, kept in logs/settings.json so every device sees the same."""
import json
import os
import threading

from core.paths import LOGS_DIR

PATH = os.path.join(LOGS_DIR, "settings.json")
SIDES = ("shorts", "video", "image")
MAX_RESERVE_GB = 64
H3_MODELS = ("gemma-4-12b", "qwen3-14b")
DEFAULTS = {"low_priority": False, "reserve_ram_gb": 0, "h3_model": H3_MODELS[0],
            "default_workflow": {side: "" for side in SIDES}}
_lock = threading.Lock()


def _gigabytes(value):
    """Whole or half gigabytes, 0 (ComfyUI's own default) up to MAX_RESERVE_GB; anything else is 0."""
    try:
        gb = float(value)
    except (TypeError, ValueError):
        return 0
    return round(min(max(gb, 0), MAX_RESERVE_GB) * 2) / 2 if gb == gb else 0


def load():
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    workflows = saved.get("default_workflow") if isinstance(saved.get("default_workflow"), dict) else {}
    model = saved.get("h3_model", DEFAULTS["h3_model"])
    if model not in H3_MODELS:
        raise ValueError(f"Unknown OpenH3-IR model: {model!r}")
    return {"low_priority": bool(saved.get("low_priority")),
            "reserve_ram_gb": _gigabytes(saved.get("reserve_ram_gb")),
            "h3_model": model,
            "default_workflow": {side: str(workflows.get(side) or "") for side in SIDES}}


def update(changes):
    """Applies the settings in changes (the ones it knows, valid ones only) and returns them all."""
    with _lock:
        current = load()
        if "low_priority" in changes:
            current["low_priority"] = bool(changes["low_priority"])
        if "reserve_ram_gb" in changes:
            current["reserve_ram_gb"] = _gigabytes(changes["reserve_ram_gb"])
        if "h3_model" in changes:
            if changes["h3_model"] not in H3_MODELS:
                raise ValueError("Unknown OpenH3-IR model.")
            current["h3_model"] = changes["h3_model"]
        if isinstance(changes.get("default_workflow"), dict):
            for side in SIDES:
                if side in changes["default_workflow"]:
                    current["default_workflow"][side] = str(changes["default_workflow"][side] or "")
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return current

"""The few settings the gear in the header holds, kept in logs/settings.json so every device sees the same."""
import json
import os
import threading

from core.paths import LOGS_DIR

PATH = os.path.join(LOGS_DIR, "settings.json")
SIDES = ("shorts", "video", "image")
DEFAULTS = {"low_priority": False, "default_workflow": {side: "" for side in SIDES}}
_lock = threading.Lock()


def load():
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    workflows = saved.get("default_workflow") if isinstance(saved.get("default_workflow"), dict) else {}
    return {"low_priority": bool(saved.get("low_priority")),
            "default_workflow": {side: str(workflows.get(side) or "") for side in SIDES}}


def update(changes):
    """Applies the settings in changes (the ones it knows, valid ones only) and returns them all."""
    with _lock:
        current = load()
        if "low_priority" in changes:
            current["low_priority"] = bool(changes["low_priority"])
        if isinstance(changes.get("default_workflow"), dict):
            for side in SIDES:
                if side in changes["default_workflow"]:
                    current["default_workflow"][side] = str(changes["default_workflow"][side] or "")
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return current

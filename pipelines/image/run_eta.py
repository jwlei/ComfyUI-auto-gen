import json
import os
import re
import time

from pipelines.image.media import FROM_IMAGE, PIXELS_PER_MEGAPIXEL, parse_size
from pipelines.image.run_state import STAGES, RunState

# Used until a finished step has been timed. Each workflow learns its own image time: they differ by minutes.
DEFAULT_PROMPTS_SECONDS = 45
DEFAULT_IMAGE_SECONDS = 120
LEARNING_RATE = 0.3
SMALL_DRIFT_SECONDS = 60


def _option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


def _workflow_key(workflow):
    return f"image:{os.path.basename(workflow)}" if workflow else "image"


def _size_share(argv):
    """How much of one megapixel this is. Time follows the pixel count, near enough."""
    size = parse_size(_option(argv, "--size"))
    if size:
        return size[0] * size[1] / PIXELS_PER_MEGAPIXEL
    megapixels = _option(argv, "--megapixels", "1")
    return 1 if megapixels == FROM_IMAGE or not re.fullmatch(r"\d+(\.\d+)?", megapixels) else float(megapixels)


class RunEta:
    """As the Video side's: estimates what is left from how long each step took before, and learns as it goes."""

    def __init__(self, argv, env, path):
        self.path = path
        self.timings = self._load()
        self.plan_only = "--plan-only" in argv
        self.direct = "--direct" in argv
        self.workflow = (env or {}).get("COMFYUI_WORKFLOW")
        try:
            self.images = int(_option(argv, "--count", 1))
        except ValueError:
            self.images = 1
        self.size_share = _size_share(argv)
        self.kept = 0
        self.prompts_kept = False

        state = RunState.load(_option(argv, "--resume"))
        if state and state.shots:
            redo = _option(argv, "--redo")
            self.workflow = state.workflow
            self.images = len(state.shots)
            self.size_share = state.width * state.height / PIXELS_PER_MEGAPIXEL
            self.kept = len([shot for number, shot in enumerate(state.shots, 1) if shot.get("file") and str(number) != redo])
            self.prompts_kept = True

        self.number = self.total = 0
        self.name = None
        self.stage_started = None
        self.finished = {}
        self.image_started = None
        self.done_images = 0
        self.changes = []
        self.started = time.time()
        self.original_total = self.plan()["total"]

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return {key: value for key, value in saved.items() if isinstance(value, (int, float))}
        except (OSError, ValueError, AttributeError):
            return {}  # A broken file only costs accuracy

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.timings, f, indent=2)
        except OSError:
            pass  # Next run just starts from older numbers

    def _learn(self, key, default, value):
        old = self.timings.get(key, default)
        # The first real measurement beats a guess made on no machine in particular.
        self.timings[key] = value if key not in self.timings else old + LEARNING_RATE * (value - old)
        self._save()

    # --- What the run reports ---

    def stage(self, number, total, name):
        if self.number and number == self.number + 1:
            self._finish_stage()
        self.number, self.total, self.name = number, total, name
        self.stage_started = time.time()

    def image(self, number, total):
        self._finish_image(time.time())
        self.images = total
        self.image_started = time.time()

    def clip_retry(self, attempt, attempts, problem):
        """A render failed or was cut off: another starts now, unless that was the last try."""
        problem = problem[:80]
        if attempt < attempts:
            self.image_started = time.time()
            self.changes.append({"text": f"An image is rendering again ({problem})", "seconds": round(self._image_seconds())})
        else:
            self.image_started = None
            self.changes.append({"text": f"An image failed on all {attempts} tries ({problem})", "seconds": 0})

    def finish(self):
        if self.number and self.number == self.total:
            self._finish_stage()

    def _finish_image(self, now):
        if self.image_started is not None:
            self._learn(_workflow_key(self.workflow), DEFAULT_IMAGE_SECONDS, (now - self.image_started) / self.size_share)
            self.done_images += 1
            self.image_started = None

    def _finish_stage(self):
        now = time.time()
        self.finished[self.number] = now - self.stage_started
        if self.name == "Images":
            self._finish_image(now)
        elif not self.direct:
            self._learn("Prompts", DEFAULT_PROMPTS_SECONDS, now - self.stage_started)

    # --- Estimating ---

    def _image_seconds(self):
        return self.timings.get(_workflow_key(self.workflow), self.timings.get("image", DEFAULT_IMAGE_SECONDS)) * self.size_share

    def _stage_seconds(self, name):
        if name == "Prompts":
            return 0 if self.direct or self.prompts_kept else self.timings.get("Prompts", DEFAULT_PROMPTS_SECONDS)
        return self._image_seconds() * max(self.images - self.kept, 0)

    def plan(self):
        """What a run with these options should take, for the panel to show before it starts."""
        names = STAGES[:1] if self.plan_only else STAGES
        steps = [{"name": name, "seconds": round(self._stage_seconds(name))} for name in names]
        return {"steps": steps, "total": sum(step["seconds"] for step in steps), "images": self.images,
                "image_render": round(self._image_seconds())}

    def _current_left(self, now):
        if self.name == "Images":
            after = max(self.images - self.kept - self.done_images - (self.image_started is not None), 0) * self._image_seconds()
            this_one = max(self._image_seconds() - (now - self.image_started), 0) if self.image_started is not None else 0
            return this_one + after
        return max(self._stage_seconds(self.name) - (now - self.stage_started), 0)

    def estimate(self):
        """Returns seconds per step and left overall, or None before the first step."""
        if not self.number:
            return None
        now = time.time()
        steps = []
        left = 0
        for number, name in enumerate(STAGES[:self.total], 1):
            if name == "Prompts" and self.prompts_kept:
                steps.append({"state": "kept", "seconds": 0})
            elif number < self.number:
                steps.append({"state": "done", "seconds": round(self.finished.get(number, 0))})
            elif number == self.number:
                current = self._current_left(now)
                left += current
                steps.append({"state": "now", "seconds": round(current)})
            else:
                seconds = self._stage_seconds(name)
                left += seconds
                steps.append({"state": "next", "seconds": round(seconds)})
        elapsed = now - self.started
        adjusted = elapsed + left
        changes = list(self.changes)
        drift = adjusted - self.original_total - sum(change["seconds"] for change in changes)
        if abs(drift) >= SMALL_DRIFT_SECONDS:
            changes.append({"text": f"Running {'slower' if drift > 0 else 'faster'} than estimated", "seconds": round(drift)})
        return {"steps": steps, "total_left": round(left), "original_total": self.original_total,
                "original_left": round(self.original_total - elapsed), "adjusted_total": round(adjusted), "changes": changes}

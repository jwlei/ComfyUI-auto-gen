import re
import time

from core.eta_keys import SEPARATOR, Timings, options_in, render_keys
from pipelines.image.media import FROM_IMAGE, PIXELS_PER_MEGAPIXEL, parse_size
from pipelines.image.run_state import STAGES, RunState

# Used until a finished step has been timed. An image's time is learned per megapixel, under its model, workflow,
# options and size (core/eta_keys.py): they differ by minutes.
DEFAULT_PROMPTS_SECONDS = 45
DEFAULT_IMAGE_SECONDS = 120
SMALL_DRIFT_SECONDS = 60


def _option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


def _size_label(argv):
    """The size as chosen, for telling apart what's timed: exact, or as megapixels, or the image's own."""
    size = parse_size(_option(argv, "--size"))
    if size:
        return f"{size[0]}x{size[1]}"
    megapixels = _option(argv, "--megapixels", "1")
    return f"{megapixels} MP" if re.fullmatch(r"\d+(\.\d+)?", megapixels) else "as the image"


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
        self.timings = Timings(path, {"Prompts": DEFAULT_PROMPTS_SECONDS})
        self.plan_only = "--plan-only" in argv
        self.direct = "--direct" in argv
        self.llm = _option(argv, "--llm")
        self.options = options_in(argv)
        self.size = _size_label(argv)
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
            self.size = f"{state.width}x{state.height}"
            self.options = {**(state.options or {}), **self.options}
            self.kept = len([shot for number, shot in enumerate(state.shots, 1) if shot.get("file") and str(number) != redo])
            self.prompts_kept = True

        self.render_keys = render_keys(self.workflow, self.options, self.size, fallback="image")
        self.number = self.total = 0
        self.name = None
        self.stage_started = None
        self.finished = {}
        self.image_started = None
        self.done_images = 0
        self.changes = []
        self.started = time.time()
        self.original_total = self.plan()["total"]

    def _prompt_keys(self):
        """Writing the prompts is timed per LLM and per number of images, then per LLM, then at all."""
        llm = self.llm or "default"
        return [SEPARATOR.join(("Prompts", llm, f"{self.images} images")), SEPARATOR.join(("Prompts", llm)), "Prompts"]

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
            self.timings.learn_render(self.render_keys, (now - self.image_started) / self.size_share)
            self.timings.save()
            self.done_images += 1
            self.image_started = None

    def _finish_stage(self):
        now = time.time()
        self.finished[self.number] = now - self.stage_started
        if self.name == "Images":
            self._finish_image(now)
        elif not self.direct:
            self.timings.learn_step(self._prompt_keys(), now - self.stage_started)
            self.timings.save()

    # --- Estimating ---

    def _image_seconds(self):
        return self.timings.render(self.render_keys, DEFAULT_IMAGE_SECONDS) * self.size_share

    def _stage_seconds(self, name):
        if name == "Prompts":
            return 0 if self.direct or self.prompts_kept else self.timings.step(self._prompt_keys())
        return self._image_seconds() * max(self.images - self.kept, 0)

    def plan(self):
        """What a run with these options should take, for the panel to show before it starts."""
        names = STAGES[:1] if self.plan_only else STAGES
        steps = [{"name": name, "seconds": round(self._stage_seconds(name))} for name in names]
        return {"steps": steps, "total": sum(step["seconds"] for step in steps), "images": self.images,
                "image_render": round(self._image_seconds()), "timed": self.timings.timed(self.render_keys)}

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

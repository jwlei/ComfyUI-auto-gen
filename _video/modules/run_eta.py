import json
import os
import re
import time

from modules.media import RENDER_AREA
from modules.planner import AVERAGE_SHOT_SECONDS, shot_count
from modules.run_state import STAGES, RunState

# Used until a finished step has been timed.
DEFAULT_TIMINGS = {
    "Plan": 60,
    "render": 31,  # Render time per second of film at full size, measured on this GPU in yt-auto-shorts
    "part": 6,  # One clip evened out for stitching
    "Upload": 120,
}
# How much each new measurement moves the stored value.
LEARNING_RATE = 0.3
# Unexplained drift from the original estimate below this isn't worth a line in the panel.
SMALL_DRIFT_SECONDS = 60


def _load_timings(path):
    timings = dict(DEFAULT_TIMINGS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        timings.update({key: value for key, value in saved.items()
                        if key in timings and isinstance(value, (int, float))})
    except (OSError, ValueError, AttributeError):
        pass  # A broken file only costs accuracy
    return timings


def _option(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


def _size_share(argv):
    """How much of a full-size render this size is. Render time follows the pixel count, near enough."""
    match = re.fullmatch(r"(\d+)x(\d+)", str(_option(argv, "--size", "")))
    return int(match.group(1)) * int(match.group(2)) / RENDER_AREA if match else 1


class RunEta:
    """
    Estimates what is left of a run from how long each step took before. Steps are
    timed as they finish and saved to `path`, so estimates improve with every run.
    A step cut short by a failure or Stop is never counted. With --resume, only
    what that run still has to do counts.
    """

    def __init__(self, argv, path):
        self.path = path
        self.timings = _load_timings(path)
        self.plan_only = "--plan-only" in argv
        self.upload = "--upload" in argv
        try:
            self.film_seconds = int(_option(argv, "--duration", 0))
        except ValueError:
            self.film_seconds = 0
        self.shots = shot_count(self.film_seconds)
        self.size_share = _size_share(argv)
        self.kept_stages = ()
        self.kept_seconds = 0

        state = RunState.load(_option(argv, "--resume", None))
        if state and state.shots:
            redo = _option(argv, "--redo", None)
            if redo:
                state.redo(int(redo) - 1)
            self.upload = bool(state.upload)
            self.film_seconds = sum(shot["seconds"] for shot in state.shots)
            self.shots = len(state.shots)
            self.size_share = state.width * state.height / RENDER_AREA
            self.kept_seconds = state.kept_seconds()
            self.kept_stages = ("Plan",)

        self.number = 0
        self.total = 0
        self.name = None
        self.stage_started = None
        self.finished = {}
        self.clip_number = 0
        self.done_seconds = 0
        self.attempt_started = None
        self.attempt_seconds = 0
        self.changes = []
        self.started = time.time()
        # Stays as it was at the start, however the run goes, to compare the live ETA with.
        self.original_total = self.plan()["total"]

    @property
    def names(self):
        if self.plan_only:
            return STAGES[:1]
        return STAGES if self.upload else STAGES[:-1]

    # --- What the run reports ---

    def stage(self, number, total, name):
        if self.number and number == self.number + 1:
            self._finish_stage()
        self.number, self.total, self.name = number, total, name
        self.stage_started = time.time()

    def clip(self, number, total, seconds):
        now = time.time()
        self._finish_attempt(now)
        self.clip_number, self.shots = number, total
        self.attempt_started = now
        self.attempt_seconds = seconds

    def clip_retry(self, attempt, attempts, problem):
        """A render failed or was cut off: another starts now, unless that was the last try."""
        clip = f"Clip {self.clip_number} of {self.shots}"
        problem = problem[:80]
        if attempt < attempts:
            self.attempt_started = time.time()
            self._note(f"{clip} rendering again ({problem})", self._render_seconds(self.attempt_seconds))
        else:
            self.attempt_started = None
            self._note(f"{clip} failed on all {attempts} tries ({problem})", 0)

    def finish(self):
        """The whole run succeeded, so the step it ended on did finish."""
        if self.number and self.number == self.total:
            self._finish_stage()

    # --- Why the ETA moved ---

    def _note(self, text, seconds):
        self.changes.append({"text": text, "seconds": round(seconds)})

    def _all_changes(self, adjusted):
        """The noted changes, plus whatever they don't explain, so they add up to the adjusted ETA."""
        changes = list(self.changes)
        drift = adjusted - self.original_total - sum(change["seconds"] for change in changes)
        if abs(drift) >= SMALL_DRIFT_SECONDS:
            pace = "slower" if drift > 0 else "faster"
            changes.append({"text": f"Running {pace} than estimated", "seconds": round(drift)})
        return changes

    # --- Learning ---

    def _learn(self, key, value):
        self.timings[key] += LEARNING_RATE * (value - self.timings[key])

    def _finish_attempt(self, now):
        if self.attempt_started is not None and self.attempt_seconds:
            # As if at full size, or a 360p film would make the next 720p estimate far too short.
            self._learn("render", (now - self.attempt_started) / self.attempt_seconds / self.size_share)
            self.done_seconds += self.attempt_seconds
            self.attempt_started = None
            self._save()

    def _finish_stage(self):
        now = time.time()
        self.finished[self.number] = now - self.stage_started
        if self.name == "Clips":
            self._finish_attempt(now)
        elif self.name == "Stitching":
            self._learn("part", (now - self.stage_started) / max(1, self.shots))
        else:
            self._learn(self.name, now - self.stage_started)
        self._save()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.timings, f, indent=2)
        except OSError:
            pass  # Next run just starts from older numbers

    # --- Estimating ---

    def _render_seconds(self, film_seconds):
        return self.timings["render"] * self.size_share * film_seconds

    def _stage_seconds(self, name):
        """What a step takes from the start, without the part a continued run kept."""
        if name in self.kept_stages:
            return 0
        if name == "Clips":
            return self._render_seconds(max(self.film_seconds - self.kept_seconds, 0))
        if name == "Stitching":
            return self.shots * self.timings["part"]
        return self.timings[name]

    def _current_left(self, now):
        if self.name == "Clips" and self.attempt_started is not None:
            this_clip = self._render_seconds(self.attempt_seconds) - (now - self.attempt_started)
            after = self.film_seconds - self.kept_seconds - self.done_seconds - self.attempt_seconds
            return max(this_clip, 0) + self._render_seconds(max(after, 0))
        return max(self._stage_seconds(self.name) - (now - self.stage_started), 0)

    def plan(self):
        """What a run with these options should take, for the panel to show before it starts."""
        steps = [{"name": name, "seconds": round(self._stage_seconds(name))} for name in self.names]
        return {"steps": steps, "total": sum(step["seconds"] for step in steps),
                "clips": self.shots,
                "clip_seconds": round(self.film_seconds / self.shots) if self.film_seconds else AVERAGE_SHOT_SECONDS,
                "clip_render": round(self._render_seconds(self.film_seconds / self.shots)) if self.film_seconds else 0}

    def estimate(self):
        """Returns seconds per step and left overall, or None before the first step (or for a run without steps)."""
        if not self.number:
            return None
        now = time.time()
        steps = []
        left = 0
        for number, name in enumerate(STAGES[:self.total], 1):
            if name in self.kept_stages:
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
        return {"steps": steps, "total_left": round(left),
                "original_total": self.original_total, "original_left": round(self.original_total - elapsed),
                "adjusted_total": round(adjusted), "changes": self._all_changes(adjusted)}

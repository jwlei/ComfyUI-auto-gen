import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime

from pipelines.image import paths

STAGES = ("Prompts", "Images")
RUN_ID = re.compile(r"image_\d{8}_\d{6}")
SAVE_ATTEMPTS = 5


@dataclass
class RunState:
    """
    What one set of images has finished, saved after the prompts and after every accepted
    image, so a run that was stopped, failed or cut off can continue without making them again.
    The images go straight to the set's folder in output/, where you can see them as they come.
    """

    id: str
    brief: str
    count: int
    direct: bool  # The brief is the prompt, word for word: no LLM wrote anything
    width: int
    height: int
    llm: str = None
    title: str = None
    shots: list = None  # [{number, title, prompt, file}]. file is set once its image is accepted.
    workflow: str = None
    image_count: int = 0
    inputs: str = None  # Fingerprint of the images it was planned from, for Generate to recognise them
    options: dict = None  # The workflow's options as chosen, such as {"group_2nd_pass": "0"} (core/workflow_options.py)

    @classmethod
    def new(cls, brief, count, direct, width, height, llm=None, workflow=None, image_count=0):
        return cls(id=f"image_{datetime.now():%Y%m%d_%H%M%S}", brief=brief, count=count, direct=direct, width=width, height=height,
                   llm=llm, workflow=workflow, image_count=image_count)

    @classmethod
    def load(cls, run_id):
        """The saved run, with anything whose file has gone marked as still to do. None if there isn't one."""
        if not RUN_ID.fullmatch(str(run_id)):
            return None
        try:
            with open(os.path.join(paths.RUNS_DIR, run_id, "state.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            state = cls(**{name: value for name, value in saved.items() if name in cls.__dataclass_fields__})
            for shot in state.shots or []:
                if shot.get("file") and not os.path.exists(state.image_path(shot)):
                    shot["file"] = None
        except (OSError, ValueError, TypeError, AttributeError):
            return None  # Nothing saved, or a broken file: nothing to continue
        return state

    # --- Where its files go ---

    @property
    def folder(self):
        return os.path.join(paths.RUNS_DIR, self.id)

    @property
    def path(self):
        return os.path.join(self.folder, "state.json")

    @property
    def input_paths(self):
        return [os.path.join(self.folder, "input.png" if number == 1 else f"input_{number}.png")
                for number in range(1, self.image_count + 1)]

    @property
    def fitted_path(self):
        return os.path.join(self.folder, f"input_{self.width}x{self.height}.png")

    @property
    def output_dir(self):
        name = re.sub(r"[^\w-]+", "_", self.title or "").strip("_")[:60] or "images"
        return os.path.join(paths.OUTPUT_DIR, f"{name}_{self.id}")

    def image_path(self, shot):
        return os.path.join(self.output_dir, shot["file"])

    def thumbnail_path(self, shot):
        return os.path.join(self.folder, "thumbnails", f"{os.path.splitext(shot['file'])[0]}.jpg")

    # --- Saving ---

    def update(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)
        self.save()

    def save(self):
        temp = self.path + ".tmp"
        os.makedirs(self.folder, exist_ok=True)
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=1)
        for attempt in range(SAVE_ATTEMPTS):
            try:
                os.replace(temp, self.path)
                return
            except PermissionError:  # The panel has it open for a moment
                if attempt == SAVE_ATTEMPTS - 1:
                    raise
                time.sleep(0.2)

    def remove(self):
        """State first: without it the run is gone from the list, even if a file refuses to be deleted."""
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        shutil.rmtree(self.folder, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    # --- What is left ---

    @property
    def done(self):
        return [shot for shot in self.shots or [] if shot.get("file")]

    @property
    def finished(self):
        return bool(self.shots) and len(self.done) == len(self.shots)


def list_runs():
    """Every saved run, newest first."""
    try:
        names = os.listdir(paths.RUNS_DIR)
    except OSError:
        return []
    runs = (RunState.load(name) for name in sorted(names, reverse=True))
    return [run for run in runs if run]

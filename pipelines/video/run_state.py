import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from pipelines.video import paths
from pipelines.video.media import on_grid

STAGES = ("Plan", "Clips", "Stitching")
RUN_ID = re.compile(r"run_\d{8}_\d{6}")
SAVE_ATTEMPTS = 5


@dataclass
class RunState:
    """
    What one film's run has finished so far, saved after the plan and after every
    accepted clip, so a run that was stopped, failed or cut off by a crash can
    continue without rendering that work again.

    A clip counts once it passed its check: a file on disk may be a rejected render.
    """

    id: str
    brief: str
    seconds: int  # How long the film should be
    start: str  # 'chain': each clip starts on the last frame of the one before. 'image': on the input image.
    width: int
    height: int
    llm: str = "gemini"  # Who wrote the plan, and writes the upload's title and description
    title: str = None
    shots: list = None  # [{title, seconds, prompt}]. Edit the prompts here before rendering, if you like.
    clips: dict = field(default_factory=dict)  # {shot index: file name in clips/}
    final: str = None
    workflow: str = None  # The workflow file it renders with, if not the built-in one
    image_count: int = 1  # 0 for a film from text alone, several for a workflow that takes references
    inputs: str = None  # Fingerprint of the images and audio it was planned from, for Generate to recognise them
    has_audio: bool = False  # The film has an audio track of its own, in the run's folder

    @classmethod
    def new(cls, brief, seconds, start, width, height, llm="gemini", workflow=None, image_count=1):
        return cls(id=f"run_{datetime.now():%Y%m%d_%H%M%S}", brief=brief, seconds=seconds, start=start,
                   width=width, height=height, llm=llm, workflow=workflow, image_count=image_count)

    @classmethod
    def load(cls, run_id):
        """The saved run, with anything whose files have gone marked as still to do. None if there isn't one."""
        if not RUN_ID.fullmatch(str(run_id)):
            return None
        try:
            with open(os.path.join(paths.RUNS_DIR, run_id, "state.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Runs made before uploads moved to Shorts say where to upload to.
            state = cls(**{name: value for name, value in saved.items() if name in cls.__dataclass_fields__})
            state.clips = {int(index): name for index, name in state.clips.items()}
            state._forget_missing()
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
    def image_path(self):
        return os.path.join(self.folder, "input.png")

    @property
    def image_paths(self):
        return [self.image_path if number == 1 else os.path.join(self.folder, f"input_{number}.png")
                for number in range(1, self.image_count + 1)]

    @property
    def audio_path(self):
        return os.path.join(self.folder, "audio.wav")

    @property
    def frame_path(self):
        return os.path.join(self.folder, "last_frame.png")

    @property
    def clips_dir(self):
        return os.path.join(self.folder, "clips")

    @property
    def parts_dir(self):
        return os.path.join(self.folder, "parts")

    def clip_path(self, index):
        return os.path.join(self.clips_dir, self.clips[index])

    # --- Saving ---

    def update(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)
        self.save()

    def adopt(self, flags):
        """Takes the options chosen now, for Generate to continue this run. The plan stays: it's what was paid for."""
        self.llm = next((flags[i + 1] for i, flag in enumerate(flags[:-1]) if flag == "--llm"), self.llm)
        size = next((flags[i + 1] for i, flag in enumerate(flags[:-1]) if flag == "--size"), None)
        if size:
            width, height = (on_grid(int(side)) for side in size.split("x"))
            if (width, height) != (self.width, self.height):
                self.width, self.height, self.clips, self.final = width, height, {}, None
        self.save()

    def keep_clip(self, index, name):
        self.clips[index] = name
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

    # --- What is left ---

    def redo(self, index):
        """A chained clip starts on the last frame of the one before, so everything after it has to go too."""
        if self.start == "chain":
            self.clips = {i: name for i, name in self.clips.items() if i < index}
        else:
            self.clips.pop(index, None)
        self.final = None

    def kept_seconds(self):
        return sum(shot["seconds"] for i, shot in enumerate(self.shots or []) if i in self.clips)

    def next_stage(self):
        if len(self.clips) < len(self.shots or []):
            return "Clips"
        return "Stitching"

    def _forget_missing(self):
        for index in [i for i in sorted(self.clips) if not os.path.exists(self.clip_path(i))]:
            self.redo(index)
        if self.final and not os.path.exists(self.final):
            self.final = None


def run_id_in(name):
    """Which run a film belongs to, from its file name. None if it doesn't say."""
    match = RUN_ID.search(name)
    return match.group(0) if match else None


def list_runs():
    """Every saved run, newest first."""
    try:
        names = os.listdir(paths.RUNS_DIR)
    except OSError:
        return []
    runs = (RunState.load(name) for name in sorted(names, reverse=True))
    return [run for run in runs if run]


def folder_size(path):
    return sum(os.path.getsize(os.path.join(folder, name))
               for folder, _, names in os.walk(path) for name in names
               if os.path.exists(os.path.join(folder, name)))

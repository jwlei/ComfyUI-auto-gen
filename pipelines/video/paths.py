import os

from core.paths import DATA_ROOT, PROJECT_ROOT  # noqa: F401

RUNS_DIR = os.path.join(DATA_ROOT, "runs", "video")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", "video")
# Images and audio the panel received, until the run they were queued for has made its own copy.
INPUTS_DIR = os.path.join(DATA_ROOT, "inputs")

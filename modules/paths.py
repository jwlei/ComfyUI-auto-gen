import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Moved only by the tests, so a real pipeline process can run against a throwaway folder.
DATA_ROOT = os.getenv("AUTOCOMFYUI_DATA") or PROJECT_ROOT
RUNS_DIR = os.path.join(DATA_ROOT, "runs")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output")
UPLOADED_DIR = os.path.join(DATA_ROOT, "uploaded")
# Images the panel received, until the run they were queued for has made its own copy.
INPUTS_DIR = os.path.join(DATA_ROOT, "inputs")
LOGS_DIR = os.path.join(DATA_ROOT, "logs")


def move_to_uploaded(path):
    """A file of the same name already there is replaced: re-uploading a video shouldn't fail on it."""
    os.makedirs(UPLOADED_DIR, exist_ok=True)
    target = os.path.join(UPLOADED_DIR, os.path.basename(path))
    shutil.move(path, target)
    return target

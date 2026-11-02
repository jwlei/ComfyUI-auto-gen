import os
import shutil

from core.paths import DATA_ROOT, PROJECT_ROOT

# Finished videos land here; once uploaded they are moved to UPLOADED_DIR, so
# OUTPUT_DIR only ever holds videos still waiting to go up.
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", "shorts", "ready")
UPLOADED_DIR = os.path.join(DATA_ROOT, "output", "shorts", "uploaded")
RUNS_DIR = os.path.join(DATA_ROOT, "runs", "shorts")


def thumbnail_path(video_path):
    """Where a video's thumbnail lives: beside it, with the same name."""
    return os.path.splitext(video_path)[0] + ".jpg"


def avatar_still_path(video_path):
    """Where a video waiting for upload keeps the still of the avatar its thumbnail is made from."""
    return os.path.splitext(video_path)[0] + ".still.jpg"


def move_to_uploaded(video_path):
    """Moves an uploaded video into UPLOADED_DIR. Returns the new path (or the old one if the move failed)."""
    os.makedirs(UPLOADED_DIR, exist_ok=True)
    target = os.path.join(UPLOADED_DIR, os.path.basename(video_path))
    try:
        shutil.move(video_path, target)
        print(f"   📁 Moved to {os.path.relpath(target, PROJECT_ROOT)}")
        return target
    except OSError as e:
        # The upload itself succeeded; a file left behind is only untidy.
        print(f"   ⚠️ Uploaded, but could not move {os.path.basename(video_path)}: {e}")
        return video_path

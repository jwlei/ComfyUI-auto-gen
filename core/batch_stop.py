"""
Stop after this one: the panel asks a batch (Shorts' videos, Image's images) to end once what it's making now is done,
rather than killing it part way. A file the panel writes and the pipeline looks for between one and the next. The
panel clears it when a job starts, so a request only ever counts for the job it was made for.
"""
import os

from core.paths import LOGS_DIR

PATH = os.path.join(LOGS_DIR, "stop_after_current")


def request():
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write("1")


def clear():
    try:
        os.remove(PATH)
    except FileNotFoundError:
        pass


def requested():
    return os.path.exists(PATH)

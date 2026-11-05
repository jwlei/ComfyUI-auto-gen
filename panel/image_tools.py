"""
The Image side's tools that work on one picture while you look at it, on the CPU (pipelines/image/segment.py), so
the GPU stays with ComfyUI:

    Select     a click on a thing gives its mask, for painting the areas to change
"""
import io
import threading
from collections import OrderedDict

from flask import request, send_file

from panel.server import app, password_required
from pipelines.image import segment
from pipelines.image.run_state import RunState

# Images sent for Select, by key, kept for the clicks that follow. The most recent last.
KEPT_IMAGES = 6
_images = OrderedDict()
_images_lock = threading.Lock()
MOST_POINTS = 24


def keep_image(data):
    key = segment.image_key(data)
    with _images_lock:
        _images[key] = data
        _images.move_to_end(key)
        while len(_images) > KEPT_IMAGES:
            _images.popitem(last=False)
    return key


def kept_image(key):
    with _images_lock:
        return _images.get(key)


def image_bytes(form, files):
    """The picture a tool works on: a file sent with the request, or one in a set (set and position). None if neither."""
    upload = files.get("image")
    if upload and upload.filename:
        return upload.read()
    state = RunState.load(form.get("set") or "")
    try:
        shot = state.shots[int(form.get("position") or 0) - 1] if state and state.shots else None
    except (ValueError, IndexError):
        shot = None
    if not shot or not shot.get("file"):
        return None
    with open(state.image_path(shot), "rb") as f:
        return f.read()


@app.route("/api/select/image", methods=["POST"])
@password_required
def select_image():
    """Takes the picture Select will be used on, and says the key its clicks name it by."""
    data = image_bytes(request.form, request.files)
    if not data:
        return {"ok": False, "error": "No picture to select in."}, 400
    return {"ok": True, "key": keep_image(data)}


@app.route("/api/select", methods=["POST"])
@password_required
def select():
    """The mask of what was clicked, as a PNG the size of the picture: white inside."""
    body = request.get_json(silent=True) or {}
    data = kept_image(str(body.get("key") or ""))
    if not data:
        return {"ok": False, "error": "The picture is no longer here: open it again."}, 410
    try:
        points = [(float(x), float(y), bool(inside)) for x, y, inside in body.get("points") or []][:MOST_POINTS]
    except (TypeError, ValueError):
        return {"ok": False, "error": "Those aren't points on the picture."}, 400
    if not points:
        return {"ok": False, "error": "Click on what to select."}, 400
    size = body.get("size")
    try:
        mask = segment.select(data, points, size if size in (0, 1, 2) else None)
    except segment.Missing as e:
        return {"ok": False, "error": str(e)}, 503
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


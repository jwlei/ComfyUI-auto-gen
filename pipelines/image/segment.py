"""
Finding things in an image without the GPU, which stays with ComfyUI: on the CPU with ONNX Runtime, as Kokon does.

    select()   MobileSAM: a click on a thing gives its mask. The image is read once (about a second), and every
               click after that on the same image takes a fraction of one.

The model files are in models/, which scripts/get_models.py downloads. They aren't loaded until Select is used.
"""
import hashlib
import io
import os
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

from core.paths import PROJECT_ROOT

MODELS = os.path.join(PROJECT_ROOT, "models")
SAM_ENCODER = os.path.join(MODELS, "mobilesam", "mobile_sam_image_encoder.onnx")
SAM_DECODER = os.path.join(MODELS, "mobilesam", "sam_mask_decoder_multi.onnx")
# MobileSAM reads an image at 1024 on its longer side.
SAM_SIDE = 1024
# Images read by MobileSAM kept for the clicks that follow, most recent last.
KEPT_EMBEDDINGS = 4

_sessions = {}
_lock = threading.Lock()
_embeddings = OrderedDict()


class Missing(RuntimeError):
    """A model file isn't there."""


def _session(path):
    with _lock:
        if path not in _sessions:
            if not os.path.exists(path):
                raise Missing(f"{os.path.relpath(path, PROJECT_ROOT)} is missing: run python scripts/get_models.py.")
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            _sessions[path] = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        return _sessions[path]


def image_key(data):
    return hashlib.sha256(data).hexdigest()


def _embedding(image, key):
    """MobileSAM's reading of the image, and the scale it read it at, kept for the next clicks on it."""
    with _lock:
        if key in _embeddings:
            _embeddings.move_to_end(key)
            return _embeddings[key]
    scale = SAM_SIDE / max(image.size)
    resized = image.convert("RGB").resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    embedding = _session(SAM_ENCODER).run(None, {"input_image": pixels})[0]
    with _lock:
        _embeddings[key] = (embedding, scale)
        while len(_embeddings) > KEPT_EMBEDDINGS:
            _embeddings.popitem(last=False)
    return embedding, scale


def select(data, points, size_choice=None):
    """
    The mask of what was clicked, as an L image the size of the image (255 inside). data: the image file's bytes.
    points: [(x, y, inside)] in the image's pixels; a point outside (inside False) takes that part away. size_choice:
    0-2 for the smaller, middle or larger of SAM's three guesses at what one click means; None for its best.
    """
    image = Image.open(io.BytesIO(data))
    image.load()
    embedding, scale = _embedding(image, image_key(data))
    coords = [[x * scale, y * scale] for x, y, _ in points] + [[0.0, 0.0]]
    labels = [1.0 if inside else 0.0 for _, _, inside in points] + [-1.0]
    masks, scores, _ = _session(SAM_DECODER).run(None, {
        "image_embeddings": embedding,
        "point_coords": np.array([coords], dtype=np.float32),
        "point_labels": np.array([labels], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([image.height, image.width], dtype=np.float32),
    })
    # One click is ambiguous (a button, the shirt, the person): SAM's 2-4 are its three readings, small to large.
    # More points say enough for its single answer, the first.
    if len(points) == 1:
        index = 1 + (size_choice if size_choice in (0, 1, 2) else int(np.argmax(scores[0, 1:4])))
    else:
        index = 0
    return Image.fromarray(((masks[0, index] > 0) * 255).astype(np.uint8))


"""
Which model a LoRA file was made for, read from its header alone. Qwen-Image 2.1 changed its layers: a LoRA made for
an earlier Qwen-Image (1.x, 2512 or Edit) loads without an error and does nothing (Kokon refuses them for that reason,
github.com/GabrieleLozupone/kokon crates/qls-core/src/lora.rs). Their feed-forward layers tell them apart:

    Qwen-Image 2.1        transformer_blocks.N.img_mlp.gate_up, or gate_layer and proj
    earlier Qwen-Image    transformer_blocks.N.img_mlp.net.0.proj and net.2

ComfyUI's folder is found from COMFYUI_DIR, else from COMFYUI_LAUNCHER (the portable build keeps it in ComfyUI/ beside
its .bat). A file that can't be found or read gives no answer, and nothing is refused for it.
"""
import json
import os
import struct

QWEN_21, QWEN_EARLIER = "qwen_image_21", "qwen_image_earlier"
MOST_HEADER = 64 * 1024 * 1024
_found = {}


def comfyui_folders():
    """Where ComfyUI may be: COMFYUI_DIR, and the folders around its launcher."""
    folders = []
    if os.getenv("COMFYUI_DIR"):
        folders.append(os.getenv("COMFYUI_DIR"))
    launcher = os.getenv("COMFYUI_LAUNCHER") or ""
    if launcher:
        here = os.path.dirname(os.path.abspath(launcher))
        folders += [os.path.join(here, "ComfyUI"), here]
    return [folder for folder in folders if os.path.isdir(os.path.join(folder, "models"))]


def lora_path(name):
    """The LoRA file ComfyUI lists by this name (as its Load LoRA node does, relative to models/loras), or None."""
    for folder in comfyui_folders():
        path = os.path.join(folder, "models", "loras", name.replace("\\", os.sep).replace("/", os.sep))
        if os.path.isfile(path):
            return path
    return None


def tensor_names(path):
    """The names of a .safetensors file's tensors, from its header. Empty if it isn't one."""
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size <= MOST_HEADER:
                return []
            header = json.loads(f.read(size))
    except (OSError, ValueError, struct.error):
        return []
    return [name for name in header if name != "__metadata__"]


def family(name):
    """Which Qwen-Image a LoRA by this name was made for: QWEN_21, QWEN_EARLIER, or None if it isn't telling."""
    path = lora_path(name)
    if not path:
        return None
    key = (path, os.path.getmtime(path))
    if key not in _found:
        names = tensor_names(path)
        if any(".img_mlp.gate_up" in tensor or ".img_mlp.gate_layer" in tensor for tensor in names):
            _found[key] = QWEN_21
        elif any(".img_mlp.net." in tensor for tensor in names):
            _found[key] = QWEN_EARLIER
        else:
            _found[key] = None
    return _found[key]


def problem(name, model):
    """Why this LoRA does nothing on this model, or None."""
    if model and model.startswith("qwen_image_21") and family(name) == QWEN_EARLIER:
        return "It was made for an earlier Qwen-Image (1.x, 2512 or Edit): on Qwen-Image 2.1 it does nothing."
    return None

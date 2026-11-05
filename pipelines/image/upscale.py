"""
Makes one image of a set bigger, as a new image in the set beside the one it came from: sharper with SeedVR2, in
minutes, or in seconds with Superscale-SP (--method fast), which adds no new content.

    python -m pipelines.image.upscale --resume image_20261104_120000 --image 3

The settings are those the Krea 2 SFW/NSFW workflow runs SeedVR2 with on this 12 GB card: the 7B model, swapping
36 blocks to the CPU, tiled VAE. Attention is sdpa, as everywhere here (sageattn isn't installed).
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from core.paths import comfyui_client_id
from core.saved_workflows import unique_name
from pipelines.image.comfy import IMAGE_EXTENSIONS
from pipelines.image.run_state import RunState
from pipelines.video.comfy import DEFAULT_COMFYUI_URL, DEFAULT_TIMEOUT, ComfyClient

# Twice the shorter side, up to what the card holds at once with its tiled VAE.
FACTOR = 2
MOST_EDGE = 2560
DIT_MODEL = "seedvr2_ema_7b_fp16.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
SUFFIX = "_upscaled"


class UpscaleClient(ComfyClient):
    """ComfyClient's queueing, waiting and downloading, for a workflow made here rather than one saved in ComfyUI."""

    def __init__(self):
        self.url = (os.getenv("COMFYUI_URL") or DEFAULT_COMFYUI_URL).rstrip("/")
        self.timeout = int(os.getenv("COMFYUI_TIMEOUT") or DEFAULT_TIMEOUT)
        self.client_id = comfyui_client_id()


def upscale_workflow(image, resolution):
    """SeedVR2 on one image, its shorter side to resolution, into ComfyUI's temp folder (PreviewImage), not its output."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": DIT_MODEL, "device": "cuda:0", "blocks_to_swap": 36, "swap_io_components": False,
            "offload_device": "cpu", "cache_model": False, "attention_mode": "sdpa"}},
        "3": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": VAE_MODEL, "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
            "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128, "tile_debug": "false",
            "offload_device": "cpu", "cache_model": False}},
        "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": 777, "resolution": resolution, "max_resolution": 0,
            "batch_size": 1, "uniform_batch_size": False, "color_correction": "lab", "temporal_overlap": 0, "prepend_frames": 0,
            "input_noise_scale": 0.0, "latent_noise_scale": 0.0, "offload_device": "cpu", "enable_debug": False}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
    }
    # Titled as ComfyUI's own exports are, so its queue and history say what each node is.
    for node in workflow.values():
        node["_meta"] = {"title": node["class_type"]}
    return workflow


# The fast way, in seconds: the Superscale-SP model (ESRGAN, 4x) Kokon settled on, then down to the size wanted. Strength
# mixes it with a plain resize, to tone down texture the model invents (Kokon's upscaler strength).
FAST_MODEL = "4x_NMKD-Superscale-SP_178000_G.pth"
FAST_MOST_EDGE = 8192
METHODS = ("seedvr2", "fast")


def fast_workflow(image):
    """Superscale-SP on one image, 4x, into ComfyUI's temp folder (PreviewImage)."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": FAST_MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    }
    for node in workflow.values():
        node["_meta"] = {"title": node["class_type"]}
    return workflow


def fast_size(width, height):
    """Twice the size, its longer side held to what an image here can be."""
    scale = min(FACTOR, FAST_MOST_EDGE / max(width, height))
    return round(width * scale), round(height * scale)


def blended(made, source, size, strength):
    """The model's image at size, mixed with the source plainly resized: strength 1 is the model's alone."""
    from PIL import Image
    sharp = made.convert("RGB").resize(size, Image.LANCZOS)
    if strength >= 1:
        return sharp
    plain = source.convert("RGB").resize(size, Image.LANCZOS)
    return Image.blend(plain, sharp, max(0.0, strength))


def target_resolution(width, height):
    """The shorter side it's made at: twice what it is, held to what fits."""
    return min(MOST_EDGE, FACTOR * min(width, height))


def upscale(state, position, comfy, method="seedvr2", strength=1.0, resolution=0):
    """Makes the image at position again, bigger, and puts it in the set after it. Returns the new file's path."""
    from PIL import Image
    shot = state.shots[position - 1]
    source = state.image_path(shot)
    with Image.open(source) as image:
        width, height = image.size
    uploaded = comfy.upload(source, f"autocomfyui_{state.id}_upscale_{position}.png")
    stem = unique_name(state.output_dir, os.path.splitext(shot["file"])[0] + SUFFIX, ".png")
    if method == "fast":
        size = fast_size(width, height)
        print(f"🔍 Upscaling image {position}, {width}x{height}, to {size[0]}x{size[1]} with Superscale-SP, strength {strength:g}...")
        entry = comfy._run(fast_workflow(uploaded), 777)
        path = comfy._download(entry, os.path.join(state.output_dir, stem), IMAGE_EXTENSIONS, "an image")
        with Image.open(path) as made, Image.open(source) as original:
            finished = blended(made, original, size, strength)
        part = path + ".part"
        finished.save(part, format="PNG")
        os.replace(part, path)
    else:
        resolution = resolution or target_resolution(width, height)
        print(f"🔍 Upscaling image {position}, {width}x{height}, to {resolution} on its shorter side...")
        entry = comfy._run(upscale_workflow(uploaded, resolution), 777)
        path = comfy._download(entry, os.path.join(state.output_dir, stem), IMAGE_EXTENSIONS, "an image")
    how = "upscaled" if method == "seedvr2" else "upscaled fast"
    made = {**{key: value for key, value in shot.items() if key not in ("redo",)},
            "title": f"{shot['title']} · {how}", "file": os.path.basename(path), "upscaled_from": position}
    state.shots.insert(position, made)
    state.save()
    with Image.open(path) as image:
        print(f"✅ Upscaled to {image.size[0]}x{image.size[1]}: {path}")
    return path


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--resume", metavar="RUN", required=True, help="The set the image is in.")
    parser.add_argument("--image", type=int, required=True, metavar="N", help="Its place in the set, from 1.")
    parser.add_argument("--method", choices=METHODS, default="seedvr2",
                        help="seedvr2: redraws the detail, minutes (the default). fast: Superscale-SP, seconds, adds no new content.")
    parser.add_argument("--strength", type=float, default=1.0, help="With fast: how much of the model's result, 0-1, the rest a plain resize.")
    parser.add_argument("--resolution", type=int, default=0, metavar="PX",
                        help="With seedvr2: the shorter side to make it, as the Krea 2 workflow's 720p/1080p/1440p/4K. 0: twice what it is.")
    args = parser.parse_args(argv)
    state = RunState.load(args.resume)
    if not state or not 1 <= args.image <= len(state.shots or []) or not state.shots[args.image - 1].get("file"):
        print(f"❌ No image {args.image} in a set called '{args.resume}'.")
        return 1
    print(f"🗂️ Run folder: {state.id}")
    comfy = UpscaleClient()
    if not comfy.wait_until_reachable():
        print(f"❌ ComfyUI isn't running at {comfy.url}.")
        return 1
    try:
        upscale(state, args.image, comfy, args.method, min(1.0, max(0.0, args.strength)), max(0, args.resolution))
    except (RuntimeError, OSError) as e:
        print(f"❌ Could not upscale it: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

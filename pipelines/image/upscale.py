"""
Makes one image of a set bigger and sharper with SeedVR2, as a new image in the set beside the one it came from.

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


def target_resolution(width, height):
    """The shorter side it's made at: twice what it is, held to what fits."""
    return min(MOST_EDGE, FACTOR * min(width, height))


def upscale(state, position, comfy):
    """Makes the image at position again, bigger, and puts it in the set after it. Returns the new file's path."""
    from PIL import Image
    shot = state.shots[position - 1]
    source = state.image_path(shot)
    with Image.open(source) as image:
        width, height = image.size
    resolution = target_resolution(width, height)
    print(f"🔍 Upscaling image {position}, {width}x{height}, to {resolution} on its shorter side...")
    uploaded = comfy.upload(source, f"autocomfyui_{state.id}_upscale_{position}.png")
    entry = comfy._run(upscale_workflow(uploaded, resolution), 777)
    stem = unique_name(state.output_dir, os.path.splitext(shot["file"])[0] + SUFFIX, ".png")
    path = comfy._download(entry, os.path.join(state.output_dir, stem), IMAGE_EXTENSIONS, "an image")
    made = {**{key: value for key, value in shot.items() if key not in ("redo",)},
            "title": f"{shot['title']} · upscaled", "file": os.path.basename(path), "upscaled_from": position}
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
        upscale(state, args.image, comfy)
    except (RuntimeError, OSError) as e:
        print(f"❌ Could not upscale it: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

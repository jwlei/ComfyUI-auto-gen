"""
Makes one finished film bigger, as a new film beside it, with NVIDIA RTX Video Super Resolution (the ComfyUI
RTX VSR Long Video node), which works on the video in pieces on disk and keeps the sound.

    python -m pipelines.video.upscale --film "My film.mp4"
"""
import argparse
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

from core import paths as core_paths
from core.lora_check import comfyui_folders
from core.saved_workflows import unique_name
from pipelines.image.upscale import UpscaleClient
from pipelines.video import paths

SUFFIX = "_upscaled"
SCALES = (1.5, 2.0, 3.0, 4.0)
QUALITIES = ("HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA", "STREAMING_ULTRA", "ULTRA")
DEFAULT_QUALITY = "HIGHBITRATE_HIGH"
# H264 NVENC stops at 4096 pixels on a side.
MOST_H264_SIDE = 4096


def probe(path):
    """(width, height, seconds) of a video, or None if ffprobe can't read it."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                              "-of", "json", path], capture_output=True, text=True, timeout=30, check=True).stdout
        data = json.loads(out)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"]), float(data.get("format", {}).get("duration") or 0)
    except (OSError, subprocess.SubprocessError, ValueError, LookupError):
        return None


def encoder_for(width, height, scale):
    return "hevc_nvenc" if max(width, height) * scale > MOST_H264_SIDE else "h264_nvenc"


def upscale_workflow(video, scale, quality, strength, encoder):
    workflow = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "RTX_VSR_Long_Video", "inputs": {
            "video": ["1", 0], "scale": scale, "quality": quality, "strength": strength, "preprocess": "none",
            "preprocess_strength": 1.0, "frames_per_chunk": 48, "encoder": encoder}},
    }
    for node in workflow.values():
        node["_meta"] = {"title": node["class_type"]}
    return workflow


def upscale(path, comfy, scale=2.0, quality=DEFAULT_QUALITY, strength=1.0):
    """Makes the film at path again, bigger, next to it. Returns the new file's path."""
    size = probe(path)
    width, height = (size[0], size[1]) if size else (0, 0)
    encoder = encoder_for(width, height, scale)
    print(f"🔍 Upscaling {os.path.basename(path)}" + (f", {width}x{height}" if size else "") + f", {scale:g}x with RTX VSR ({quality}, {encoder})...")
    uploaded = comfy.upload(path, f"autocomfyui_upscale_{os.path.basename(path)}")
    entry = comfy._run(upscale_workflow(uploaded, scale, quality, strength, encoder), 777)
    folder = os.path.dirname(path)
    stem = unique_name(folder, os.path.splitext(os.path.basename(path))[0] + SUFFIX, ".mp4")
    made = comfy._download(entry, os.path.join(folder, stem), (".mp4",), "a video")
    remove_comfyui_copies(entry)
    print(f"✅ Upscaled: {made}")
    return made


def remove_comfyui_copies(entry):
    """The node saves its video under ComfyUI's own output folder; ours is the one kept, so that copy goes."""
    for node_output in entry.get("outputs", {}).values():
        for items in node_output.values():
            for item in items if isinstance(items, list) else []:
                if not (isinstance(item, dict) and item.get("type") == "output" and str(item.get("filename", "")).lower().endswith(".mp4")):
                    continue
                for root in comfyui_folders():
                    output = os.path.realpath(os.path.join(root, "output"))
                    path = os.path.realpath(os.path.join(output, item.get("subfolder", ""), item["filename"]))
                    if path.startswith(output + os.sep) and os.path.isfile(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass


def main(argv=None):
    load_dotenv(os.path.join(core_paths.PROJECT_ROOT, ".env"))
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--film", required=True, metavar="NAME", help="The film's file name in the Video output folder.")
    parser.add_argument("--scale", type=float, default=2.0, help="How many times bigger, 1 to 4.")
    parser.add_argument("--quality", choices=QUALITIES, default=DEFAULT_QUALITY)
    parser.add_argument("--strength", type=float, default=1.0, help="NVIDIA's effect strength, 0 to 1.")
    args = parser.parse_args(argv)
    path = os.path.join(paths.OUTPUT_DIR, os.path.basename(args.film))
    if not os.path.exists(path):
        print(f"❌ No film called '{args.film}'.")
        return 1
    comfy = UpscaleClient()
    if not comfy.wait_until_reachable():
        print(f"❌ ComfyUI isn't running at {comfy.url}.")
        return 1
    try:
        upscale(path, comfy, min(4.0, max(1.0, args.scale)), args.quality, min(1.0, max(0.0, args.strength)))
    except (RuntimeError, OSError) as e:
        print(f"❌ Could not upscale it: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

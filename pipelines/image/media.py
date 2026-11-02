import math
import os
import re

from PIL import Image, ImageOps

# Qwen-Image and Krea-2 both take sizes on this grid.
SIZE_STEP = 16
MIN_EDGE = 256
MAX_EDGE = 8192
# As in ComfyUI's own Resolution Selector, where 1 megapixel is 1024x1024.
PIXELS_PER_MEGAPIXEL = 1024 * 1024
DEFAULT_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 16.0
# Means: take this from the source image.
FROM_IMAGE = "image"
# What the panel offers, as (value, what it says). Any W:H works from the command line.
ASPECTS = (("1:1", "1:1 square"), ("2:3", "2:3 portrait photo"), ("3:2", "3:2 photo"), ("3:4", "3:4 portrait"), ("4:3", "4:3 standard"),
           ("9:16", "9:16 vertical"), ("16:9", "16:9 widescreen"), ("21:9", "21:9 ultrawide"))
MEGAPIXELS = (0.5, 1, 1.5, 2, 3, 4, 6, 8)
# Exact resolutions the panel offers, as (what it says, width, height): screens, to fill one pixel for pixel.
DESKTOP_SIZES = (("1080p Full HD", 1920, 1080), ("1440p QHD", 2560, 1440), ("1440p ultrawide", 3440, 1440), ("4K UHD", 3840, 2160))
PHONE_SIZES = (("OnePlus 15", 1272, 2772), ("Android Full HD+", 1080, 2400), ("Android QHD+", 1440, 3200),
               ("iPhone 16", 1179, 2556), ("iPhone 16 Pro Max", 1320, 2868))
THUMBNAIL_EDGE = 480


def on_grid(value):
    return min(MAX_EDGE, max(MIN_EDGE, round(value / SIZE_STEP) * SIZE_STEP))


def parse_size(text):
    """'1920x1080' as (width, height), or None if it isn't a size this can make."""
    match = re.fullmatch(r"(\d{2,5})x(\d{2,5})", (text or "").strip().lower())
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if MIN_EDGE <= width <= MAX_EDGE and MIN_EDGE <= height <= MAX_EDGE else None


def render_size(width, height):
    """What to have the model make for this size: the nearest on its grid that's no smaller, so it can be cropped to fit."""
    return tuple(math.ceil(side / SIZE_STEP) * SIZE_STEP for side in (width, height))


def parse_aspect(text):
    """'16:9' as a width/height ratio, or None if it isn't one."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", (text or "").strip())
    if not match or not float(match.group(1)) or not float(match.group(2)):
        return None
    return float(match.group(1)) / float(match.group(2))


def parse_megapixels(text):
    """A pixel count in megapixels, or None if it isn't a usable one."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if 0.05 <= value <= MAX_MEGAPIXELS else None


def pick_size(size=None, aspect=FROM_IMAGE, megapixels=DEFAULT_MEGAPIXELS, source=None):
    """
    The size to make, from what was asked for. source is the first image's (width, height), if there is one:
    with none, "the same as the image" means square and one megapixel, which is what a blank canvas gets.
    """
    if size:
        return parse_size(size)
    width, height = source or (1, 1)
    ratio = width / height if aspect == FROM_IMAGE else parse_aspect(aspect) or 1
    if megapixels == FROM_IMAGE:
        area = min(width * height, MAX_MEGAPIXELS * PIXELS_PER_MEGAPIXEL) if source else DEFAULT_MEGAPIXELS * PIXELS_PER_MEGAPIXEL
    else:
        area = float(megapixels) * PIXELS_PER_MEGAPIXEL
    return on_grid(math.sqrt(area * ratio)), on_grid(math.sqrt(area / ratio))


def save_source(image, path):
    """Whole: its own size may be the one asked for."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def fit(path, width, height, output):
    """The image at exactly this size, cropped at the edges if its shape differs, for a workflow that follows its source's size."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with Image.open(path) as image:
        ImageOps.fit(image.convert("RGB"), (width, height), Image.LANCZOS).save(output)
    return output


def trim(path, rendered, wanted):
    """
    Crops the finished image to the size asked for, when that isn't on the model's grid: 1080 isn't, so 1080p
    renders 1088 high and loses four rows off the top and bottom. Left alone if it came out any other size,
    as from a workflow that sets its own.
    """
    if rendered == tuple(wanted):
        return
    with Image.open(path) as image:
        if image.size != tuple(rendered):
            return
        kind = image.format
        left, top = (rendered[0] - wanted[0]) // 2, (rendered[1] - wanted[1]) // 2
        cropped = image.crop((left, top, left + wanted[0], top + wanted[1]))
        cropped.load()
    # Under another name until complete: a cut-off save must not pass as the image.
    part = path + ".part"
    cropped.save(part, format=kind)
    os.replace(part, path)


def image_problem(path):
    """What is wrong with this file as an image, or None."""
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError) as e:
        return f"it can't be read ({e})"
    return None


def thumbnail(path, output):
    """A small JPEG of the image, made once, so a gallery of big PNGs doesn't crawl over a phone connection."""
    if os.path.exists(output) and os.path.getmtime(output) >= os.path.getmtime(path):
        return output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with Image.open(path) as image:
        small = image.convert("RGB")
        small.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE))
        # Under another name until complete: two requests can ask for it at once.
        part = f"{output}.{os.getpid()}.part"
        small.save(part, format="JPEG", quality=85)
    os.replace(part, output)
    return output

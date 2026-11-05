"""
Changing part of an image, or expanding it, with a model that edits from images (Qwen-Image 2.1). The model can't be
given a mask, so it's shown where to work: as Kokon does (github.com/GabrieleLozupone/kokon, docs/engineering-notes.md),

    one area     <image1> is the part of the image around it, with what's removed or replaced blurred away first so it
                 isn't drawn back; <image2> the same with the area marked in a colour the prompt names.
    several      each area painted in its own colour on <image1>, and the prompt says what goes in each colour.
    expand       <image1> is the image on a larger canvas, the new borders a plain colour the prompt asks to fill.

Whatever the model makes, only the area (or the new borders) is taken from it: the rest of the image is the original,
pixel for pixel, through a feathered mask so the seam doesn't show.

prepare_areas() and prepare_expand() write what's shown to the model into the run's folder and return the run's
region, a dict of plain values kept with it; composite() puts a finished image back together.
"""
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

from pipelines.image.media import PIXELS_PER_MEGAPIXEL

ACTIONS = ("change", "remove", "replace", "add")
# Colours a mark can be, by the name the prompt uses. The panel paints with these.
COLOURS = {"red": (230, 40, 40), "blue": (40, 90, 235), "green": (40, 190, 70), "yellow": (245, 210, 30),
           "magenta": (225, 50, 200), "cyan": (40, 210, 220), "orange": (245, 140, 30), "purple": (130, 60, 210)}
# The model's tokens a job has room for, made image and images shown together (models.TOKEN_BUDGET), and what the part
# worked on is made at: about a megapixel, as Kokon's edits, fewer if it can't fit with the images it's shown.
WORK_MEGAPIXELS = 1.0
# The marked copy is shown at a quarter of the part's size: enough to find the area by, cheap in tokens (Kokon).
GUIDE_SHARE = 0.25
# How much of the image around the area the model sees, as a share of the area's longer side, and the least it works on.
CONTEXT = 0.6
LEAST_SIDE = 384
GRID = 32
# A mark is drawn this see-through on the copy the model reads it from.
MARK_ALPHA = 150


def _named(colour):
    """The mark colour nearest this one, as (name, rgb)."""
    rgb = _rgb(colour)
    return min(COLOURS.items(), key=lambda item: sum((a - b) ** 2 for a, b in zip(item[1], rgb)))


def _rgb(colour):
    if isinstance(colour, str):
        if colour.lower() in COLOURS:
            return COLOURS[colour.lower()]
        text = colour.lstrip("#")
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(colour[:3])


def _on_grid(value):
    return max(GRID, round(value / GRID) * GRID)


def work_size(width, height, budget_megapixels, shown_share):
    """
    The size to make a part at: about WORK_MEGAPIXELS, its shape kept, on the 32 grid, so the made image, image 1 at
    the same size and whatever else is shown at shown_share of it fit in the budget.
    """
    most = budget_megapixels * PIXELS_PER_MEGAPIXEL / (2 + shown_share) if budget_megapixels else WORK_MEGAPIXELS * PIXELS_PER_MEGAPIXEL
    area = min(WORK_MEGAPIXELS * PIXELS_PER_MEGAPIXEL, most)
    scale = math.sqrt(area / (width * height))
    return _on_grid(width * scale), _on_grid(height * scale)


def _area_masks(mask_path, size, areas):
    """Each area's mask (L, 255 inside), from the painted mask: a painted pixel belongs to the area of the colour nearest it."""
    with Image.open(mask_path) as painted:
        painted = np.asarray(painted.convert("RGBA").resize(size, Image.NEAREST)).astype(np.int32)
    inside = painted[..., 3] > 96
    if len(areas) == 1:
        return [Image.fromarray((inside * 255).astype(np.uint8))]
    wanted = np.array([_rgb(area["colour"]) for area in areas], dtype=np.int32)
    distance = ((painted[..., None, :3] - wanted[None, None]) ** 2).sum(axis=-1)
    nearest = distance.argmin(axis=-1)
    return [Image.fromarray(((inside & (nearest == index)) * 255).astype(np.uint8)) for index in range(len(areas))]


def _region(box, size):
    """The part worked on: the area's box with some of the image around it, at least LEAST_SIDE a side, within the image."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    margin = max(width, height) * CONTEXT
    side_w = max(width + 2 * margin, LEAST_SIDE)
    side_h = max(height + 2 * margin, LEAST_SIDE)
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    left = max(0, min(size[0] - side_w, centre_x - side_w / 2))
    top = max(0, min(size[1] - side_h, centre_y - side_h / 2))
    return [int(left), int(top), int(min(size[0], left + side_w)), int(min(size[1], top + side_h))]


def _dilate(mask, radius):
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1)) if radius >= 1 else mask


def _feather(mask, radius):
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius >= 1 else mask


def _contrasting(image, mask):
    """The mark colour that stands out most from what's under the area."""
    mean = ImageStat.Stat(image.convert("RGB"), mask).mean
    return max(("red", "green", "blue", "magenta", "cyan", "yellow"), key=lambda name: sum((a - b) ** 2 for a, b in zip(COLOURS[name], mean)))


def _blurred_out(image, mask):
    """The image with what's under the mask blurred away, so the model rebuilds it rather than keeping it."""
    radius = max(8, max(image.size) // 12)
    blurred = image.filter(ImageFilter.GaussianBlur(radius)).filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(blurred, image, _feather(_dilate(mask, 2), 2))


def _marked(image, mask, rgb, alpha=MARK_ALPHA):
    tint = Image.new("RGB", image.size, rgb)
    return Image.composite(tint, image, mask.point(lambda value: alpha if value > 127 else 0))


def area_prompt(action, text, colour):
    """The instruction for one area, in the frame Kokon tested on Qwen-Image 2.1: images by <imageN>, the area by colour."""
    area = f"the area marked in {colour} in <image2>"
    keep = "Keep everything outside the marked area exactly as it is."
    text = text.strip().rstrip(".")
    if action == "remove":
        what = text or "what was there"
        return (f"In <image1>, {area} has been blurred out to remove {what}. Fill the blurred area with what would naturally be "
                f"behind it, continuing the surrounding scene, textures, light and perspective, so that no outline or shadow of "
                f"it remains. {keep}")
    if action == "replace":
        return (f"In <image1>, {area} has been blurred out to clear what was there. Fill the blurred area with {text}, matching "
                f"the scene's light, perspective, scale and style so it looks naturally part of the image. {keep}")
    if action == "add":
        return f"In <image1>, add {text} inside {area}, fitted into the scene with matching light, perspective, scale and shadows. {keep}"
    return f"In <image1>, change only {area}: {text}. Blend the change into the surrounding scene. {keep}"


def colours_prompt(areas):
    """The instruction for several areas, each painted on <image1> in its own colour."""
    parts = [f"The area marked in {_named(area['colour'])[0]}: {area['text'].strip().rstrip('.')}." for area in areas]
    return ("<image1> has coloured marks painted on it, each showing where one change goes. " + " ".join(parts)
            + " Remove every coloured mark, blend each change into the surrounding scene, and keep everything outside the marked areas exactly as it is.")


def prepare_areas(source_path, mask_path, areas, folder, budget_megapixels=None):
    """
    What the model is shown to change the painted areas of an image, written into folder. areas: [{colour, action,
    text}], one per colour painted, in the order they were painted. Returns the region to keep with the run, with the
    prompt to use. Raises ValueError saying what's wrong.
    """
    if not areas:
        raise ValueError("Paint the area to change.")
    for area in areas:
        if area.get("action", "change") not in ACTIONS:
            raise ValueError(f"Unknown action: {area.get('action')}.")
        if not area.get("text", "").strip() and not (len(areas) == 1 and area.get("action") == "remove"):
            raise ValueError("Say what goes in each painted area (only Remove can go without).")
    with Image.open(source_path) as source:
        source = source.convert("RGB")
    masks = _area_masks(mask_path, source.size, areas)
    union = masks[0]
    for mask in masks[1:]:
        union = ImageChops.lighter(union, mask)
    box = union.getbbox()
    if not box:
        raise ValueError("Nothing is painted: paint the area to change.")
    region = _region(box, source.size)
    crop = source.crop(region)
    crop_union = union.crop(region)
    longest = max(crop.size)
    single = len(areas) == 1
    action = areas[0].get("action", "change") if single else "change"
    # A removal takes the halo and contact shadow just outside the stroke too.
    blend = _dilate(crop_union, max(4, round(longest * 0.01))) if action == "remove" else crop_union
    blend = _feather(_dilate(blend, max(1, round(longest * 0.006))), max(3, round(longest * 0.012)))
    size = work_size(*crop.size, budget_megapixels, GUIDE_SHARE if single else 0)
    work = crop.resize(size, Image.LANCZOS)
    work_mask = crop_union.resize(size, Image.NEAREST)
    os.makedirs(folder, exist_ok=True)
    shown = []
    if single:
        image1 = _blurred_out(work, work_mask) if action in ("remove", "replace") else work
        colour = _contrasting(work, work_mask)
        guide_size = tuple(_on_grid(side * math.sqrt(GUIDE_SHARE)) for side in size)
        guide = _marked(image1.resize(guide_size, Image.LANCZOS), work_mask.resize(guide_size, Image.NEAREST), COLOURS[colour], 200)
        shown = [("region_1.png", image1), ("region_2.png", guide)]
        prompt = area_prompt(action, areas[0].get("text", ""), colour)
    else:
        image1 = work
        for area, mask in zip(areas, masks):
            image1 = _marked(image1, mask.crop(region).resize(size, Image.NEAREST), _named(area["colour"])[1])
        shown = [("region_1.png", image1)]
        prompt = colours_prompt(areas)
    for name, image in shown:
        image.save(os.path.join(folder, name))
    blend.save(os.path.join(folder, "region_blend.png"))
    return {"kind": "areas", "box": region, "work": list(size), "size": list(source.size), "shown": [name for name, _ in shown],
            "blend": "region_blend.png", "prompt": prompt, "areas": areas}


def prepare_expand(source_path, left, top, right, bottom, folder, text="", budget_megapixels=None):
    """
    What the model is shown to expand an image by these many pixels a side, written into folder. Returns the region to
    keep with the run, with the prompt to use. Raises ValueError saying what's wrong.
    """
    sides = [max(0, int(round(value))) for value in (left, top, right, bottom)]
    if not any(sides):
        raise ValueError("Say how far to expand it on at least one side.")
    with Image.open(source_path) as source:
        source = source.convert("RGB")
    canvas = (source.width + sides[0] + sides[2], source.height + sides[1] + sides[3])
    if canvas[0] * canvas[1] > 64 * PIXELS_PER_MEGAPIXEL:
        raise ValueError("That would be more than 64 megapixels: expand it less.")
    size = work_size(*canvas, budget_megapixels, 0)
    scale = size[0] / canvas[0]
    mean = tuple(round(value) for value in ImageStat.Stat(source).mean)
    work = Image.new("RGB", size, mean)
    inner = (round(source.width * scale), round(source.height * scale))
    work.paste(source.resize(inner, Image.LANCZOS), (round(sides[0] * scale), round(sides[1] * scale)))
    os.makedirs(folder, exist_ok=True)
    work.save(os.path.join(folder, "region_1.png"))
    what = f" The new parts show {text.strip().rstrip('.')}." if text.strip() else ""
    prompt = ("Extend <image1> outward: the plain flat border around the picture is empty canvas. Fill it with a natural "
              f"continuation of the scene, matching its perspective, light, colours and style, with no visible seam or frame.{what} "
              "Keep the original picture in the middle exactly as it is.")
    return {"kind": "expand", "sides": sides, "work": list(size), "size": list(canvas), "source_size": list(source.size),
            "shown": ["region_1.png"], "prompt": prompt, "text": text}


def _seam_mask(size, inner_box, expanded, feather):
    """255 where the original is kept: the inner box, fading to 0 over feather pixels on each side that was expanded."""
    left, top, right, bottom = inner_box
    xs = np.arange(size[0], dtype=np.float32)[None, :] + 0.5
    ys = np.arange(size[1], dtype=np.float32)[:, None] + 0.5
    fade = np.ones((size[1], size[0]), dtype=np.float32)
    span = float(max(feather, 1))
    for grown, distance in zip(expanded, (xs - left, ys - top, right - xs, bottom - ys)):
        fade = np.minimum(fade, np.clip(distance / span, 0, 1) if grown else (distance > 0).astype(np.float32))
    fade = fade * fade * (3 - 2 * fade)
    return Image.fromarray(np.round(fade * 255).astype(np.uint8))


def composite(result_path, region, source_path, folder, output_path):
    """The finished image: the model's result where it was to work, the original everywhere else."""
    with Image.open(result_path) as made:
        made = made.convert("RGB")
    with Image.open(source_path) as source:
        source = source.convert("RGB")
    if region["kind"] == "areas":
        left, top, right, bottom = region["box"]
        with Image.open(os.path.join(folder, region["blend"])) as blend:
            blend = blend.convert("L")
        patch = made.resize((right - left, bottom - top), Image.LANCZOS)
        part = Image.composite(patch, source.crop(region["box"]), blend)
        finished = source.copy()
        finished.paste(part, (left, top))
    else:
        sides, size = region["sides"], tuple(region["size"])
        finished = made.resize(size, Image.LANCZOS)
        inner = (sides[0], sides[1], sides[0] + source.width, sides[1] + source.height)
        feather = max(2, round(min(source.size) * 0.03))
        keep = Image.new("RGB", size)
        keep.paste(source, (sides[0], sides[1]))
        finished = Image.composite(keep, finished, _seam_mask(size, inner, [bool(side) for side in sides], feather))
    part = output_path + ".part"
    finished.save(part, format="PNG")
    os.replace(part, output_path)
    return output_path

"""
What sets each image model apart: how big it can make an image well, and how many prompts an LLM can
write for it in one reply. core/saved_workflows.py says which model a workflow runs.

    Fibo 2.0 (Bria), edit turbo: trained at about 1 megapixel, every size within 10% of it. Above that
        the coordinates it was trained on run out from the top left corner, so the prompt leaks into
        that corner and it stops following it. Its prompts are long JSON captions, written in fibo2_prompt.py.
    Krea-2 Turbo: makes 1K to 2K images, up to 2048x2048, in 8 steps.
    Qwen-Image 2.1 Turbo: native 2K, but on this 12 GB card less. Kokon measured what fits: about 12,600 of the
        model's tokens (16 x 16 pixels each) on 12 GB, about 3.2 megapixels, which an edit shares between the image
        it makes and the one it's shown at the same size. So 3.2 to make one, half that to edit one.

Sources: the notes in ComfyUI's Fibo 2.0, Krea-2 and Qwen-Image 2.1 templates, its TextEncodeFibo2 node, the
model cards at huggingface.co/briaai/Fibo-2.0-edit-turbo and huggingface.co/Comfy-Org/Krea-2, and Kokon's planner
(token_budget in github.com/GabrieleLozupone/kokon, crates/qls-core/src/planner.rs).
"""
from pipelines.image.media import PIXELS_PER_MEGAPIXEL

FIBO2, KREA2, QWEN_IMAGE_EDIT = "fibo2", "krea2", "qwen_image_edit"
QWEN_IMAGE_21, QWEN_IMAGE_21_EDIT = "qwen_image_21", "qwen_image_21_edit"
LABELS = {FIBO2: "Fibo 2.0", KREA2: "Krea-2", QWEN_IMAGE_EDIT: "Qwen-Image-Edit", QWEN_IMAGE_21: "Qwen-Image 2.1",
          QWEN_IMAGE_21_EDIT: "Qwen-Image 2.1 (edit)"}
# The most megapixels each makes well. A model left out takes what the form offers.
MOST_MEGAPIXELS = {FIBO2: 1.1, KREA2: 4.2, QWEN_IMAGE_21: 3.2, QWEN_IMAGE_21_EDIT: 1.6}
# One reply can only hold so many of Fibo 2.0's captions, each a few hundred words of JSON.
MOST_WRITTEN = {FIBO2: 30}
# How many images it takes at most, where that's fewer than its workflow has room for. Qwen-Image 2.1 shows the model
# every image it's given beside the one it makes, so on 12 GB each one more leaves less for the rest.
MOST_IMAGES = {QWEN_IMAGE_21_EDIT: 4}
# The model's tokens a job has room for on this card, image made and images shown together, in megapixels (Kokon's
# token_budget for 12 GB). References are shrunk to share what's left: see reference_sizes().
TOKEN_BUDGET = {QWEN_IMAGE_21: 3.2, QWEN_IMAGE_21_EDIT: 3.2}
# What it calls the images it's given, in a prompt. Qwen-Image 2.1's encoder labels them <image1>, <image2>...
IMAGE_NAMES = {QWEN_IMAGE_21_EDIT: "<image{}>", QWEN_IMAGE_EDIT: "Picture {}"}
# Models that can make a new picture from images used as references, rather than edit image 1, by the workflow option
# that gives it a size of its own: on Qwen-Image 2.1's template, Custom size.
COMPOSE_SWITCH = {QWEN_IMAGE_21_EDIT: "face_custom_size"}
# A new picture from references has the references beside it: about one 1 MP reference's room less.
MOST_MEGAPIXELS_COMPOSE = {QWEN_IMAGE_21_EDIT: 2.2}
# Models that can work on part of an image: one marked in a colour on a second image it's shown (see regions.py).
REGION_EDITS = (QWEN_IMAGE_21_EDIT,)
# Models that make an image with a transparent background when the prompt says so, and how to say it (ComfyUI's
# Qwen-Image 2.1 template).
TRANSPARENT = {QWEN_IMAGE_21: "This is an RGBA format image with transparency. {} The image has an alpha channel and a transparent background.",
               QWEN_IMAGE_21_EDIT: "This is an RGBA format image with transparency. {} The image has an alpha channel and a transparent background."}
# Standard reference detail: images shown to the model besides the one it edits, at about a megapixel each.
REFERENCE_MEGAPIXELS = 1.0


def label(model):
    return LABELS.get(model, "This workflow's model")


def size_problem(model, width, height):
    """Why this model shouldn't make an image this size, or None."""
    most = MOST_MEGAPIXELS.get(model)
    if not most or width * height <= most * PIXELS_PER_MEGAPIXEL:
        return None
    return (f"{label(model)} makes images up to {most:g} megapixels, and {width}x{height} is {width * height / PIXELS_PER_MEGAPIXEL:.1f}. "
            "Choose fewer megapixels or a smaller size.")


def count_problem(model, count, direct):
    """Why the LLM can't write this many prompts for this model in one go, or None."""
    most = MOST_WRITTEN.get(model)
    if direct or not most or count <= most:
        return None
    return f"The LLM writes at most {most} prompts for {label(model)} at a time: its prompts are long. Make fewer, or use your own prompt."


def most_megapixels(model, compose=False):
    return (MOST_MEGAPIXELS_COMPOSE.get(model) if compose else None) or MOST_MEGAPIXELS.get(model)


def size_problem_for(model, width, height, compose=False):
    most = most_megapixels(model, compose)
    if not most or width * height <= most * PIXELS_PER_MEGAPIXEL:
        return None
    return (f"{label(model)} makes images up to {most:g} megapixels{' from references' if compose else ''}, and {width}x{height} is "
            f"{width * height / PIXELS_PER_MEGAPIXEL:.1f}. Choose fewer megapixels or a smaller size.")


def image_name(model, number):
    """What the prompt calls image number, for this model."""
    return IMAGE_NAMES.get(model, "image {}").format(number)


def reference_sizes(model, made, shown, detail="standard", editing=True):
    """
    The size to show the model each of its images at, so all of them fit in what the card holds with the one it makes.
    made: (width, height) of the image it makes. shown: [(width, height)] of the images it's shown, image 1 first.
    editing: image 1 is the image being edited, made at the size of the one made, and left as it is. The rest are at
    most a megapixel each with standard detail (Kokon's reference detail), less if they'd crowd out the rest. Full
    keeps them as they are. Returns [(width, height)], each on the 32 grid, its shape kept.
    """
    budget = TOKEN_BUDGET.get(model)
    sizes = [tuple(size) for size in shown]
    if detail == "full" or not sizes:
        return sizes
    fixed = 1 if editing else 0
    room = REFERENCE_MEGAPIXELS * PIXELS_PER_MEGAPIXEL
    if budget:
        used = made[0] * made[1] * (1 + fixed)
        others = len(sizes) - fixed
        if others:
            room = min(room, max(0.25 * PIXELS_PER_MEGAPIXEL, (budget * PIXELS_PER_MEGAPIXEL - used) / others))
    result = sizes[:fixed]
    for width, height in sizes[fixed:]:
        scale = (room / (width * height)) ** 0.5
        # One that fits is left as it is: the model's own encoder puts it on its grid.
        result.append((width, height) if scale >= 1 else tuple(max(32, round(side * scale / 32) * 32) for side in (width, height)))
    return result


def for_the_page():
    """What the form needs to know to grey out what a model can't do, and what it can do besides."""
    return {model: {"label": LABELS[model], "most_megapixels": MOST_MEGAPIXELS.get(model), "most_written": MOST_WRITTEN.get(model),
                    "most_images": MOST_IMAGES.get(model), "image_name": IMAGE_NAMES.get(model, "image {}"),
                    "compose": model in COMPOSE_SWITCH, "most_megapixels_compose": MOST_MEGAPIXELS_COMPOSE.get(model),
                    "regions": model in REGION_EDITS, "transparent": model in TRANSPARENT}
            for model in LABELS}

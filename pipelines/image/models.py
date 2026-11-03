"""
What sets each image model apart: how big it can make an image well, and how many prompts an LLM can
write for it in one reply. core/saved_workflows.py says which model a workflow runs.

    Fibo 2.0 (Bria), edit turbo: trained at about 1 megapixel, every size within 10% of it. Above that
        the coordinates it was trained on run out from the top left corner, so the prompt leaks into
        that corner and it stops following it. Its prompts are long JSON captions, written in fibo2_prompt.py.
    Krea-2 Turbo: makes 1K to 2K images, up to 2048x2048, in 8 steps.

Sources: the notes in ComfyUI's Fibo 2.0 and Krea-2 templates, its TextEncodeFibo2 node, and the
model cards at huggingface.co/briaai/Fibo-2.0-edit-turbo and huggingface.co/Comfy-Org/Krea-2.
"""
from pipelines.image.media import PIXELS_PER_MEGAPIXEL

FIBO2, KREA2, QWEN_IMAGE_EDIT = "fibo2", "krea2", "qwen_image_edit"
LABELS = {FIBO2: "Fibo 2.0", KREA2: "Krea-2", QWEN_IMAGE_EDIT: "Qwen-Image-Edit"}
# The most megapixels each makes well. A model left out takes what the form offers.
MOST_MEGAPIXELS = {FIBO2: 1.1, KREA2: 4.2}
# One reply can only hold so many of Fibo 2.0's captions, each a few hundred words of JSON.
MOST_WRITTEN = {FIBO2: 30}


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


def for_the_page():
    """What the form needs to know to grey out what a model can't do."""
    return {model: {"label": LABELS[model], "most_megapixels": MOST_MEGAPIXELS.get(model), "most_written": MOST_WRITTEN.get(model)}
            for model in LABELS}

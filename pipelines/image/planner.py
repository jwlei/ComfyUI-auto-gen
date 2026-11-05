from pydantic import BaseModel, ValidationError

from pipelines.image import fibo2_prompt, models
from pipelines.video import planner as llm

PROMPT_ATTEMPTS = 3
# More than one reply should be asked to hold.
MAX_IMAGES = 100


class Version(BaseModel):
    title: str
    prompt: str


class Versions(BaseModel):
    title: str
    versions: list[Version]


# Claude's structured output needs every object closed and every field required.
VERSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "versions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "prompt": {"type": "string"}},
                "required": ["title", "prompt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "versions"],
    "additionalProperties": False,
}


def _names(images, model):
    """How the prompts refer to the attached images, when there's more than one."""
    if images < 2:
        return ""
    first, second = models.image_name(model, 1), models.image_name(model, 2)
    return f"""
    - You have been given {images} images. The model knows them as {first}, {second} and so on, in the order they are
      attached, so call them that whenever you refer to one."""


def _edit_rules(images, model=None):
    """
    The model gets the image and this instruction, and redraws the image following it. What Kokon's blind A/B tests on
    Qwen-Image settled (enhancer.rs there): lead with the operation, describe only what changes, and end on one clause
    keeping the rest. Describing what stays makes the model draw it again; an example in the instructions turns up in
    the prompts, so there are none.
    """
    return f"""
    An AI image editing model ({models.label(model) if model else "an image editor"}) will edit the attached image
    once per prompt. It sees the image and that prompt, and knows nothing about the others. Look at the image closely
    first: who or what is in it, the pose, the framing, the setting, the light, the medium.

    Each prompt is an instruction to edit that image:
    - Start with the operation: Change, Replace, Remove, Add or Turn. Then say exactly what changes, where it is, and
      what it becomes, plainly and specifically: name its colours, materials, shapes, sizes and textures.
    - Describe only what changes. Never describe what stays: the model redraws whatever a prompt describes.
    - Keep the same person, with the same face and identity, unless the brief asks otherwise; a change to someone
      never makes them someone else.
    - End with exactly one sentence keeping the rest: "Keep everything else in the image unchanged." For a change to
      the whole picture's style, light or mood, end with "Keep the composition and the subjects unchanged." instead.
    - Copy anything the brief fixes exactly: words to show, in double quotes, counts and colours.
    - One short paragraph, 20-60 words, up to about 120 for several changes at once. No lists, no labels, no
      markdown.{_names(images, model)}"""


def _compose_rules(images, model=None):
    """
    The images are references for a new picture, not one to edit: as Kokon's compose. Never asking to keep a
    reference unchanged, which turns the new picture into an edit of image 1.
    """
    names = ", ".join(models.image_name(model, number) for number in range(1, images + 1))
    return f"""
    An AI image model ({models.label(model) if model else "an image model"}) will make a new picture once per prompt,
    using the attached images as references: {names}. It sees them and that prompt, and knows nothing about the
    others. Look at each image closely first, and decide what the brief wants from it: a person's face and look, a
    product with its shape, colours and markings, a place, a style.

    Each prompt describes the new picture:
    - Say what is taken from each image, calling it {models.image_name(model, 1)}, {models.image_name(model, 2)} and so on,
      then describe the new scene around it: what happens, the setting, the composition and camera view, the medium
      and the light.
    - Never ask to keep an image unchanged or as it is: the picture is a new one, and only borrows from them.
    - Everyone in an image is an adult, and looks it.
    - Copy anything the brief fixes exactly: words to show, in double quotes, counts and colours.
    - One paragraph of plain prose, 100-200 words. No lists, no labels, no markdown."""


def _create_rules(model=None):
    """The rules for a new image from text: the medium early, the light always, fixed content copied word for word."""
    return f"""
    An AI image model ({models.label(model) if model else "an image model"}) will make one image per prompt. It sees
    only that prompt, and knows nothing about the others, so each has to say everything about how its image looks.
    Everyone in an image is an adult, and looks it.

    Each prompt describes one finished image:
    - Name the medium early: a photograph, a painting, an illustration, a render. If the brief names one, use that.
    - Then the subject and what it is doing, the setting and background, the composition and camera view, and always
      the light. Keep each thing's details next to it, and be concrete about colours, materials and textures.
    - If several images share a subject, describe it in the same words in each, or they won't look alike.
    - No text in the image: no letters, captions, signs, logos or watermarks, unless the brief gives the exact words,
      in which case copy them exactly, in double quotes.
    - Copy anything else the brief fixes exactly, such as counts and colours.
    - No quality tags, sizes or negative prompts.
    - One paragraph of plain prose, 80-150 words, up to about 250 for a crowded scene. No lists, no labels, no markdown."""


TRANSPARENT_RULE = """

    The images are made with a transparent background, to be cut out and placed on others: describe only the subject,
    whole and in view, with no background, floor, frame or scenery around it."""


def _instructions(brief, count, images, model=None, compose=False, transparent=False):
    """How to write the prompts, for editing what's attached, making a new picture from it, or making one from nothing."""
    versions = "one image" if count == 1 else f"{count} different versions"
    apart = ("" if count == 1 else f"""

    Every one of the {count} has to be clearly different from all the others in what the brief wants varied: someone
    looking through them should never think two are the same idea. Spread them widely instead of making small
    changes around one idea, and don't let them drift off the brief to get there. Change only what the brief wants varied.""")
    if model == models.FIBO2:
        rules = fibo2_prompt.rules(images)
    elif images and compose:
        rules = _compose_rules(images, model)
    elif images:
        rules = _edit_rules(images, model)
    else:
        rules = _create_rules(model)
    return f"""
    You are writing the prompts for {versions} from this brief, in the words of the person making them: "{brief}"

    The brief is short on purpose. Invent the detail, in keeping with it and never contradicting it.{apart}
    {rules}{TRANSPARENT_RULE if transparent else ""}

    Return a title for the set, 2-5 words, and exactly {count} {"version" if count == 1 else "versions"}. For each:
    - title: 2-5 words saying what sets this one apart.
    - prompt: the text the model gets, following the rules above."""


def _tidy(model, has_images):
    """How a prompt goes to this model. Fibo 2.0's are JSON captions: one that isn't the caption asked for comes back empty."""
    return (lambda prompt: fibo2_prompt.tidy(prompt, has_images)) if model == models.FIBO2 else str.strip


def _usable(versions, tidy):
    """The versions with a prompt the model can use, each prompt once, as [{title, prompt}]."""
    seen, usable = set(), []
    for version in versions:
        prompt = tidy(version.prompt)
        if prompt and prompt not in seen:
            seen.add(prompt)
            usable.append({"title": version.title.strip() or f"Version {len(usable) + 1}", "prompt": prompt})
    return usable


def write_prompts(image_jpegs, brief, count, llm_name, model=None, compose=False, transparent=False):
    """
    The set's title and {count} versions from one request, however many: the prompts have to know each other
    to differ, and it spends one of a free model's daily requests rather than {count}.
    """
    instructions = _instructions(brief, count, len(image_jpegs), model, compose, transparent)
    tidy = _tidy(model, bool(image_jpegs))
    best, title, problem = [], None, None
    for attempt in range(1, PROMPT_ATTEMPTS + 1):
        correction = f"\n\n    Your last reply came back with {problem}. Fix that." if problem else ""
        reply = llm._ask(llm_name, instructions + correction, Versions, VERSIONS_SCHEMA, image_jpegs)
        try:
            versions = Versions.model_validate_json(reply)
        except ValidationError as e:
            problem = f"an unreadable reply ({e.error_count()} errors)"
        else:
            usable = _usable(versions.versions, tidy)
            # Extra ones were paid for and do no harm, so they're only left out.
            if len(usable) >= count:
                return versions.title.strip() or "Untitled", usable[:count]
            if len(usable) > len(best):
                best, title = usable, versions.title.strip() or "Untitled"
            problem = f"{len(usable)} different prompts instead of {count}"
        print(f"   ⚠️ The prompts came back with {problem} (attempt {attempt}/{PROMPT_ATTEMPTS}).")
    if best:
        print(f"   ⚠️ Going on with the {len(best)} it did write, rather than paying for more attempts.")
        return title, best
    raise RuntimeError(f"Could not get usable prompts: {problem}.")

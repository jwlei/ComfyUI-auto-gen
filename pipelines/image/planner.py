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


def _edit_rules(images):
    """The model gets the image and this instruction, and redraws the image following it."""
    pictures = "" if images == 1 else f"""
    - You have been given {images} images. The model knows them as Picture 1, Picture 2 and so on, in the order
      they are attached, so call them that whenever you refer to one."""
    return f"""
    An AI image editing model (Qwen-Image-Edit) will redraw the attached image once per prompt. It sees the image
    and that prompt, and knows nothing about the others. Look at the image closely first: who or what is in it,
    the pose, the framing, the setting, the light, the medium.

    Each prompt is an instruction to edit that image:
    - Say what changes, plainly and specifically: "Change her outfit to a fitted navy wool coat over a cream
      turtleneck" rather than "give her a nicer outfit". Name colours, materials, cuts, lengths, textures.
    - Then say what stays. Unless the brief asks to change it, keep the same person with the same face, identity,
      skin tone and build, and the same pose, framing, background and light. Say so in the prompt.
    - Don't describe again in full what the image already shows and isn't changing. It is in front of the model.
    - One or two plain sentences, 25-70 words. No lists, no labels, no markdown.{pictures}"""


def _create_rules():
    return """
    An AI image model (Krea-2) will make one image per prompt. It sees only that prompt, and knows nothing about the
    others, so each has to say everything about how its image looks.

    Each prompt describes one finished image:
    - Start with the subject and what it is doing, then the setting, then the medium or photographic style,
      the framing and camera angle, and the light. Keep each thing's details next to it.
    - Be concrete: name colours, materials, textures and the time of day. Say "photo of" or "oil painting of"
      when the medium matters, and if the brief names one, use that.
    - If several images share a subject, describe it in the same words in each, or they won't look alike.
    - No text in the image: no letters, captions, signs, logos or watermarks, unless the brief gives the exact words,
      in which case put those words in quotes.
    - One paragraph of plain prose, 40-110 words. No lists, no labels, no markdown, no negative prompts."""


def _instructions(brief, count, images, model=None):
    """How to write the prompts, for editing what's attached or for making an image from nothing."""
    versions = "one image" if count == 1 else f"{count} different versions"
    apart = ("" if count == 1 else f"""

    Every one of the {count} has to be clearly different from all the others in what the brief wants varied: someone
    looking through them should never think two are the same idea. Spread them widely instead of making small
    changes around one idea, and don't let them drift off the brief to get there. Change only what the brief wants varied.""")
    return f"""
    You are writing the prompts for {versions} from this brief, in the words of the person making them: "{brief}"

    The brief is short on purpose. Invent the detail, in keeping with it and never contradicting it.{apart}
    {fibo2_prompt.rules(images) if model == models.FIBO2 else _edit_rules(images) if images else _create_rules()}

    Return a title for the set, 2-5 words, and exactly {count} {"version" if count == 1 else "versions"}. For each:
    - title: 2-5 words saying what sets this one apart.
    - prompt: the text the model gets, following the rules above."""


def write_prompts(image_jpegs, brief, count, llm_name, model=None):
    """
    The set's title and {count} versions from one request, however many: the prompts have to know each other
    to differ, and it spends one of a free model's daily requests rather than {count}.
    """
    instructions = _instructions(brief, count, len(image_jpegs), model)
    # Fibo 2.0's prompts are JSON captions: one that isn't the caption asked for can't be used.
    tidy = (lambda prompt: fibo2_prompt.tidy(prompt, bool(image_jpegs))) if model == models.FIBO2 else str.strip
    best, title, problem = [], None, None
    for attempt in range(1, PROMPT_ATTEMPTS + 1):
        correction = f"\n\n    Your last reply came back with {problem}. Fix that." if problem else ""
        reply = llm._ask(llm_name, instructions + correction, Versions, VERSIONS_SCHEMA, image_jpegs)
        try:
            versions = Versions.model_validate_json(reply)
        except ValidationError as e:
            problem = f"an unreadable reply ({e.error_count()} errors)"
        else:
            seen, usable = set(), []
            for version in versions.versions:
                prompt = tidy(version.prompt)
                if prompt and prompt not in seen:
                    seen.add(prompt)
                    usable.append({"title": version.title.strip() or f"Version {len(usable) + 1}", "prompt": prompt})
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

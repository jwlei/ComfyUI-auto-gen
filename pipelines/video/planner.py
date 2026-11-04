import base64
import math
import os
import re

import anthropic
from google.genai import types
from pydantic import BaseModel, ValidationError

from core.api_log import log_api_call
from core.gemini import _generate
from pipelines.video import h3_prompt, roles, wan_prompt

# Who can write the plan: (name, what the panel calls it).
LLMS = (("gemini", "Gemini (free tier)"), ("claude", "Claude Opus 5.5 (paid per use)"))

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
# Its default is 'medium'. A film's worth of prompts that have to fit together earns more thought.
CLAUDE_EFFORT = "high"
# Long replies are streamed, so a high ceiling can't run into a request timeout.
CLAUDE_MAX_TOKENS = 64000

# A whole film's prompts come back in one reply: about 75 of them at the longest,
# which takes a while. A model that says nothing for this long is overloaded or
# stuck, so move on.
PLAN_TIMEOUT = 360

# How long a whole film can be, in seconds: one shortest clip, up to 10 minutes.
MIN_FILM_SECONDS = 5
MAX_FILM_SECONDS = 600
# How long the video model can make a clip, in seconds: (shortest, longest, usual). Shorter clips
# can't hold an action; longer ones go beyond what the model was trained on, or the workflow's memory.
# - LTX-2.3 (the built-in workflow, and any model without a line of its own): about 10 s holds up.
# - MiniMax H3: trained on 124-362 frames at 24 fps, so about 5-15 s.
# - Wan 2.2: trained on 81 frames at 16 fps, so 5 s at most.
CLIP_SECONDS = {
    None: (5, 10, 8),
    h3_prompt.MODEL: (5, 15, 12),
    wan_prompt.MODEL: (3, 5, 5),
}
MIN_SHOT_SECONDS, MAX_SHOT_SECONDS, AVERAGE_SHOT_SECONDS = CLIP_SECONDS[None]
# Too few clips to tell a story with: they show one moment instead.
MIN_SHOTS_FOR_A_STORY = 3
PLAN_ATTEMPTS = 3
# What each image after the first is. EXTRA_CONTEXT leaves it to the LLM to tell by looking.
EXTRA_CONTEXT, ANOTHER_CHARACTER, SAME_SUBJECT, PLACE = roles.CONTEXT, roles.CHARACTER, roles.VIEW, roles.PLACE
IMAGE_ROLES = IMAGE_2_ROLES = roles.ROLES
ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"}

CONTINUITY = {
    "chain": """Each clip starts on the exact last frame of the clip before it, so this is one
    continuous take with no cuts. Every shot picks up exactly where the previous one ended. A new
    place can only be reached by moving there on screen: the subject walks, the camera travels.
    End each shot on a clear, steady moment that the next one can continue from.""",
    "image": """Every clip starts again from the attached image, with a fade between clips. So each
    shot is its own moment growing out of that same opening frame: a different action, a different
    camera move, a different development of the idea, in an order that still builds.""",
}
# With no image every clip starts from its prompt, and with references from all of them.
SEPARATE_CLIPS = """Every clip is made on its own, with a fade between clips. So each shot is its own
    moment: a different action, a different camera move, a different development of the idea, in an
    order that still builds."""


class Shot(BaseModel):
    title: str
    seconds: int
    prompt: str


class Plan(BaseModel):
    title: str
    shots: list[Shot]


class Metadata(BaseModel):
    title: str
    description: str
    tags: list[str]


# Claude's structured output needs every object closed and every field required.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "seconds": {"type": "integer"}, "prompt": {"type": "string"}},
                "required": ["title", "seconds", "prompt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "shots"],
    "additionalProperties": False,
}
METADATA_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"},
                   "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "description", "tags"],
    "additionalProperties": False,
}


def default_llm():
    configured = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    return configured if configured in dict(LLMS) else "gemini"


def llm_configured(llm):
    """Whether this LLM has credentials. Claude's can also be an `ant auth login` profile on disk."""
    if llm == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE")
                or os.path.isdir(os.path.expanduser(os.path.join("~", ".config", "anthropic"))))


def clip_seconds(video_model=None, chosen=None):
    """
    (shortest, longest, usual) clip on this video model, in seconds. chosen: the clip length asked for, (shortest,
    longest), held within what the model makes; its usual length is then the middle of it.
    """
    shortest, longest, usual = CLIP_SECONDS.get(video_model, CLIP_SECONDS[None])
    if chosen:
        low, high = max(shortest, chosen[0]), min(longest, chosen[1])
        if low <= high:
            return low, high, min(high, max(low, (low + high + 1) // 2))
    return shortest, longest, usual


def shot_count(seconds, video_model=None, chosen=None):
    """The ceiling matters just past the longest clip, which rounds to one clip that couldn't be long enough."""
    shortest, longest, usual = clip_seconds(video_model, chosen)
    count = max(math.ceil(seconds / longest), round(seconds / usual), 1)
    # Never so many that even the shortest clips add up to more than the film.
    return max(1, min(count, seconds // shortest)) if seconds >= shortest else 1


def fit_durations(lengths, target, video_model=None, chosen=None):
    """The model's lengths within the clip limits, nudged a second at a time until they add up to the film's."""
    shortest, longest, _ = clip_seconds(video_model, chosen)
    fitted = [min(longest, max(shortest, length)) for length in lengths]
    while sum(fitted) != target:
        longer = sum(fitted) < target
        room = [(longest - length) if longer else (length - shortest) for length in fitted]
        if not any(room):
            break
        fitted[room.index(max(room))] += 1 if longer else -1
    return fitted


def _what_it_opens_on(images):
    if not images:
        return "There is no image: you decide how everything looks, in keeping with the brief."
    if images == 1:
        return "The attached image is the film's opening frame. Look at it closely: the subject,\n    the place, the light, the style."
    return (f"The {images} attached images are references the video model gets with every clip. Look at them closely:\n"
            f"    the subjects, the places, the light, the style.")


def _how_it_looks(images):
    if not images:
        return """- Nothing shows the video model how anything looks, so describe the subject, place, light and
      style in full, in the same words every time, or the clips won't match."""
    if images == 1:
        return """- The start frame already shows how everything looks, so call the subject by the same few words
      every time ("the woman in the red coat") instead of describing it again in full."""
    pictures = " and ".join(f"<Picture {number}>" for number in range(1, images + 1))
    return f"""- The references show how everything looks. Call them {pictures}, in the order they are attached,
      instead of describing them again in full."""


def _second_image(images, image_roles, in_words, start, video_model):
    """
    What the images after the first are for, if there are any. in_words: a second image only the
    LLM sees, so it reaches the video model as text.
    """
    total = images + bool(in_words)
    if total < 2:
        return ""
    image_roles = roles.image_roles(image_roles, total)
    if total > 2:
        return _more_images(image_roles, video_model)
    role = image_roles[0]
    is_h3 = video_model == h3_prompt.MODEL
    # Wan takes no labels, and a label would mean nothing for an image the video model doesn't get.
    labels = not in_words and video_model != wan_prompt.MODEL
    first, second = ("<Picture 1>", "<Picture 2>") if labels else ("the first image", "the second image")
    opening = first[0].upper() + first[1:]
    alone = """A second image is attached for you alone. The video model never sees it, so all it can get from
    that image is what your prompts say. """ if in_words else ""
    if role == ANOTHER_CHARACTER:
        if in_words:
            enter = "Describe" if is_h3 else "They aren't in the opening frame, so bring them on screen as part of the action, and describe"
            how = f"{enter} how they look in full, in the same words, in every prompt they appear in."
        else:
            how = "" if is_h3 else "Call them <Picture 2> in every clip they appear in." if labels else "Call them by the same few words in every clip they appear in."
        return f"""{alone}{opening} shows the film's subject and {second} shows another character, who counts
    as asked for. Give them a real part in the film, alongside the subject. {how}"""
    same = "Name those details in your prompts whenever they would be in view." if in_words else "Both pictures show one subject."
    place = ""
    if in_words:
        arrive = "early in each clip" if start == "image" else "early in the film"
        reach = "Describe" if is_h3 else f"""The opening frame still shows the subject where {first} was taken, so get them there on screen
      {arrive}, and from then on describe"""
        place = f"""{reach} that place in the same words in every prompt. What this image adds isn't in anything the
      video model can see, so it does get described, whatever the rules below say about not describing things."""
    if role == SAME_SUBJECT:
        return f"""{alone}{opening} shows the film's subject, and {second} shows the same subject again: another
    angle, the rest of an outfit or details {first} hides. It is more of how they look, and brings in nobody new. {same}"""
    if role == PLACE:
        return f"""{alone}{opening} shows the film's subject, and {second} shows a place: that is where the film
    happens, so put the subject from {first} there and leave the place in {first} behind. It brings in nobody new. {place}"""
    return f"""{alone}{opening} shows the film's subject, and {second} adds to it. If the brief says how to use it, do
    that. Otherwise look at it: it is one of these two, and it brings in nobody new.
    - It shows the same subject: it is more of how they look, such as another angle, the rest of an outfit or
      details {first} hides. {same}
    - It shows a place: that is where the film happens, so put the subject from {first} there and leave
      the place in {first} behind. {place}"""


def _more_images(image_roles, video_model):
    """Three images or more: what each one after the first is for, by the role it was given."""
    is_h3 = video_model == h3_prompt.MODEL
    labels = video_model != wan_prompt.MODEL
    picture = lambda n: f"<Picture {n}>" if labels else f"the {ORDINALS.get(n, f'number {n}')} image"
    first = picture(1)
    lines = []
    for number, role in enumerate(image_roles, 2):
        this = picture(number)
        if role == SAME_SUBJECT:
            lines.append(f"- {this} shows the same subject as {first}: another angle, the rest of an outfit or details {first} hides. It brings in nobody new.")
        elif role == ANOTHER_CHARACTER:
            call = "" if is_h3 else f" Call them {this} in every clip they appear in." if labels else " Call them by the same few words in every clip they appear in."
            lines.append(f"- {this} shows another character, who counts as asked for. Give them a real part in the film, alongside the subject.{call}")
        elif role == PLACE:
            lines.append(f"- {this} shows a place where the film happens: put the subject there, and leave the place in {first} behind.")
        else:
            lines.append(f"- {this} adds to the subject. Look at it: if it shows the same subject, it is more of how they look; if it shows a place, the film happens there. It brings in nobody new.")
    moves = ("\n    More than one image shows a place: the film moves between them, in an order that suits the brief."
             if image_roles.count(PLACE) > 1 else "")
    opening = first[0].upper() + first[1:]
    return (f"{opening} shows the film's subject. Each image after it adds something, and if the brief says how to use one, do that.\n    "
            + "\n    ".join(lines) + moves)


def _instructions(brief, seconds, start, images=1, video_model=None, image_2_role=EXTRA_CONTEXT, image_2_in_words=False,
                  references=None, chosen=None):
    """references: the reference videos and audio the clips get, as h3_prompt.References."""
    second_image = _second_image(images, image_2_role, image_2_in_words, start, video_model)
    guide = _guide(brief, seconds, start, images, video_model, image_2_role, second_image, references, chosen)
    # Every guide speaks of one attached image, and now there are two.
    return guide.replace("attached image", "first attached image") if image_2_in_words else guide


def _guide(brief, seconds, start, images, video_model, image_2_role, second_image, references=None, chosen=None):
    count = shot_count(seconds, video_model, chosen)
    shortest, longest, _ = clip_seconds(video_model, chosen)
    if video_model == h3_prompt.MODEL:
        return h3_prompt.instructions(brief, seconds, count, images, shortest, longest,
                                      second_image, roles.image_roles(image_2_role, images), references)
    clips = "one clip" if count == 1 else f"{count} consecutive clips"
    shots = "one shot" if count == 1 else f"{count} shots in order"
    shape = (f"Give the film a beginning, a development and a real ending, rather than {count} versions of the same moment."
             if count >= MIN_SHOTS_FOR_A_STORY else "This is too short for a story: show one clear moment that feels complete.")
    together = f"Use {seconds}." if count == 1 else f"All of them together add up to {seconds}."
    continuity = "" if count == 1 else SEPARATE_CLIPS if start == "image" and images != 1 else CONTINUITY[start]
    if video_model == wan_prompt.MODEL:
        return wan_prompt.instructions(brief, seconds, count, images, continuity, shortest, longest, second_image)
    in_keeping = "the brief and\n    the image, and never contradicting them" if images else "the brief, and never contradicting it"
    sees = "one prompt and one start frame" if images == 1 else "one prompt and its images" if images else "one prompt"
    return f"""
    You are directing a {seconds}-second film made of {clips} from an AI video
    model (LTX-2). {_what_it_opens_on(images)}

    The brief, in the words of the person making it: "{brief}"

    The brief is short on purpose. Invent the detail the film needs, in keeping with {in_keeping}. {shape} Bring in no new main characters unless
    the brief asks for them.

    {second_image}

    {continuity}

    Return a title for the film, and exactly {shots}. For each shot:
    - title: 2-5 words.
    - seconds: a whole number from {shortest} to {longest}. {together}
    - prompt: the text the video model gets, following the rules below.

    The video model sees {sees} at a time, and knows nothing about the film:
    - One flowing paragraph in the present tense, 60-120 words.
    - Describe what happens in order, as it can really happen in that shot's seconds. One clear main
      action per shot: a crowded prompt comes out as a blur.
    - Name one camera move, and describe the light and the sound.
    {_how_it_looks(images)}
    - Sound comes from the scene itself: footsteps, wind, a room. No music, because it would jump
      every time one clip hands over to the next. No speech unless the brief asks for it.
    - No titles, captions, text on screen or logos.
    - Never mention other shots, cuts or the film. Write only what is seen and heard in this clip.
    """


# --- Gemini ---

def _image_list(image_jpeg):
    """One image's bytes, several in a list, or None."""
    return image_jpeg if isinstance(image_jpeg, list) else [image_jpeg] if image_jpeg else []


def _ask_gemini(prompt, model, image_jpeg=None):
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=model)
    contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in _image_list(image_jpeg)] + [prompt]
    return _generate(contents, config=config, timeout=int(os.getenv("GEMINI_TIMEOUT") or PLAN_TIMEOUT)).text or ""


# --- Claude ---

def _ask_claude(prompt, schema, image_jpeg=None):
    """
    Returns the reply's JSON text. Asks for the server-side fallback: a request one
    model's safety classifier declines is run on another instead of failing the plan.
    """
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(image).decode("utf-8")}}
               for image in _image_list(image_jpeg)] + [{"type": "text", "text": prompt}]
    try:
        with anthropic.Anthropic().beta.messages.stream(
            model=os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": CLAUDE_EFFORT, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude refused the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.PermissionDeniedError as e:
        raise RuntimeError(f"Claude doesn't allow this key to do that: {e.message}")
    except anthropic.RateLimitError:
        raise RuntimeError("Claude is rate limiting this key. Try again in a minute.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude returned an error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach Claude. Check the internet connection.")
    except TypeError:  # What the SDK raises when it finds no credentials at all
        raise RuntimeError("No Claude credentials. Put ANTHROPIC_API_KEY in .env (see .env.example).")

    log_api_call("Claude", response.model)
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this image or prompt. Try Gemini, or change the prompt.")
    # After a fallback the reply can hold the declined model's partial text first: the last block is the answer.
    texts = [block.text for block in response.content if block.type == "text"]
    return texts[-1] if texts else ""


def _ask(llm, prompt, model, schema, image_jpeg=None):
    if llm == "claude":
        return _ask_claude(prompt, schema, image_jpeg)
    return _ask_gemini(prompt, model, image_jpeg)


# --- What the pipeline asks for ---

def plan_film(image_jpeg, brief, seconds, start, llm="gemini", video_model=None, image_2_role=EXTRA_CONTEXT, image_2_in_words=None,
              references=None, chosen=None):
    """
    Turns the image (several in a list, or none) and the short brief into a film's shot list. One request for the
    whole film rather than one per clip: the prompts have to know each other, and a
    long film would otherwise spend more than a free model's whole daily quota.
    image_2_in_words: a second image the clips have no room for, which only the LLM sees.
    references: the reference videos and audio every clip gets, as h3_prompt.References. The LLM only reads what
    they are for: it can't watch or hear them.
    """
    images = len(_image_list(image_jpeg))
    shortest, longest, _ = clip_seconds(video_model, chosen)
    instructions = _instructions(brief, seconds, start, images, video_model, image_2_role, bool(image_2_in_words), references, chosen)
    attached = _image_list(image_jpeg) + _image_list(image_2_in_words)
    image_roles = roles.image_roles(image_2_role, images)
    is_h3 = video_model == h3_prompt.MODEL

    problem = None
    for attempt in range(1, PLAN_ATTEMPTS + 1):
        # H3's rules are strict enough that saying what went wrong is worth it.
        correction = f"\n\n    Your last reply came back with {problem}. Fix that." if is_h3 and problem else ""
        reply = _ask(llm, instructions + correction, Plan, PLAN_SCHEMA, attached)
        try:
            plan = Plan.model_validate_json(reply)
        except ValidationError as e:
            problem = f"an unreadable reply ({e.error_count()} errors)"
        else:
            shots = [shot for shot in plan.shots if shot.prompt.strip()]
            # Too few or too many shots can't be fixed by changing their lengths.
            if is_h3:
                for shot, prompt in zip(shots, h3_prompt.tidy([shot.prompt for shot in shots], images, image_roles, references)):
                    shot.prompt = prompt
            wrong_prompts = is_h3 and shots and h3_prompt.film_problem([shot.prompt for shot in shots], images, references)
            thin = is_h3 and shots and not wrong_prompts and h3_prompt.too_thin([shot.prompt for shot in shots], images)
            if wrong_prompts:
                problem = wrong_prompts
            # Worth asking once. A film of short prompts is better than no film.
            elif thin and attempt == 1:
                problem = thin
            elif len(shots) * shortest <= seconds <= len(shots) * longest:
                if thin:
                    print(f"   ⚠️ Using {thin}: expect less detail in the film.")
                lengths = fit_durations([shot.seconds for shot in shots], seconds, video_model, chosen)
                return plan.title.strip() or "Untitled", [
                    {"title": shot.title.strip(), "seconds": length, "prompt": shot.prompt.strip()}
                    for shot, length in zip(shots, lengths)]
            else:
                problem = f"{len(shots)} shots for a {seconds}s film"
        print(f"   ⚠️ The plan came back with {problem} (attempt {attempt}/{PLAN_ATTEMPTS}).")

    raise RuntimeError(f"Could not get a usable plan: {problem}.")


def write_metadata(title, brief, shots, llm="gemini"):
    """The upload's title, description and tags. A weak title is better than a failed upload, so this can't fail."""
    outline = "\n".join(f"- {shot['title']}" for shot in shots or [])
    prompt = f"""
    Write YouTube metadata for this short AI-generated film.

    Film: {title}
    What it was asked to show: {brief}
    Its shots:
    {outline}

    - title: under 80 characters, true to the film, no clickbait punctuation.
    - description: 2-3 sentences about the film, with no hashtags.
    - tags: 8-12 lowercase search keywords, no "#".
    """
    fallback = {"title": title[:80], "description": brief, "tags": []}
    try:
        metadata = Metadata.model_validate_json(_ask(llm, prompt, Metadata, METADATA_SCHEMA))
    except (RuntimeError, ValidationError) as e:
        print(f"   ⚠️ Could not write the metadata ({e}). Using the film's title.")
        return fallback
    if not metadata.title.strip():
        return fallback
    return {"title": metadata.title.strip(), "description": metadata.description.strip() or brief,
            "tags": [tag.strip() for tag in metadata.tags if tag.strip()]}

import base64
import math
import os
import re
import threading
import time

import anthropic
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

# Who can write the plan: (name, what the panel calls it).
LLMS = (("gemini", "Gemini (free tier)"), ("claude", "Claude Opus 5.5 (paid per use)"))

# Tried in order; each has its own free-tier daily quota, so a used-up model just
# hands over to the next. Override with GEMINI_MODEL (one model or a comma-separated chain).
DEFAULT_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite")
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
# Its default is 'medium'. A film's worth of prompts that have to fit together earns more thought.
CLAUDE_EFFORT = "high"
# Long replies are streamed, so a high ceiling can't run into a request timeout.
CLAUDE_MAX_TOKENS = 64000

# Gemini returns these when it is overloaded or rate-limiting rather than when
# the request itself is wrong, so they are worth waiting out.
RETRYABLE_STATUSES = (429, 500, 502, 503, 504)
# Longest a single retry will sleep. A daily-quota 429 reports delays far past
# anything worth blocking a run on.
MAX_RETRY_WAIT = 90
# A whole film's prompts come back in one reply: about 75 of them at the longest,
# which takes a while. A model that says nothing for this long is overloaded or
# stuck, so move on.
DEFAULT_TIMEOUT = 360

# How long a whole film can be, in seconds: one shortest clip, up to 10 minutes.
MIN_FILM_SECONDS = 5
MAX_FILM_SECONDS = 600
# How long the video model can make a clip, in seconds. Shorter clips can't hold an
# action; longer ones run out of the workflow's memory.
MIN_SHOT_SECONDS = 5
MAX_SHOT_SECONDS = 10
AVERAGE_SHOT_SECONDS = 8
# Too few clips to tell a story with: they show one moment instead.
MIN_SHOTS_FOR_A_STORY = 3
PLAN_ATTEMPTS = 3

CONTINUITY = {
    "chain": """Each clip starts on the exact last frame of the clip before it, so this is one
    continuous take with no cuts. Every shot picks up exactly where the previous one ended. A new
    place can only be reached by moving there on screen: the subject walks, the camera travels.
    End each shot on a clear, steady moment that the next one can continue from.""",
    "image": """Every clip starts again from the attached image, with a fade between clips. So each
    shot is its own moment growing out of that same opening frame: a different action, a different
    camera move, a different development of the idea, in an order that still builds.""",
}


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


def shot_count(seconds):
    """The ceiling matters at 11s, which rounds to one clip that couldn't be long enough."""
    return max(math.ceil(seconds / MAX_SHOT_SECONDS), round(seconds / AVERAGE_SHOT_SECONDS), 1)


def fit_durations(lengths, target):
    """The model's lengths within the clip limits, nudged a second at a time until they add up to the film's."""
    fitted = [min(MAX_SHOT_SECONDS, max(MIN_SHOT_SECONDS, length)) for length in lengths]
    while sum(fitted) != target:
        longer = sum(fitted) < target
        room = [(MAX_SHOT_SECONDS - length) if longer else (length - MIN_SHOT_SECONDS) for length in fitted]
        if not any(room):
            break
        fitted[room.index(max(room))] += 1 if longer else -1
    return fitted


def _instructions(brief, seconds, start):
    count = shot_count(seconds)
    clips = "one clip" if count == 1 else f"{count} consecutive clips"
    shots = "one shot" if count == 1 else f"{count} shots in order"
    shape = (f"Give the film a beginning, a development and a real ending, rather than {count} versions of the same moment."
             if count >= MIN_SHOTS_FOR_A_STORY else "This is too short for a story: show one clear moment that feels complete.")
    together = f"Use {seconds}." if count == 1 else f"All of them together add up to {seconds}."
    continuity = "" if count == 1 else CONTINUITY[start]
    return f"""
    You are directing a {seconds}-second film made of {clips} from an AI video
    model (LTX-2). The attached image is the film's opening frame. Look at it closely: the subject,
    the place, the light, the style.

    The brief, in the words of the person making it: "{brief}"

    The brief is short on purpose. Invent the detail the film needs, in keeping with the brief and
    the image, and never contradicting them. {shape} Bring in no new main characters unless
    the brief asks for them.

    {continuity}

    Return a title for the film, and exactly {shots}. For each shot:
    - title: 2-5 words.
    - seconds: a whole number from {MIN_SHOT_SECONDS} to {MAX_SHOT_SECONDS}. {together}
    - prompt: the text the video model gets, following the rules below.

    The video model sees one prompt and one start frame at a time, and knows nothing about the film:
    - One flowing paragraph in the present tense, 60-120 words.
    - Describe what happens in order, as it can really happen in that shot's seconds. One clear main
      action per shot: a crowded prompt comes out as a blur.
    - Name one camera move, and describe the light and the sound.
    - The start frame already shows how everything looks, so call the subject by the same few words
      every time ("the woman in the red coat") instead of describing it again in full.
    - Sound comes from the scene itself: footsteps, wind, a room. No music, because it would jump
      every time one clip hands over to the next. No speech unless the brief asks for it.
    - No titles, captions, text on screen or logos.
    - Never mention other shots, cuts or the film. Write only what is seen and heard in this clip.
    """


# --- Gemini ---

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
    return genai.Client(api_key=api_key)


def _get_models():
    configured = os.getenv("GEMINI_MODEL")
    if configured:
        return [m.strip() for m in configured.split(",") if m.strip()]
    return list(DEFAULT_MODELS)


def _is_retryable(error):
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    if status in RETRYABLE_STATUSES:
        return True
    # Not every client error exposes a numeric code, so fall back to the text.
    text = str(error)
    return any(str(code) in text for code in RETRYABLE_STATUSES)


def _is_daily_quota(error):
    text = str(error).lower()
    return "per day" in text or "perday" in text


def _retry_delay(error):
    """
    Reads the retryDelay Gemini attaches to a 429. Plain backoff is often
    shorter than what the server actually wants, which just burns the retries.
    """
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(error))
    if not match:
        return None
    return min(float(match.group(1)) + 1, MAX_RETRY_WAIT)


def _call_with_deadline(fn, seconds):
    """
    Runs fn, raising TimeoutError if it has not returned within `seconds`. The
    SDK can sit on a silent request (or a long retry-after) indefinitely, so the
    call runs in a daemon thread that is simply abandoned if it hangs.
    """
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"no response within {seconds}s")
    if "error" in result:
        raise result["error"]
    return result["value"]


def _generate(contents, config, retries=4):
    """
    Calls Gemini, waiting out overload and per-minute rate limits. A model whose
    daily quota is used up, that doesn't answer within GEMINI_TIMEOUT, or that keeps
    failing hands over to the next model in the chain. Raises once every model has failed.
    """
    client = _get_client()
    timeout = int(os.getenv("GEMINI_TIMEOUT") or DEFAULT_TIMEOUT)
    failures = []

    for model in _get_models():
        for attempt in range(retries):
            try:
                return _call_with_deadline(
                    lambda: client.models.generate_content(model=model, contents=contents, config=config), timeout)
            except Exception as e:
                reason = " ".join(str(e).split())[:200]
                if isinstance(e, TimeoutError) or _is_daily_quota(e) or getattr(e, "code", None) == 404:
                    print(f"   ⚠️ {model}: {reason}. Trying the next model...")
                    failures.append(f"{model}: {reason}")
                    break
                if attempt == retries - 1 or not _is_retryable(e):
                    failures.append(f"{model}: {reason}")
                    break
                wait = _retry_delay(e) or 2 ** (attempt + 1)
                print(f"   ⚠️ {model} unavailable (attempt {attempt+1}/{retries}). Retrying in {wait:.0f}s...")
                time.sleep(wait)

    raise RuntimeError("Gemini failed on every model:\n   " + "\n   ".join(failures))


def _ask_gemini(prompt, model, image_jpeg=None):
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=model)
    contents = [prompt]
    if image_jpeg:
        contents.insert(0, types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"))
    return _generate(contents, config).text or ""


# --- Claude ---

def _ask_claude(prompt, schema, image_jpeg=None):
    """
    Returns the reply's JSON text. Asks for the server-side fallback: a request one
    model's safety classifier declines is run on another instead of failing the plan.
    """
    content = [{"type": "text", "text": prompt}]
    if image_jpeg:
        data = base64.standard_b64encode(image_jpeg).decode("utf-8")
        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
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

def plan_film(image_jpeg, brief, seconds, start, llm="gemini"):
    """
    Turns the image and the short brief into a film's shot list. One request for the
    whole film rather than one per clip: the prompts have to know each other, and a
    long film would otherwise spend more than a free model's whole daily quota.
    """
    instructions = _instructions(brief, seconds, start)

    problem = None
    for attempt in range(1, PLAN_ATTEMPTS + 1):
        reply = _ask(llm, instructions, Plan, PLAN_SCHEMA, image_jpeg)
        try:
            plan = Plan.model_validate_json(reply)
        except ValidationError as e:
            problem = f"an unreadable reply ({e.error_count()} errors)"
        else:
            shots = [shot for shot in plan.shots if shot.prompt.strip()]
            # Too few or too many shots can't be fixed by changing their lengths.
            if len(shots) * MIN_SHOT_SECONDS <= seconds <= len(shots) * MAX_SHOT_SECONDS:
                lengths = fit_durations([shot.seconds for shot in shots], seconds)
                return plan.title.strip() or "Untitled", [
                    {"title": shot.title.strip(), "seconds": length, "prompt": shot.prompt.strip()}
                    for shot, length in zip(shots, lengths)]
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

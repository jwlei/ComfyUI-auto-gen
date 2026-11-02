"""
What the LLM is told when a film renders on Wan. Wan reads plain prose, makes no
sound, and has a negative prompt of its own saved in the workflow, so none of the
fields, labels or sound that H3 needs belong in its prompts.
"""
MODEL = "wan"
# Nodes of Wan's own start with this, and the text encoder's loader says so otherwise.
NODE_PREFIX = "Wan"
ENCODER_TYPE = "wan"


def uses_wan(workflow):
    return any(str(node.get("class_type", "")).startswith(NODE_PREFIX) or node.get("inputs", {}).get("type") == ENCODER_TYPE
               for node in workflow.values())


def _what_it_opens_on(images):
    if images:
        return "The attached image is the film's opening frame. Look at it closely: the subject, the place, the light, the style."
    return "There is no image: you decide how everything looks, in keeping with the brief."


def _what_to_write(images):
    if images:
        return """- Wan can already see the start frame, so write how things move rather than what is there. Call the
      subject by the same few words every time ("the woman in the red coat"), and spend the words on the motion: what
      moves, in which direction, how fast and how hard, and what the wind, water or light does meanwhile."""
    return """- Nothing shows Wan how anything looks, so give it all: the subject first, then what it does, the camera, then
      the place. Describe the subject, the place and the light in the same words every time, or the clips won't match."""


def instructions(brief, seconds, count, images, continuity, shortest, longest, second_image=""):
    clips = "one clip" if count == 1 else f"{count} consecutive clips"
    shots = "one shot" if count == 1 else f"{count} shots in order"
    together = f"Use {seconds}." if count == 1 else f"All of them together add up to {seconds}."
    shape = ("This is too short for a story: show one clear moment that feels complete." if count < 3 else
             f"Give the film a beginning, a development and a real ending, rather than {count} versions of the same moment.")
    return f"""
    You are directing a {seconds}-second film made of {clips} from Wan, an AI video model. {_what_it_opens_on(images)}

    The brief, in the words of the person making it: "{brief}"

    The brief is short on purpose. Invent the detail the film needs, in keeping with the brief, and never contradicting
    it. {shape} Bring in no new main characters unless the brief asks for them. Whatever language the brief is in, write
    the prompts in English.

    {second_image}

    {continuity}

    Return a title for the film, and exactly {shots}. For each shot:
    - title: 2-5 words.
    - seconds: a whole number from {shortest} to {longest}. {together}
    - prompt: the text Wan gets, following the rules below.

    Wan sees one prompt at a time, and knows nothing about the film:
    - One flowing paragraph of plain sentences in the present tense, 80-120 words. No labels, headings or lists.
    - Lead with what matters most in the clip, because the start of a prompt counts for more than the end.
    {_what_to_write(images)}
    - One clear main action per shot, sized to really happen in that shot's seconds. Too much comes out as a blur, and too
      little comes out in slow motion, so finish on the action settling.
    - Name one camera move in plain words: pushes in, pulls out, pans left, tilts up, tracks alongside, or stays still.
      Add the shot size and angle where they matter: a close-up, a wide shot, a low angle.
    - Name the look in concrete terms: the light source and its direction (low sun from the left, a neon sign, soft
      overcast), the time of day, and the colour (warm, cool, muted).
    - Describe only what you want to see. What to avoid is already handled elsewhere, and saying "no people" brings people.
    - Wan makes no sound, so don't describe any, and no speech: a mouth can move, but nobody will hear it.
    - No titles, captions, text on screen or logos.
    - Never mention other shots, cuts or the film. Write only what is seen in this clip.
    """

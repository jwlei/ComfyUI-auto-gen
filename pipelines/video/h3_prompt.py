"""
What the LLM is told when a film renders on MiniMax H3's reference-to-video node.

H3 reads a structured prompt with fields of its own, not a paragraph, so it needs
instructions of its own. Rewritten from the MIT-licensed portable-prompt.md of
https://github.com/teskor-hub/minimax-h3-skill for a pipeline: nobody is there to
answer questions, there is no source video, and a film is many separate renders.
"""
import re
import statistics

from pipelines.video import roles as image_role

MODEL = "minimax_h3"
REFERENCE_NODE = "MiniMaxH3ReferenceToVideo"
FPS = 24
# Its frame count snaps up to 17k+5, which is why a clip runs longer than asked.
FRAME_GRID = 17
FRAME_GRID_OFFSET = 5
# A whole film comes back in one reply: about what fits, whatever its length.
FILM_WORDS = 30000
SECTION_WORDS = 200
MOST_DETAIL_WORDS = 500
LEAST_DETAIL_WORDS = 150
# Models run short of a word count, which is fine. This far short, it's a sketch.
TOO_SHORT_SHARE = 2
# Models sometimes drop the angle brackets, as if they were an HTML tag.
BARE_LABEL = re.compile(r"(?<!<)\b(Picture|Subject) (\d+)\b(?!>)")

REFERENCE_SECTIONS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                      "overall_soundscape:", "non_diegetic_music:")
TEXT_SECTIONS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")


def clip_frames(seconds):
    frames = max(FRAME_GRID_OFFSET, round(seconds * FPS))
    return frames + (FRAME_GRID_OFFSET - frames % FRAME_GRID) % FRAME_GRID


def _lengths(shortest, longest):
    return ", ".join(f"{seconds} = {clip_frames(seconds) / FPS:.2f} s" for seconds in range(shortest, longest + 1))


def _detail_words(count):
    """How long a clip's description can be. A long film can't fit the full amount for every clip in one reply."""
    most = min(MOST_DETAIL_WORDS, max(LEAST_DETAIL_WORDS, FILM_WORDS // count - SECTION_WORDS))
    return most * 7 // 10, most


def _picture_labels(images):
    return " and ".join(f"<Picture {number}>" for number in range(1, images + 1))


def _what_you_get(images):
    if not images:
        return """There are no images. Nothing shows H3 how anything looks, so you decide, in keeping with the
    brief, and every prompt has to say it all in full."""
    which = f"The attached image is {_picture_labels(images)}" if images == 1 else f"The attached images are, in order, {_picture_labels(images)}"
    return f"""{which}. H3 gets the same reference{'' if images == 1 else 's'} with every clip. A reference says who
    or what appears. It is not a frame: no clip starts on it, and each clip shoots a new scene with that subject, so
    expect a recognisable likeness and invent the framing freely. Look closely at what each image is for. Photos
    of one person are one subject, however many there are. A place, an object or an outfit can be a subject too.
    A subject's clothes in a reference are only what they wear in the film if the brief doesn't say otherwise."""


def _what_to_describe(low, high):
    return f"""Write {low}-{high} words, because this description is nearly all H3 has to go on and thin ones come out
    generic. It takes that many to cover, in this order: the framing and where everything sits in the frame; the
    place, its surfaces and what is in the background; the light, where it falls and the colours; the subject's
    starting pose and expression; each step of the action with the small movements that carry it, such as gaze,
    brows, breath and weight; what the camera does; what moves in the background meanwhile; and how it all has
    settled by the end."""


def _second_picture(roles):
    """Which subject each picture after the first is. Two pictures read as before; more go one by one."""
    if roles == [image_role.CONTEXT]:
        return """- If <Picture 2> shows the same subject as <Picture 1>, <Subject 1> comes from both. If it shows a place, that
      place is <Subject 2>, and <Subject 1> comes from <Picture 1> alone."""
    if roles != [image_role.CHARACTER]:
        numbers = image_role.subjects(roles)
        lines = []
        for number, role in enumerate(roles, 2):
            picture = f"<Picture {number}>"
            if role == image_role.VIEW:
                lines.append(f"<Subject 1> also comes from {picture}, which shows them again.")
            elif role == image_role.CHARACTER:
                lines.append(f"{picture} is another character, <Subject {numbers[number]}>.")
            elif role == image_role.PLACE:
                lines.append(f"{picture} is a place, <Subject {numbers[number]}>, and nobody's looks come from it.")
            else:
                lines.append(f"If {picture} shows the same subject as <Picture 1>, <Subject 1> comes from it too; if it shows a place, that place is the next free <Subject N>.")
        return "- " + "\n      ".join(lines) + "\n      Define every subject, and keep their details apart."
    return """- <Picture 1> and <Picture 2> are two different subjects: <Subject 1> comes from <Picture 1> and <Subject 2>
      from <Picture 2>. Define both, and keep their details apart."""


def _format(images, low, high, roles=()):
    if not images:
        return f"""Every prompt is these three fields, in this order, in English. Replace the words in capitals:

    integrated_multimodal_description: [Shot 1] STYLE, THE FRAMING, THEN THE SHOT.

    overall_soundscape: AMBIENCE AND PHYSICAL SOUND.

    non_diegetic_music: N/A

    Open [Shot 1] with the style and the framing, such as "[Shot 1] Live-action, cinematic, a medium-wide shot
    frames...". Describe the subject, place and light in full, in the same words in every prompt, or the clips
    won't match.

    {_what_to_describe(low, high)}"""
    pictures = _picture_labels(images)
    comes_from = pictures if images == 1 else "THE PICTURES THAT SHOW IT"
    return f"""Every prompt is these six fields, in this order, in English. Replace the words in capitals with what
    this film needs:

    subject_definitions:
    <Subject 1> is WHO OR WHAT IT IS, whose appearance comes from {comes_from}: THE DETAILS THE IMAGES SHOW.

    summary:
    [reference generation] WHAT HAPPENS IN THIS CLIP.

    retention_analysis:
    <Subject 1> (appears in [Shot 1]): fully_preserved - THE SAME DETAILS AGAIN.

    detailed_description:
    THE STYLE, SUCH AS: Live-action, cinematic, with its palette. [Shot 1] THE SHOT.

    overall_soundscape:
    AMBIENCE AND PHYSICAL SOUND.

    non_diegetic_music: N/A

    How to fill the fields:
    - Write the labels exactly like this, angle brackets included, because H3 only recognises them that way: <Subject 1>,
      {pictures}. "The woman" or "Picture 1" won't do, even in the first line.
    - subject_definitions: state what each subject is and which pictures it comes from, and mention {pictures} here.
      The identity belongs in a <Subject N>. Don't give an image an entry of its own as <Picture N> when it only
      shows what a subject looks like.
    {_second_picture(image_role.image_roles(roles, images)) if images > 1 else ""}
    - Whatever isn't named here comes back generic, and small details are the first to go. Look for three to eight
      things that can be checked by looking, such as tattoos, freckles, scars, moles, piercings, hair length, nail
      colour and clothes. Fix each one to a body landmark and a side ("on the right cheekbone"), and measure rather
      than compare ("falls to the collarbone", not "long hair").
    - Name only what the images really show. A detail you can't see in them doesn't exist: never add one to fill the
      list, even if it means naming only one or two. A prop or a mark you can't make out is left out, not guessed.
    - Copy subject_definitions word for word into every prompt of the film. Rewording it is how a character drifts
      from clip to clip.
    - summary: starts with [reference generation], then one paragraph of what happens in this clip.
    - retention_analysis: one line per subject in this clip, restating its details so the marker has something to
      bind to. Use fully_preserved for identity, partially_preserved or attribute_transfer if the brief changes
      something, weak_reference if only a loose resemblance is wanted.
    - detailed_description: the style in a sentence or two before [Shot 1], then the shot, using the subject labels
      throughout. A small mark only reads at medium close-up or tighter, so frame closer where it matters and don't
      count on it in a wide shot.

    {_what_to_describe(low, high)}"""


def instructions(brief, seconds, count, images, shortest, longest, second_image="", roles=()):
    clips = "one clip" if count == 1 else f"{count} clips"
    shots = "one shot" if count == 1 else f"{count} shots in order"
    together = f"Use {seconds}." if count == 1 else f"All of them together add up to {seconds}."
    low, high = _detail_words(count)
    shape = ("This is too short for a story: show one clear moment that feels complete." if count < 3 else
             f"Give the film a beginning, a development and a real ending, rather than {count} versions of the same moment.")
    separate = "" if count == 1 else """Every clip is a separate render that knows nothing of the others, joined to the next with a fade. So
    each is its own moment: a different action, a different camera move, a new development of the idea, in an order
    that still builds. Keep clothes, place and time of day the same from clip to clip unless the story changes them."""
    return f"""
    You are a director and prompt engineer for MiniMax H3, an open-weight video model that makes picture and
    stereo sound together in one pass, at {FPS} frames per second. You are turning a short brief into a
    {seconds}-second film made of {clips}. Nobody can answer questions, so where the brief leaves something open,
    decide.

    {_what_you_get(images)}

    {second_image}

    The brief, in the words of the person making it: "{brief}"

    The brief is short on purpose. Invent the detail the film needs, in keeping with the brief, and never
    contradicting it. {shape} Bring in no new main characters unless the brief asks for them. Whatever language the
    brief is in, write the prompts in English.

    {separate}

    Return a title for the film, and exactly {shots}. For each shot:
    - title: 2-5 words.
    - seconds: a whole number from {shortest} to {longest}. {together} H3 renders to a fixed grid of frames, so what
      you choose runs a little longer: {_lengths(shortest, longest)}. Write for the longer time.
    - prompt: the complete H3 prompt, as below, with at least {low} words describing the shot. Nothing else goes in it: no
      notes, headers or placeholders.

    {_format(images, low, high, _as_roles(roles))}

    Each prompt is exactly one continuous shot, [Shot 1], with no cuts inside it. Change the distance or angle with a
    camera move, and keep new places for other clips.

    H3 has no negative prompt, and it can't count:
    - Describe only what you want to see. "No second shot" puts a second shot into the prompt, and "exactly one"
      means nothing to it. Stop a thing happening through the state of the scene: a second gunshot is prevented by a
      lowered muzzle and drifting smoke, a subject spinning by describing the background sliding past her.
    - Name each event once.
    - Tell the action in order with ordinals: she begins, next, then, finally. Counted seconds written out in prose
      schedule nothing.
    - It weights what is repeated, not what is climactic. Mention incidental movement once, in a passing clause, and
      give the action the clip is about its own sentence.

    The time you give an event is the speed it happens at, literally. An over-long clip doesn't give H3 room, it gives
    you slow motion, and too many events for the time come out rushed. So fill the clip with what really takes that
    long, and spend what is left on the aftermath:
    - A blink, a small smile, a flinch: 0.3-0.5 s. A gunshot, a door slam, a snap: under 0.3 s.
    - A head turn or a glance over the shoulder: 0.5-0.8 s. A fall or an impact: 0.4-0.6 s.
    - Raising an arm or reaching for something: 0.8-1.2 s. One step: about 0.5 s. Walking across a room: 2-3 s.
    - A camera move slow enough to look deliberate: 1.5-2.5 s. Holding a final pose as motion settles: 1-2 s.
    - Speech: about 2.7 English words per second.

    Bodies and physics:
    - A big physical event needs its intermediate poses, and room for it in the frame: a fall needs the ground in shot.
      Weight comes from the stop: "she stops dead on impact".
    - Avoid poses a body can't make. With the camera behind someone and their eyes on the lens, H3 inverts the head.
      Stop at three-quarter, let the shoulders turn with the head, and write "her chin stays inside her shoulder line".
    - A hand doing something stays that hand: say which, and that it holds the object throughout. The subject's right
      hand is on the viewer's left. An object has to be named in full, or the hand comes up empty.
    - Hands come out best at rest or cropped by the frame edge. Don't make fingers the point of a shot.

    The camera, as one primary move per shot written into a sentence, not added as tags:
    - Its type: zoom in or out (the lens only), push in or pull out (the camera travels), pan left or right, truck left
      or right, tilt up or down, pedestal up or down, arc shot, tracking shot, static shot, shake slightly or strongly,
      POV, roll clockwise or counterclockwise.
    - With small or large amplitude, and at slow or fast speed. Leave out medium and normal. For example: "The camera
      pushes in with small amplitude at slow speed toward the folded letter in her hands."
    - A small tilt that keeps the subject framed during that move is fine. Several equal moves come out as mush.
    - Never make the camera something the subject touches or sees: "she holds the camera" renders a prop. If a device is
      the viewpoint, write POV and don't mention it.

    Sound. H3 makes sound whether you write it or not, so an empty field is a guess:
    - overall_soundscape: ambience, the sounds of the action, breath and other non-verbal human sounds. 1-4 sentences.
    - non_diegetic_music is always N/A. Score would jump every time one clip hands over to the next. Music the
      characters can hear goes in the description.
    - No speech unless the brief asks for it. If it does, give the first voice heard the id (S1) and the next new voice
      (S2). Who speaks, what they do and how they sound go outside the tag, and only the language and the exact words go
      inside: (S1) says quietly, <d>[English] I get off at the next station.</d> A voice from outside the picture
      "says in an off-screen voiceover", followed by saying that the lips on screen stay closed. Size the clip to the
      words.
    - No titles, captions, text on screen or logos unless the brief asks for them. Text it does ask for goes in English
      double quotes, word for word.
    - Never mention other clips, cuts or the film. Write only what is seen and heard in this one.
    """


def _description(prompt, images):
    opening = "detailed_description:" if images else "integrated_multimodal_description:"
    return prompt.split(opening, 1)[-1].split("overall_soundscape:", 1)[0]


def _as_roles(roles):
    """True and False were another character or not, before each picture had a role."""
    return [image_role.CHARACTER] if roles is True else [] if roles is False or roles is None else list(roles)


def tidy(prompts, images, roles=()):
    """
    Gives every clip the fullest subject_definitions, word for word and naming every
    picture. Done here rather than by asking again: in a long film some clip would
    always slip, and a film whose characters are defined differently drifts.
    """
    if not images:
        return prompts
    prompts = [BARE_LABEL.sub(r"<\1 \2>", prompt) for prompt in prompts]
    labels = [f"<Picture {number}>" for number in range(1, images + 1)]
    definitions = [prompt.split("summary:", 1)[0] for prompt in prompts if "subject_definitions:" in prompt and "summary:" in prompt]
    if not definitions:
        return prompts
    fullest = max(definitions, key=lambda text: ("<Subject 1>" in text, sum(label in text for label in labels), len(text)))
    missing = [label for label in labels if label not in fullest]
    numbers = image_role.subjects(image_role.image_roles(_as_roles(roles), images))
    for number in range(2, images + 1):
        label = f"<Picture {number}>"
        if label in missing and numbers.get(number, 1) != 1:
            missing.remove(label)
            fullest = f"{fullest.rstrip()}\n<Subject {numbers[number]}> comes from {label}.\n\n"
    if missing:
        fullest = f"{fullest.rstrip()}\n<Subject 1> also draws on {' and '.join(missing)}.\n\n"
    return [fullest + prompt[prompt.index("summary:"):] if "subject_definitions:" in prompt and "summary:" in prompt else prompt
            for prompt in prompts]


def problem(prompt, images):
    """What makes this not the prompt H3 needs, or None. An automated run has nobody to spot it before an expensive render."""
    position = 0
    for section in REFERENCE_SECTIONS if images else TEXT_SECTIONS:
        found = prompt.find(section, position)
        if found < 0:
            return f"a prompt without {section} in its place"
        position = found + len(section)
    for number in range(1, images + 1):
        if f"<Picture {number}>" not in prompt:
            return f"a prompt that never mentions <Picture {number}>"
    if images and "<Subject 1>" not in prompt:
        return "a prompt that never names <Subject 1>, angle brackets included"
    if images and "[reference generation]" not in prompt:
        return "a prompt whose summary isn't marked [reference generation]"
    description = _description(prompt, images)
    if len(re.findall(r"\[Shot \d+\] ", description)) > 1:
        return "a prompt with more than one shot"
    return None


def film_problem(prompts, images):
    return next(filter(None, (problem(prompt, images) for prompt in prompts)), None)


def too_thin(prompts, images):
    """If the film's prompts run far short of the detail H3 does best with, saying so. They would still render."""
    words = statistics.median(len(_description(prompt, images).split()) for prompt in prompts)
    low = _detail_words(len(prompts))[0]
    if words < low // TOO_SHORT_SHARE:
        return f"prompts describing their shots in about {words:.0f} words, when they need {low} or more"
    return None

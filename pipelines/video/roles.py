"""
What each image after the first is for, chosen in the panel or with --image-role.

context is the old default, where the LLM tells by looking whether an image is more of the
subject or a place. The panel asks instead, so a run started before still means what it did.
"""
from dataclasses import dataclass

CONTEXT, VIEW, CHARACTER, PLACE = "context", "view", "character", "place"
ROLES = (VIEW, CHARACTER, PLACE, CONTEXT)
LABELS = {VIEW: "Same character, new view", CHARACTER: "Another character", PLACE: "Environment"}


def image_roles(roles, images):
    """One role for each of images 2 onwards. A lone role is image 2's, and an image without one is left to the LLM."""
    if not roles:
        roles = []
    elif isinstance(roles, str):
        roles = [roles]
    wanted = max(0, images - 1)
    roles = list(roles)[:wanted]
    return roles + [CONTEXT] * (wanted - len(roles))


def subjects(roles):
    """Which <Subject N> each picture is, as far as its role says: 1 for the subject and more views of it."""
    numbers, free = {1: 1}, 2
    for number, role in enumerate(roles, 2):
        if role == VIEW:
            numbers[number] = 1
        elif role in (CHARACTER, PLACE):
            numbers[number], free = free, free + 1
    return numbers


# --- Reference videos and audio, which a node such as MiniMax H3's takes besides images ---

MOTION, VOICE, MUSIC, SOUND = "motion", "voice", "music", "sound"
VIDEO_ROLES = (MOTION, CHARACTER, PLACE)
AUDIO_ROLES = (VOICE, MUSIC, SOUND)
VIDEO_LABELS = {MOTION: "Movement and camera", CHARACTER: "A character", PLACE: "A place"}
AUDIO_LABELS = {VOICE: "Voice of the main character", MUSIC: "Music", SOUND: "Sound and ambience"}


@dataclass(frozen=True)
class Reference:
    """
    One reference video or audio clip, labelled as H3 numbers them: <Video k> in order, and <Audio j>
    first for each video that has sound of its own (its soundtrack), then for each audio clip.
    """
    kind: str  # "video" or "audio"
    role: str
    label: str
    sound: str = None  # A video's own soundtrack, which H3 labels too


def references(videos=(), audios=()):
    """videos: [(role, has sound)], audios: [role]. Returns their References, videos first."""
    found, sounds = [], 0
    for number, (role, has_sound) in enumerate(videos, 1):
        if has_sound:
            sounds += 1
        found.append(Reference("video", role if role in VIDEO_ROLES else MOTION, f"<Video {number}>",
                               f"<Audio {sounds}>" if has_sound else None))
    for role in audios:
        sounds += 1
        found.append(Reference("audio", role if role in AUDIO_ROLES else SOUND, f"<Audio {sounds}>"))
    return found


def labels(references):
    """Every label the references bring, as a prompt has to mention them."""
    return [label for reference in references or () for label in (reference.label, reference.sound) if label]

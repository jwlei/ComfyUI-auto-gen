"""
What each image after the first is for, chosen in the panel or with --image-role.

context is the old default, where the LLM tells by looking whether an image is more of the
subject or a place. The panel asks instead, so a run started before still means what it did.
"""
CONTEXT, VIEW, CHARACTER, PLACE = "context", "view", "character", "place"
ROLES = (VIEW, CHARACTER, PLACE, CONTEXT)
LABELS = {VIEW: "Same character, another view", CHARACTER: "Another character", PLACE: "Environment: where it happens"}


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

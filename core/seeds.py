"""
The seed each render uses, which the panel shows and lets you set.

A run chosen with a seed gives its first clip or image that seed, the next one the seed after
it, and so on, so each one can be made again exactly. Without one, every render gets a random
seed. Either way the seed a render used is kept, so a good one can be pinned for the next.
"""
import random

# The inputs ComfyUI's samplers take their seed by.
SEED_INPUTS = ("seed", "noise_seed")
MOST = 2 ** 32 - 1


def parse(value):
    """A seed from the form or command line: a whole number from 0 to 4294967295, or None for random. Raises ValueError."""
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "random", "none"):
        return None
    seed = int(text)
    if not 0 <= seed <= MOST:
        raise ValueError(f"A seed is a whole number from 0 to {MOST}.")
    return seed


def nth(base, index):
    """The seed of item index (from 0) in a run chosen with base, or None for random."""
    return None if base is None else (base + index) % (MOST + 1)


def pick(pinned=None, attempt=1):
    """
    The seed for one render. A pinned seed is used as it is, but only on the first attempt: ComfyUI
    gives the same result for the same seed, so a render that was rejected needs a new one.
    """
    return pinned if pinned is not None and attempt == 1 else random.randint(0, MOST)


def apply(workflow, seed):
    """Sets every seed input in the workflow to this seed. Returns whether it has any."""
    found = False
    for node in workflow.values():
        for key in SEED_INPUTS:
            if isinstance(node.get("inputs", {}).get(key), int):
                node["inputs"][key] = seed
                found = True
    return found

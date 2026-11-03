"""
What the LLM is told when a set renders on Fibo 2.0, and how its replies are checked.

Fibo 2.0 was trained on structured JSON captions (Bria's VGL), and reads them far better than prose: in Bria's
own measurements a prose edit saying "keep X unchanged" made it redraw the whole frame and invent things, where
the JSON caption changed only what was asked. So each prompt here is one JSON object: the caption of the finished
image, plus for an edit one imperative edit_instruction. TextEncodeFibo2 takes it as it is.

Rewritten from the edit rewrite prompt in ComfyUI's Fibo 2.0 template, which follows Bria's VGL
(docs.bria.ai/vgl) and their scene analyser (huggingface.co/briaai/fibo-scene-analyzer).
"""
import json
import re

CAPTION = """
    Each prompt is ONE JSON object, written out as the prompt's text: no Markdown fences, nothing around it. It is a
    structured caption of the FINISHED image, with these keys:
    - short_description: what the image shows, in at most 200 words.
    - objects: an array of at most 5 prominent things in it. For each: description (at most 100 words), location
      ("center", "top-left", "bottom-right foreground"), relative_size ("small", "medium", "large within frame"),
      shape_and_color, texture, appearance_details, relationship (to the other objects) and orientation ("upright",
      "facing left"). For a person also pose, expression, clothing, action, gender and skin_tone_and_texture. For a
      group of the same thing also number_of_objects, as a whole number.
    - background_setting: the surroundings, with anything in them that isn't one of the objects.
    - lighting: an object with conditions, direction and shadows.
    - aesthetics: an object with composition (for people, the shot: "medium shot", "close-up"), color_scheme and
      mood_atmosphere.
    - photographic_characteristics: an object with depth_of_field, focus, camera_angle and lens_focal_length. For a
      person, a standard (35-50mm) or portrait (50-85mm) lens, not a wide angle, unless the brief wants one.
    - style_medium: "photograph", "oil painting", "3D render" and so on.
    - artistic_style: only when it isn't a photograph, at most 3 words.
    - context: what kind of image this is, such as "an editorial fashion photograph for a magazine".
    - text_render: only if the brief gives exact words to show, as an array of {text, location, size, color, font,
      appearance_details}. Never invent text. Leave the key out when there is none.
    Leave out any key that doesn't apply rather than writing null or an empty value. Every other key is required."""


def _edit_rules(images):
    pictures = ("" if images == 1 else f"""
    - The {images} images are <image_1>, <image_2> and so on, in the order attached. Say which one you mean that way
      in the edit_instruction, as "Put the jacket from <image_2> on the woman in <image_1>".""")
    return f"""
    An AI image model (Fibo 2.0) will redraw the attached image once per prompt. It sees the image and that prompt,
    and knows nothing about the others. Look at the image closely first: who or what is in it, the pose, the
    framing, the setting, the light, the medium.
    {CAPTION}
    - edit_instruction: also required. One direct command in the imperative, starting with a verb such as Change,
      Add, Remove, Turn, Make or Move: "Change the woman's outfit to a fitted navy wool coat over a cream turtleneck."
      It says everything that changes, and never mentions an original or reference image.

    How to write an edit:
    - The caption describes the finished image, and everything the brief doesn't change comes straight from the
      attached image: the same person, with the same face, build and pose, the same objects in the same places, the
      same framing, camera, light and medium. Change only what the edit needs: a new setting brings its own light.
    - Describe each changed thing as it now IS: "she wears a red coat", never "her coat was changed to red". Outside
      edit_instruction, write no edit words at all: changed, added, removed, replaced, now, instead of, originally.
    - Never write "keep X unchanged", in the caption or the instruction. Describing what stays is what keeps it.
    - If the image is a photograph and the brief doesn't ask for another medium, style_medium is "photograph".{pictures}"""


def _create_rules():
    return f"""
    An AI image model (Fibo 2.0) will make one image per prompt. It sees only that prompt, and knows nothing about the
    others, so each has to describe its image in full. If several images share a subject, describe it in the same
    words in each, or they won't look alike.
    {CAPTION}"""


def rules(images):
    return _edit_rules(images) if images else _create_rules()


def tidy(prompt, editing):
    """The prompt as Fibo 2.0 takes it, compact JSON, or None if it isn't a caption of the kind asked for."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", prompt.strip())
    try:
        caption = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(caption, dict) or not str(caption.get("short_description") or "").strip():
        return None
    if editing and not str(caption.get("edit_instruction") or "").strip():
        return None
    if not editing:
        # Nothing to edit: the node would take this one for an edit of nothing.
        caption.pop("edit_instruction", None)
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))

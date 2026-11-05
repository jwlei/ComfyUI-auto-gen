import copy
import os

import requests

from core.saved_workflows import QWEN_IMAGE_21_ENCODER, _upstream as upstream, image_model, image_takes
from pipelines.image import media, models
from pipelines.video.comfy import ComfyClient

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
TEXT_GENERATORS = ("TextGenerate",)


def watch_text_generation(workflow):
    """
    A PreviewAny on each text generator the picture is made from (the workflow's prompt enhancer), so what it wrote
    comes back in the history. Only one in use: watching one that's switched off would run it for nothing.
    Returns the ids of the nodes added.
    """
    from core.workflow_options import _feeding_outputs
    used = _feeding_outputs(workflow)
    added = []
    for key in [key for key in used if workflow[key].get("class_type") in TEXT_GENERATORS]:
        node_id = f"enhanced_{key}"
        workflow[node_id] = {"class_type": "PreviewAny", "inputs": {"source": [key, 0]}, "_meta": {"title": "What the enhancer wrote"}}
        added.append(node_id)
    return added


def generated_text(entry, watched):
    """What the watched text generators wrote, from a finished run's history entry. None if nothing."""
    texts = []
    for node_id in watched:
        for value in (entry.get("outputs", {}).get(node_id) or {}).get("text") or []:
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return "\n\n".join(texts) or None
EDIT_ENCODER = "TextEncodeQwenImageEditPlus"
# A try, one after freeing ComfyUI's memory, and one a size smaller (Kokon retries smaller rather than failing).
OUT_OF_MEMORY_ATTEMPTS = 3
OUT_OF_MEMORY_SHRINK = 0.8


class ImageClient(ComfyClient):
    """
    Makes images with a workflow from the Image side of the panel, as ComfyClient makes clips. There's
    no built-in one: COMFYUI_WORKFLOW names it, with the '.map.json' core/saved_workflows.py wrote next to it.
    """

    def __init__(self, options=None):
        if not os.getenv("COMFYUI_WORKFLOW"):
            raise RuntimeError("No workflow chosen: save one in ComfyUI named image_<something>, or set COMFYUI_WORKFLOW.")
        # Its options come with the run's flags (--option), and are kept with the run.
        super().__init__(options or {})

    def model(self):
        """The image model it runs, if it's one with prompts or limits of its own (pipelines/image/models.py)."""
        return image_model(self.workflow)

    def takes(self):
        return image_takes(self.fields, self.model(), self.options)

    def render_size(self, width, height):
        """What to have the model make for this size, which gets cropped to it afterwards."""
        if any(node.get("class_type") == EDIT_ENCODER for node in self.workflow.values()):
            return media.edit_render_size(width, height)
        if any(node.get("class_type") == QWEN_IMAGE_21_ENCODER for node in self.workflow.values()):
            # Qwen-Image 2.1 rounds what it's given to whole 32s, and would come out a size that can't be cropped back.
            return media.render_size(width, height, step=32)
        return media.render_size(width, height)

    def render_image(self, prompt, images, width, height, save_as, seed=None, transparent=False):
        """
        Makes one image from these uploaded images, on this seed or a random one, and saves it as save_as plus its
        extension. Returns the path, or raises. transparent: with a transparent background, on a model that makes one.
        Out of memory, it frees ComfyUI's and tries again, then once more a size smaller, enlarged after.
        """
        model = self.model()
        wording = models.TRANSPARENT.get(model) if transparent else None
        sent = wording.format(prompt.strip()) if wording else prompt
        size = (width, height)
        for attempt in range(1, OUT_OF_MEMORY_ATTEMPTS + 1):
            try:
                path = self._render_once(sent, images, *size, save_as, seed)
                break
            except RuntimeError as e:
                if not media.out_of_memory(e) or attempt == OUT_OF_MEMORY_ATTEMPTS:
                    raise
                self.free_memory()
                if attempt == 2 and "width" in self.fields:
                    size = media.render_size(round(width * OUT_OF_MEMORY_SHRINK), round(height * OUT_OF_MEMORY_SHRINK))
                    print(f"      ⚠️ Out of memory again: trying at {size[0]}x{size[1]}, enlarged to {width}x{height} after.")
                else:
                    print("      ⚠️ ComfyUI ran out of memory: freed it, trying again.")
        if size != (width, height):
            media.resized(path, (width, height), path)
        if model in models.TRANSPARENT and not wording:
            media.opaque(path)
        return path

    def _render_once(self, prompt, images, width, height, save_as, seed):
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._fill_images(workflow, images)
        self._set(workflow, "width", width)
        self._set(workflow, "height", height)
        watched = watch_text_generation(workflow)
        entry = self._run(workflow, seed)
        self.enhanced = generated_text(entry, watched)
        return self._download(entry, save_as, IMAGE_EXTENSIONS, "an image")

    def enhance(self, prompt, images, seed=None):
        """
        What the workflow's own prompt enhancer writes from this prompt, on this seed, without making the image: only
        what it reads is run (Kokon's enhance_only). None if the workflow has no enhancer switched on.
        """
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._fill_images(workflow, images)
        watched = watch_text_generation(workflow)
        if not watched:
            return None
        entry = self._run(upstream(workflow, watched), seed)
        return generated_text(entry, watched)

    def free_memory(self):
        """Has ComfyUI unload its models and free what it holds, as its own Free model and node cache does."""
        try:
            requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except requests.RequestException:
            pass

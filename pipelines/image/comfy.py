import copy
import os

from core.saved_workflows import image_model, image_takes
from pipelines.image import media
from pipelines.video.comfy import ComfyClient

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
EDIT_ENCODER = "TextEncodeQwenImageEditPlus"


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
        return media.render_size(width, height)

    def render_image(self, prompt, images, width, height, save_as, seed=None):
        """Makes one image from these uploaded images, on this seed or a random one, and saves it as save_as plus its extension. Returns the path, or raises."""
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._fill_images(workflow, images)
        self._set(workflow, "width", width)
        self._set(workflow, "height", height)
        return self._download(self._run(workflow, seed), save_as, IMAGE_EXTENSIONS, "an image")

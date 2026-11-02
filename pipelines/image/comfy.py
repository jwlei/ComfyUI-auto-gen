import copy
import os

from core.saved_workflows import image_takes
from pipelines.video.comfy import ComfyClient

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


class ImageClient(ComfyClient):
    """
    Makes images with a workflow from the Image side of the panel, as ComfyClient makes clips. There's
    no built-in one: COMFYUI_WORKFLOW names it, with the '.map.json' core/saved_workflows.py wrote next to it.
    """

    def __init__(self):
        if not os.getenv("COMFYUI_WORKFLOW"):
            raise RuntimeError("No workflow chosen: save one in ComfyUI named image_<something>, or set COMFYUI_WORKFLOW.")
        super().__init__()

    def takes(self):
        return image_takes(self.fields)

    def render_image(self, prompt, images, width, height, save_as):
        """Makes one image from these uploaded images and saves it as save_as plus its extension. Returns the path, or raises."""
        workflow = copy.deepcopy(self.workflow)
        self._set(workflow, "prompt", prompt)
        self._fill_images(workflow, images)
        self._set(workflow, "width", width)
        self._set(workflow, "height", height)
        return self._download(self._run(workflow), save_as, IMAGE_EXTENSIONS, "an image")

import copy
import json
import os

from core import workflow_options
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
        super().__init__()
        self.options = self._offered()
        if options:
            problem = workflow_options.check(self.options, options)
            if problem:
                raise RuntimeError(problem)
            self.workflow, self.fields = self._with(options)

    def _saved(self, part):
        with open(os.path.splitext(self.workflow_path)[0] + f".{part}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _offered(self):
        """What a run can change in this workflow, as the panel listed it."""
        with open(os.path.splitext(self.workflow_path)[0] + ".map.json", "r", encoding="utf-8") as f:
            return json.load(f).get("_options") or []

    def _with(self, options):
        """The workflow made again with these options, from the copy kept when it was listed."""
        try:
            api, fields, _, _ = workflow_options.convert(self._saved("ui"), self._saved("info"), options)
        except OSError:
            raise RuntimeError("This workflow's options need it listed again: open the Image side with ComfyUI on.")
        except workflow_options.sw.Unusable as e:
            raise RuntimeError(f"With these options the workflow can't run: {e}")
        return api, {field: tuple(value) for field, value in fields.items() if isinstance(value, list)}

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

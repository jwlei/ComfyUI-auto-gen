# Image

The Image pipeline of [ComfyUI auto-gen](../README.md). It has no upload: images are downloaded, or found in `output/image`.

Makes a set of images, from text alone or by editing an image you give it. An LLM writes one prompt per image from a
short brief, or your own prompt goes in word for word.

```
image of a woman + "recreate her with different outfits and hairstyles" + 15 images
  → Gemini or Claude writes 15 prompts that differ from each other (one request for the whole set)
  → ComfyUI makes each image
  → output/image/<title>_<run>/image_001.png ...
```

## The panel

Press **Image** at the top, or open `/image`.

**Text or edit:** give it an image and it edits that image. Give it none and it makes images from text. Left on
**Automatic**, the workflow follows from that. Choosing one yourself asks for what it takes.

**Prompts:**

| Choice | What each image gets |
|---|---|
| **The LLM writes one per image** | Its own prompt, written from what you say. A sentence or two is enough. They're written together in one request, so they can differ from each other. |
| **Use my prompt, word for word** | Your prompt exactly, on a new seed each time. No LLM, so no key or quota needed. |

**How many images:** 1 to 100.

**Resolution:**

| Setting | Choices |
|---|---|
| **Aspect ratio or screen** | The same as the image; 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9 or 21:9; a desktop (1080p, 1440p, 1440p ultrawide, 4K); a phone (OnePlus 15 at 1272x2772, Android Full HD+ and QHD+, iPhone 16 and 16 Pro Max); or an exact size you type |
| **Megapixels** | The same as the image, or 0.5 to 8. Not asked with a screen or an exact size, which already say. |

Both on "the same as the image" gives you its own resolution, up to 16 megapixels. "Same as the image" with 1 MP keeps
its shape at about 1024x1024's pixel count, which is the default. The hint under them says the size you'll get.
With no image to match, "the same as the image" means square and 1 MP.

The models take sizes in multiples of 16. Sizes worked out from an aspect ratio are rounded to one. A screen or an exact
size comes out exactly that size, pixel for pixel: it renders at the next multiple up and is cropped from the middle, so
1080p renders 1920x1088 and loses four rows off the top and bottom. Above about 4 megapixels these models are well past
what they usually work at, and 4K is 8: expect a long render, and repeated details. The workflows have no upscaler.

An edit comes out the size its source goes in, so the source is fitted to the size you chose first, cropped at its edges
if the shape differs. The original stays whole in the run's folder.

**While it runs:** each image appears in the gallery as it's finished. Generate adds more to the queue, shared with
Shorts and Video. **Stop all** and **Skip, resume later** work as they do on Video. One image that won't come out, after
a second try, doesn't cost the rest of the set: the run ends as failed with the others kept, and **Continue** makes what's missing.

**Generate** continues a stopped set with the same brief, images, count and workflow rather than paying for new prompts.

**Let me read the prompts before rendering**, under How, stops once they're written so you can change them.

**Prompts** on a set shows each image with its prompt. **Make again** makes one again on a new seed, using the prompt in
the box. **Delete** takes one image and its prompt out of the set. **Delete** on the set removes all its images and prompts.

## Workflows

The form lists what you saved in ComfyUI with a name starting `image_`. One that can't run is greyed out, and the hint
under the list says why, such as a model that isn't installed. They're tried in name order on Automatic.

What the pipeline does to a saved workflow, in the copy it runs. Your saved one in ComfyUI is left as it is.

| In the workflow | What happens |
|---|---|
| A subgraph | Opened up, as if you had unpacked it. |
| Save Image, Save Image (Advanced) | Becomes Preview Image, so nothing lands in ComfyUI's output folder. Images are downloaded from its temp folder. |
| A switch set on its face, such as Enable LoRA | Goes the way it's set, for good. A style LoRA that's off needs no file. |
| A refine or enhance prompt switch, such as Krea-2's | Turned off. The LLM has already written the prompt, or you have, so it should reach the model as it is. |
| FluxKontext Image Scale, in Qwen-Image-Edit | Skipped, or it would snap the size you chose to one of its own. |

Other settings stay as saved. So Qwen-Image-Edit's `enable_turbo_mode` decides between its 40-step and 4-step LoRA
routes: change it in ComfyUI, and the next list picks that up. The ETA learns each workflow's time separately.

A workflow's Load Image nodes are its image boxes. Qwen-Image-Edit as saved has one, so it takes one image. Wire more
Load Image nodes to its `image2` and `image3` in ComfyUI and each becomes a required box.

## The command line

```
python -m pipelines.image.main "recreate her with different outfits and hairstyles" --image woman.png --count 15 --llm claude
```

Set `COMFYUI_WORKFLOW` to the workflow it runs: the panel keeps its conversions in `logs/workflows/`.

| Option | What it does |
|---|---|
| `--image FILE` | The image to edit, repeatable if the workflow takes several. |
| `--count` | How many images, 1 to 100. Default 1. |
| `--direct` | Uses the prompt word for word, with no LLM. |
| `--llm` | `gemini` or `claude`. Default: `LLM_PROVIDER`, else `gemini`. |
| `--aspect` | Such as `16:9`, or `image` for the source's (the default, square without one). |
| `--megapixels` | Such as `1.5` (default 1), or `image` for the source's. |
| `--size 1920x1080` | An exact size, in place of those two. |
| `--plan-only` | Writes the prompts and stops. |
| `--resume RUN` | Continues a saved run. |
| `--redo IMAGE` | With `--resume`: makes that image again. |
| `--list` | Lists saved runs. |
| `--check` | Checks the LLM key, the workflow and ComfyUI, and makes nothing. |

The prompts are in `runs/image/<run>/state.json`: edit any there and `--resume` uses it.

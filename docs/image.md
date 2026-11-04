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
**Automatic**, the workflow follows from that: text alone goes to one that takes no image, and an image to one that
needs one. Fibo 2.0, which does both, is Automatic's pick only when nothing else fits, such as two or more images.
Choosing one yourself asks for what it takes.

**Prompts:**

| Choice | What each image gets |
|---|---|
| **The LLM writes one per image** | Its own prompt, written from what you say. A sentence or two is enough. They're written together in one request, so they can differ from each other. |
| **Use my prompt, word for word** | Your prompt exactly, on a new seed each time. No LLM, so no key or quota needed. |

A workflow that writes its own prompts adds them to **Who writes the prompts**, beside Gemini and Claude:

| Choice | What happens |
|---|---|
| **The workflow's own prompt enhancer, in ComfyUI** | Your prompt goes in, and the workflow's enhancer (Krea-2's Refine Prompt, or a Prompt enhancer group) expands it in ComfyUI. Each image's seed reaches the enhancer too, so a set's prompts differ. No LLM is asked. |
| **The workflow, from your image (image to prompt)** | Where a workflow has an Image to prompt group, as LatentHeart's Krea2 does: it takes one image, describes it in ComfyUI and makes new images from that and your prompt. |

Otherwise both stay off: the LLM or you write the prompt.

**How many images:** 1 to 100, or 30 when the LLM writes Fibo 2.0's prompts (see below).

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

**Seed**, under Generation, is empty by default: every image gets a new random seed. The shuffle button picks one to keep, and
✕ goes back to random. With a seed, image 1 uses it and each image after it the next number, so the set can be made
again.

**Prompts** on a set shows each image with its prompt and the seed it was made with. **Make again** makes one again
using the prompt in the box, on that seed unless you change it or press shuffle. An empty seed is new each time. **Delete** takes one image and its prompt out of the set. **Delete** on the set removes all its images and prompts.

## Each model's limits

What each model makes well, from its makers and ComfyUI's templates. The form greys out the sizes and pixel counts past
a model's limit, and Generate refuses them before anything is paid for. "Same as the image" stays within it.

| Model | Size | Prompts | Steps, as saved |
|---|---|---|---|
| **Fibo 2.0** (Bria), edit turbo | Up to 1.1 MP. Trained at about 1 MP, every size within 10% of it: above that the prompt leaks into the top left corner and it stops following it. No desktop or phone screen fits, so those are greyed out. | JSON captions (below). The LLM writes at most 30 a set, as each is a few hundred words. | 4, CFG 1 |
| **Krea-2** Turbo | Up to 4.2 MP: its 1K and 2K sizes, up to 2048x2048. 1440p and the phones fit; 4K and 1440p ultrawide don't. | A descriptive paragraph | 8 |
| **Qwen-Image-Edit** 2511 | Up to the form's own 16 MP | An edit instruction | 40, or 4 with turbo |

**Fibo 2.0's prompts.** It was trained on structured JSON captions (Bria's VGL) and reads them far better than prose: in
Bria's tests an edit written as "keep X unchanged" redrew the whole frame and invented things, where a caption changed only
what was asked. So the LLM writes each prompt as one JSON object describing the finished image (its objects, background,
light, camera and medium) and, for an edit, one imperative `edit_instruction`. One that isn't valid JSON of that kind
doesn't count, and is asked for again. **Prompts** on a set shows them, and you can change them there.
Your own prompt, word for word, can be plain text or a caption of your own.

Fibo 2.0 does both kinds in one workflow: with no image it makes one from text, and with images it edits. It takes up to
16 images, a box at a time: the next box opens once the last holds an image. With several, call them `<image_1>`,
`<image_2>` in your prompt, as in "put the jacket from `<image_2>` on the woman in `<image_1>`". One image is cropped to the
size you chose, as the edit lines up with it pixel for pixel; several are each scaled to the size Fibo 2.0 was trained at for their shape.

## Workflow options

**Workflow options**, on the form, are the chosen workflow's own switches and settings, as it was saved in ComfyUI. Each
run can change them; what you don't change stays as saved. They come from the workflow itself, so any workflow you save
gets them, on Shorts and Video too, where a run keeps them for Continue and redoing a shot:

| In the workflow | On the form |
|---|---|
| A group that an rgthree Fast Groups Bypasser switches, by title or colour | A tickbox: on or off. LatentHeart's Krea2 SFW/NSFW has 2nd pass, Noise (grain), Color correction, Sharpen and the SeedVR2 upscaler. |
| A choice on a switched group's subgraph | Beside its tickbox: the SeedVR2 upscaler's resolution, 720p to 4K. |
| A switch on a subgraph's face, as ComfyUI's templates have | A tickbox, with what goes with it by name beside it: Krea-2's Enable LoRA with the LoRA and its strength, Qwen-Image-Edit's Enable turbo mode. |
| A LoRA in rgthree's Power Lora Loader | A tickbox and its strength. Saved under a folder of the author's, it's found where it's installed. |
| The Krea2T Enhancer | A tickbox and its prompt-adherence strength, 0 to 2. |
| A sampler whose steps can be found | Steps. |

Not offered, as the pipeline does those itself: what writes or enhances the prompt (an Image to prompt or Prompt
enhancer group stays off, as the LLM or you write it), what sets the size, the seed, and which model files load. One that
can't be switched on is greyed out with why, such as a LoRA or node that isn't installed: Color correction needs
ComfyUI_LayerStyle's Brightness Contrast. A run keeps its options for Continue and Make again.

LatentHeart's workflow saves through Image Saver with Civitai metadata, which writes to ComfyUI's output folder and asks
Civitai for model hashes: it becomes Preview Image, like any other save. The SeedVR2 upscaler runs on SDPA attention, as
the workflow's own note says to without sage attention, and its first run downloads its models, about 16 GB. The upscaler
makes the image larger than the size chosen, which is what the model generates at. The LLM writes Krea-2's prompts for
any of these, and may decline an explicit brief: your own prompt, word for word, goes in as written.

## Workflows

**Workflow**, at the top of the form, lists what you saved in ComfyUI for images: a name starting `image_`, or any other that saves or previews images
and no video, such as `Fibo2_turbo_edit_test`. Renamed in ComfyUI, a workflow shows under its new name within half a
minute, still on the Image side; runs made under its old name can't be continued, as its converted copy goes with it. One that can't run is greyed out, and the hint under the list says why, such as a
model that isn't installed. Automatic tries them in name order, those made for what you gave it first.

A model file the workflow names that isn't installed, but is in another folder or another precision, is used as installed:
`krea2_turbo_fp8` runs on `krea2_turbo_fp8_scaled`. Only when just one fits: two candidates, or another model, still say
it isn't installed.

What the pipeline does to a saved workflow, in the copy it runs. Your saved one in ComfyUI is left as it is.

| In the workflow | What happens |
|---|---|
| A subgraph | Opened up, as if you had unpacked it. |
| Save Image, Save Image (Advanced), Image Saver | Becomes Preview Image, so nothing lands in ComfyUI's output folder. Images are downloaded from its temp folder. One that only previews is taken as it is. |
| rgthree's labels, bookmarks and group switches | Left out: they're the editor's alone. What the switches do is in Workflow options. |
| Anything Everywhere | Its links, which the editor saves with the workflow, are made. |
| A switch set on its face, such as Enable LoRA | Goes the way it's set, for good. A style LoRA that's off needs no file. |
| A growing list of image inputs, such as Fibo 2.0's | Gets a Load Image for each, up to the node's own limit. Those left empty come out again. |
| A refine or enhance prompt switch, such as Krea-2's | Turned off. The LLM has already written the prompt, or you have, so it should reach the model as it is. |
| FluxKontext Image Scale, in Qwen-Image-Edit | Skipped, or it would snap the size you chose to one of its own. Its encoder wraps the image around if it comes out an odd number of latent pixels across, which leaves a band of the far side along one edge, so these render at the nearest size where it doesn't and are cropped to yours. |

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
| `--seed N` | Image 1's seed, and the next number for each image after it. Default: a random seed for every image. |
| `--plan-only` | Writes the prompts and stops. |
| `--resume RUN` | Continues a saved run. |
| `--redo IMAGE` | With `--resume`: makes that image again. |
| `--list` | Lists saved runs. |
| `--check` | Checks the LLM key, the workflow and ComfyUI, and makes nothing. |

The prompts are in `runs/image/<run>/state.json`: edit any there and `--resume` uses it.

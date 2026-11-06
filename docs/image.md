# Image

The Image pipeline of [ComfyUI auto-gen](../README.md). It has no upload: images are downloaded, or found in `output/image`.

Makes a set of images, from text alone or by editing an image you give it. An LLM writes one prompt per image from a
short brief, or your own prompt goes in word for word.

```
image of a woman + "recreate her with different outfits and hairstyles" + 15 images
  → Gemini or Claude writes 15 prompts that differ from each other (one request for the whole set)
  → ComfyUI makes each image
  → output/image/<title>_<workflow>/<title>_<workflow>_001.png ...
```

## The panel

Press **Image** at the top, or open `/image`.

**Text or edit:** give it an image and it edits that image. Give it none and it makes images from text. Left on
**Automatic**, the workflow follows from that: text alone goes to one that takes no image, and an image to one that
needs one. Fibo 2.0, which does both, is Automatic's pick only when nothing else fits, such as two or more images.
Choosing one yourself asks for what it takes.

**Prompts:**

Use **Write prompt** beside the prompt field to rewrite an idea locally with the model selected in Settings. On
success the text is replaced and Exact prompt is selected when available. It needs an idle, reachable ComfyUI and
temporarily unloads its models; queued panel generations wait. Your text is kept if it fails.
The writer receives only the text in the prompt box, not attached images. Its image instructions focus on a
single composition, spatial relationships, lighting and other visible details.
Alternatively, select **Local model (Settings)** under **Prompts → LLM** to write a separate prompt for each image
with the workflow's existing planning instructions. Gemma can analyze attached images; Qwen rejects them because
it is text-only. ComfyUI must be reachable and idle while the local model plans; it unloads before rendering.
Large sets may exceed the local model's context or output limits.

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

For any image input a workflow accepts, use the gallery button beside the file chooser to select an image from
Output instead. This includes additional reference images and the image-to-prompt input.

When you delete an image in the preview, it stays open on the next image in that batch, or the previous
one if there is no next image. Deleting the only remaining image closes the preview.

In Output, tap the three-dot button twice at the top of an image batch or Video card to mark it NSFW. It disappears
from normal output and Finished. Choose **NSFW**, just below **All workflows**, to see tagged Image
and Video output together and tap the button twice again to restore it. Image batches remain selectable
as inputs through that filter. Videos from the same saved run, including upscaled copies, share a tag.
Shorts are unaffected. This is a display filter, not access control: the media stays on disk.
Each time you enter Output, the filter starts at **All workflows**.

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
Shorts and Video. **Stop** and **Cancel** work as they do on Video and require two taps to confirm. One image that won't come out, after
a second try, doesn't cost the rest of the set: the run ends as failed with the others kept, and **Continue** makes what's missing.

**Generate** continues a stopped set with the same brief, images, count and workflow rather than paying for new prompts.

**Let me read the prompts before rendering**, under How, stops once they're written so you can change them. With the
workflow's own prompt enhancer it works too: ComfyUI runs only the enhancer, one prompt per image on that image's seed,
and the run stops there. Continue makes them as written, with the enhancer off (Kokon's "read the prompt first").

**Seed**, under Generation, is empty by default: every image gets a new random seed. The shuffle button picks one to keep, and
✕ goes back to random. With a seed, image 1 uses it and each image after it the next number, so the set can be made
again.

**Prompts** on a set shows each image with its prompt and the seed it was made with. When the workflow's enhancer wrote a
detailed prompt, the sparkle button opens it over the image with Copy and Close at the bottom. **Make again** makes one again
using the prompt in the box, on that seed unless you change it or press shuffle. An empty seed is new each time. **Delete** takes one image and its prompt out of the set. **Delete** on the set removes all its images and prompts.

## What else it does with images

These follow [Kokon](https://github.com/GabrieleLozupone/kokon), a local image editor on Qwen-Image 2.1, and work with
the Qwen-Image 2.1 edit workflow. The form offers each only when the workflow chosen (or one Automatic can pick) can.

**Edit**, in the header beside Image, opens the Image side with the Qwen-Image 2.1 edit workflow
(`image_Qwen21_edit_turbo`) chosen; **Edit**, under an image in the viewer, does the same with that image as image 1.

**Images:** one box, and **+** opens the next, up to what the workflow takes: Qwen-Image 2.1 edits image 1 and takes up
to 3 more, called `<image2>`, `<image3>` in the prompt. Each box takes a file, or one of your own images: the grid button
opens Output, and a tap on an image there puts it in the box. Then the seed that made it goes along, and the edit never
uses it again (seed below).

**What to do with the images**, once one is chosen:

| Choice | What happens |
|---|---|
| Edit image 1 | As before: it comes out the size you chose, image 1 fitted to it. |
| Make a new picture from them | The images are references (a person, a product, a place, a style) for a new picture at the size under Resolution: the template's Custom size, on. The LLM is told to borrow from them and never to keep one unchanged, which would make it an edit of image 1. Up to 2.2 MP on 12 GB. |
| Change painted parts of image 1 | **Paint what to change** opens Inpaint, full screen, with image 1 and icons under it to paint, erase and **Select**: a click on a thing paints all of it, Shift-click takes it away (MobileSAM, on the CPU). The wheel or two fingers zoom in for fine detail, and Space or the middle button drags the picture; the brush keeps its size on screen, so zoomed in it paints finer. One colour is one change: with one, Remove (the first choice, needing no words), Change, Replace or Add; with several, say what goes in each colour, under **Colour prompts**. **Save** keeps it, and **Generate** makes it. Only what's painted changes. |
| Expand image 1 | Left, top, right and bottom, in pixels or percent of the image. The prompt says what the new parts show, or left empty, the model continues what it sees. |

How the painted parts and Expand work, as Kokon does it: the model can't be given a mask, so it's shown where to work.
One area: it works on the part of the image around it, at about a megapixel, with what's removed or replaced blurred away
first (so it isn't drawn back), and a second, smaller copy with the area marked in a colour the prompt names. Several:
each painted in its colour on image 1 itself. Expand: the image on a larger canvas, the new borders its average colour.
Whatever it makes, only the area or the new borders are kept: everything else is the original, pixel for pixel, through
a feathered edge (`pipelines/image/regions.py`).

**Transparent background**, on Qwen-Image 2.1: the prompt is wrapped as ComfyUI's template says to, and the PNG keeps
its alpha. Otherwise Qwen-Image 2.1's alpha, which it always decodes (with faint holes, about 1% of pixels), is dropped.

**How the other images are shown:** Standard shows images 2 and on at about a megapixel each, less if they'd crowd the
card, leaving room for the result; Full shows them at their own size.

**Seeds:** an edit made on the seed that made the image it starts from begins from the same noise, and comes out
over-sharpened, ignoring the prompt (Kokon measured it on Qwen-Image 2.1). The seed is read from your image's PNG, or sent
with one chosen from your images; one that would repeat it is swapped for another, the same each time, and the log says so.

**Out of memory**, ComfyUI is told to free its memory and the image is tried again, then once more a size smaller and
enlarged after, rather than failing.

**Upscale**, under an image in the viewer: **Fast** is Superscale-SP, in seconds, and adds no new detail; its strength
mixes it with a plain resize, to tone down texture it invents. **SeedVR2** takes minutes and draws finer detail. Either is
a new image in the set. The button is only there while the image's shorter side is 1440px or less.

Select runs on the CPU, so the GPU stays with ComfyUI. Its model (MobileSAM, 44 MB) is in `models/`:
`python scripts/get_models.py` downloads it, checked against its SHA-256.

**LoRAs:** one made for an earlier Qwen-Image (1.x, 2512 or Edit) loads on 2.1 without an error and does nothing. It's
greyed out on a Qwen-Image 2.1 workflow with why, told apart by its layers from the file's header (`core/lora_check.py`).

## Each model's limits

What each model makes well, from its makers and ComfyUI's templates. The form greys out the sizes and pixel counts past
a model's limit, and Generate refuses them before anything is paid for. "Same as the image" stays within it.

| Model | Size | Prompts | Steps, as saved |
|---|---|---|---|
| **Fibo 2.0** (Bria), edit turbo | Up to 1.1 MP. Trained at about 1 MP, every size within 10% of it: above that the prompt leaks into the top left corner and it stops following it. No desktop or phone screen fits, so those are greyed out. | JSON captions (below). The LLM writes at most 30 a set, as each is a few hundred words. | 4, CFG 1 |
| **Krea-2** Turbo | Up to 4.2 MP: its 1K and 2K sizes, up to 2048x2048. 1440p and the phones fit; 4K and 1440p ultrawide don't. | A descriptive paragraph | 8 |
| **MOSS Image** | No model-specific size limit is enforced by the panel; larger sizes may need more memory. | A descriptive prompt | As saved in the workflow |
| **Qwen-Image-Edit** 2511 | Up to the form's own 16 MP | An edit instruction | 40, or 4 with turbo |
| **Qwen-Image 2.1** Turbo, 12 GB | 3.2 MP from text; 1.6 MP for an edit, which is shown its image at the same size beside it; 2.2 MP from references. Native 2K needs 16 GB. These are Kokon's measurements of what fits on 12 GB. | An instruction or a paragraph. Images are `<image1>`, `<image2>`. | 8 (its own schedule), CFG 1 |

The imported MOSS text-to-image workflow is offered once its saved diffusion model, text encoder and VAE are installed
in ComfyUI. The panel uses its prompt, seed and requested size, with steps, CFG and negative prompt under workflow options.

**Fibo 2.0's prompts.** It was trained on structured JSON captions (Bria's VGL) and reads them far better than prose: in
Bria's tests an edit written as "keep X unchanged" redrew the whole frame and invented things, where a caption changed only
what was asked. So the LLM writes each prompt as one JSON object describing the finished image (its objects, background,
light, camera and medium) and, for an edit, one imperative `edit_instruction`. One that isn't valid JSON of that kind
doesn't count, and is asked for again. **Prompts** on a set shows them, and you can change them there.
Your own prompt, word for word, can be plain text or a caption of your own.

Fibo 2.0 does both kinds in one workflow: with no image it makes one from text, and with images it edits. It takes up to
16 images, a box at a time: **+** opens the next once the last holds an image. With several, call them `<image_1>`,
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
| A sampler whose steps can be found | Steps, under Advanced. |
| A KSampler, KSamplerSelect, BasicScheduler, CFGGuider or CFG primitive | Beside Steps, under Advanced: Sampler, Scheduler and CFG, wherever the workflow has just one of each. A distilled graph (ManualSigmas) has its own fixed steps and CFG, so they show greyed out. |
| A switch such as turbo that changes the steps | Steps, CFG, Sampler and Scheduler follow it as it's ticked: Qwen-Image-Edit's Enable turbo mode, MiniMax's lightning LoRA. |
| One text input that is the negative prompt (CLIPTextEncode into a negative, or a Fibo 2 negative_prompt) | Negative prompt, under Advanced. Not offered on a distilled graph, whose CFG of 1 never reads it, unless it has NAG (LTX). |
| An LTX graph | Frames per second, Image strength (both passes), NAG scale, and Add a LoRA (any installed LoRA and its strength, into an empty Power Lora Loader), under Advanced. |

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
The panel hides the `image_`, `video_` and `shorts_` categorization prefixes in displayed workflow names;
the original names remain unchanged in ComfyUI.

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
routes: change it in ComfyUI, and the next list picks that up. The ETA learns each model, workflow, set of options and size separately (`core/eta_keys.py`).

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
| `--enhance-only` | With the workflow's prompt enhancer on (`--option prompt_enhancer=1`): it writes each prompt in ComfyUI, and the run stops there. |
| `--compose` | The images are references for a new picture at the size asked for. |
| `--areas FILE` | Changes painted parts of image 1: JSON with `mask` (a PNG painted a colour per area) and `areas` (`colour`, `action`, `text`). |
| `--expand L,T,R,B` | Expands image 1 by these many pixels; the prompt says what the new parts show. |
| `--transparent` | A transparent background, on Qwen-Image 2.1. |
| `--reference-detail full` | Shows images 2 and on at their own size. Default `standard`, about 1 MP. |
| `--source-seed N` | A seed that made one of the images, repeatable: an edit never uses it. |
| `--resume RUN` | Continues a saved run. |
| `--redo IMAGE` | With `--resume`: makes that image again. |
| `--list` | Lists saved runs. |
| `--check` | Checks the LLM key, the workflow and ComfyUI, and makes nothing. |

The prompts are in `runs/image/<run>/state.json`: edit any there and `--resume` uses it.

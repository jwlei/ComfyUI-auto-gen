# Video

The Video pipeline of [ComfyUI auto-gen](../README.md). It has no YouTube upload: films are downloaded, or found in `output/video`.

Turns one image and a short prompt into a film, from a single 5-second clip up to 10 minutes. An LLM looks at the image and writes a detailed
prompt per shot, ComfyUI renders each shot as a clip, and ffmpeg joins them. It runs from a web panel or the
command line.

```
image + "a storm rolls in over the harbour"
  → Gemini or Claude writes the shot list (one request for the whole film)
  → ComfyUI renders each shot (LTX-2.3: 5-10 seconds a clip, MiniMax H3: 5-15, Wan: 3-5)
  → ffmpeg joins the clips into output/<title>_<workflow>.mp4 (the workflow without video_; LTX23_I2V_T2V for the built-in)
  → optionally, YouTube
```

## Setup

You need Python 3.10+, ffmpeg on the PATH, and ComfyUI with the models and custom nodes that
`workflows/video_workflow.json` uses (the same ones as yt-auto-shorts' avatar workflow).

```
pip install -r requirements.txt
copy .env.example .env
```

In `.env`, set at least one LLM key:

| Setting | What it's for |
|---|---|
| `GEMINI_API_KEY` | Writing the prompts with Gemini, on its free tier. |
| `ANTHROPIC_API_KEY` | Writing the prompts with Claude Opus 5.5, paid per use. Keys come from platform.claude.com. |
| `LLM_PROVIDER` | Which one to use when you don't choose: `gemini` (default) or `claude`. |
| `COMFYUI_LAUNCHER` | Lets the panel start ComfyUI when a render needs it. |
| `APP_PASSWORD` | Needed to reach the panel from your phone. Without it, it only answers this PC. |
| `YOUTUBE_*` | Uploading. Run `scripts/authorize_youtube.py` once and paste what it prints. |

## The panel

Double-click **ComfyUI auto-gen** on the desktop. To create that shortcut:

```
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
```

Or run it yourself and open http://localhost:8801:

```
python webapp.py
```

**Making a film:** choose an image, say what should happen, and pick a length. By default the panel stops once
the prompts are written and opens them for you. Change any of them, then press **Render the film**. Untick
"Let me read the prompts" to go straight through.

**Write prompt:** enter an idea and press the button beside the prompt field to rewrite it locally using the model
selected in Settings. On success it replaces your text and selects Exact prompt. It needs an idle, reachable ComfyUI
and temporarily unloads its models; queued panel generations wait until the writer finishes. The text is kept if it fails.
The writer receives only the text in the prompt box, not attached references or the selected duration. Its video
instructions describe shot composition, motion, sound and user-supplied dialogue and text; it cannot inspect references.

Alternatively, under **Prompts → LLM**, choose **Local model (Settings)** to plan a separate prompt per clip using
the existing workflow-specific planner. Gemma can analyze attached images; Qwen rejects them because it is text-only.
The local model unloads before rendering and needs an idle, reachable ComfyUI. Large plans can take substantial
time and may exceed the local model's context or output limits.

**OpenH3-IR (MiniMax H3).** The Prompt section offers this optional compiler on supported H3 workflows. It is off
by default. Choose **before review** to compile each planned prompt before editing it (using the original reference
images), or **before each clip** to compile with the actual start frame, including a frame from the previous clip.
The latter happens after prompt review; its resulting brief is saved with the shot. Pick **Restrained**, **Balanced**
(default), **Bold**, or **Extreme** for how much the compiler may introduce. Your written constraints still apply.

The compiler runs locally through the installed [ComfyUI-OpenH3-IR](https://github.com/ruashots/ComfyUI-OpenH3-IR)
pack and its `open-h3-ir` dependency in ComfyUI's Python. It starts llama.cpp for the compile and stops it before
rendering to release GPU memory; it refuses to unload ComfyUI's models if another ComfyUI job is running or queued.
Compiling can take several minutes per clip and uses substantial RAM. The original renderer, model, and settings stay
in place; H3 director workflows use H3's 24-fps frame grid for enhanced clips. Reference bindings are checked before
rendering, and duplicate reference files are refused rather than ambiguously assigned.
Motion Director is supported when the panel has adapted its inputs; manually wired Director timelines are not.

Choose the OpenH3-IR model in Settings: **Gemma 4 12B** is the default and supports images and reference videos;
**Qwen3 14B** is text-only and rejects visual references. This selection applies to subsequent compilations.
The GGUF models live in the project's ignored `models/gemma-4-12b-heretic` and `models/qwen3-14b-abliterated`
directories. Gemma also requires its `mmproj.gguf`.

For the portable Windows ComfyUI layout, the defaults locate `python_embeded/python.exe` and
`ComfyUI/custom_nodes/ComfyUI-OpenH3-IR` beside `COMFYUI_LAUNCHER`, and expect the Vulkan llama.cpp
at `1_Misc/llama.cpp/b12177-vulkan/llama-server.exe` beside the ComfyUI directory. Other layouts can set
`H3IR_COMFY_ROOT`, `H3IR_PYTHON`, `H3IR_LLAMA_SERVER`, `H3IR_MODEL_FILE`, and `H3IR_PROJECTOR_FILE`
in `.env`; `H3IR_GPU_LAYERS` defaults to 16. Model-file overrides apply to the currently selected model.
From the command line use `--h3-enhance review` or `--h3-enhance render`, and optionally
`--h3-creativity restrained|balanced|bold|extreme`.

**Every image has a box.** Image 1 is the first frame, or the main character in a reference workflow such as
MiniMax H3, which starts no clip on it. Each image after it is optional, and you say what it is:

| The image is | What the LLM does with it |
|---|---|
| **Same character, new view** (default) | More details of the subject to get right: another angle, the rest of an outfit. |
| **Another character** | It gives them a real part in the film, alongside the subject. |
| **Environment** | It's where the film happens: the subject goes there. Several places and the film moves between them. |

If your prompt says how to use an image, that wins. On H3 each character and place becomes a `<Subject>` of its own.

You can upload an image or use the gallery button in any box to choose one of your generated images from Output.
You start with one box, and **+** below it opens the next once it holds an image. ✕ removes an image and the ones after
it move up. It goes up to 20 boxes, or as many as the workflow has Load Image nodes if that's fewer. A reference node
whose inputs grow is different: wire one or two references in ComfyUI and the pipeline adds Load Image nodes up to 20,
or the node's own limit if that's lower, which is 9 on MiniMax H3. The boxes you leave empty come out of the workflow
again.

**Reference video and audio.** On MiniMax H3 the form also has boxes for reference videos and audio clips. Every clip
gets them, as it gets the images. H3 takes up to 3 videos and 3 audio clips, and 12 references in all, images
included: with 9 images there is room for 3 more of any kind. Each one is 2-15 seconds. A longer one is cut to its
first 15 seconds, and a video is turned to 24 fps, the rate H3 reads it at. Choose what each one is for:

| A video is | What the LLM does with it |
|---|---|
| **Movement and camera** (default) | The action and the camera follow it. |
| **A character** | Someone in the film, the next `<Subject>`. |
| **A place** | Where the film happens. |

| An audio clip is | What the LLM does with it |
|---|---|
| **Voice of the main character** | `<Subject 1>` speaks in that voice. |
| **Music** | The film's music, in `non_diegetic_music`. |
| **Sound and ambience** (default) | The soundscape follows it. |

The LLM can't watch or hear them, so it only knows what you chose and what your prompt says. It calls them
`<Video 1>` and `<Audio 1>`, as H3 does. A video with sound of its own brings that too, as an `<Audio>` of its own,
numbered before the audio clips. Every prompt has to name each one, or the missing ones are added.

**How long a clip is** depends on the video model, and the LLM is held to it:

| Video model | A clip | Why |
|---|---|---|
| LTX-2.3 (LTX23_I2V_T2V, and any other) | 5-10 s | About as long as it holds up. |
| MiniMax H3 | 5-15 s | It was trained on 124-362 frames at 24 fps. |
| Wan 2.2 | 3-5 s | It was trained on 81 frames at 16 fps. |

The film length under the prompt says how many clips that comes to on the workflow you chose.

A workflow with one image input, like LTX23_I2V_T2V, renders from image 1 alone. Only the LLM sees image 2 there, so
it reaches the clips as words in the prompts: expect a description of that place or person, not a copy of it.
A reference workflow gives image 2 to every clip.

**Seed**, under Generation, is empty by default: every clip gets a new random seed. The shuffle button picks one to keep, and ✕
goes back to random. With a seed, clip 1 uses it, clip 2 the next number and so on, so the same prompts make the
same film again. **Shots** shows the seed each clip was made with, and rendering it again keeps that seed unless you
change it or press shuffle, so a changed prompt changes only what the prompt does. An empty seed is new each time. A seed you set is used once per render: a clip that
fails its check is tried again on a new one.

**Resolution** offers 360p, 480p, 540p and 720p, in landscape 16:9 or portrait 9:16. The default and highest preset
is 720p (1280x736) in the shape of image 1, and **Custom size** takes anything else. The sizes are multiples of 32,
so they're as near 16:9 as the models allow. An image of another shape is fitted to the size, usually by cropping
its edges. The ETA scales with the pixel count.

**While it runs:** you see each step with its ETA, which clip is rendering, ComfyUI's own progress, GPU use and the
log. Generate adds more films to a queue, and each keeps the options it was queued with.
The setup button on a queued generation moves it to Pending while you edit its settings and inputs.
To reorder queued jobs, select an entry to reveal its drag handle; drag it or use the arrow keys.
Tap the Queue heading to collapse or expand it. Queue, Pending and Failed scroll internally when their
lists exceed three cards or the available screen height.
Queue and Pending entries show their resolution and a shortened prompt. Finished items open their output when available;
failed items open and highlight their entry under Failed. A failure before a run was saved has no saved run to retry.
Generate keeps the prompt after queuing so you can make variants. Use **Clear** beside Prompt to empty it, or leave
Generate or switch sides to clear it automatically; the other settings stay in place.
**Save to queue** replaces it at the front of the queue; **Cancel editing** leaves the original in Pending.
Upscale jobs use their own settings dialog and follow the same Pending and queue replacement behavior.
Continued runs cannot currently be edited this way.
For a saved run in Pending, the settings icon opens its setup in Generate for a new or matching run;
the original stays in Pending. The prompts icon also opens the saved prompts for a continued job in
the queue or Pending.
While a continuation is in Pending, its prompts and seeds can be edited; save them before starting it.
It still reserves the saved run so it cannot be queued twice or discarded while pending.

**Stop** stops the run and cancels its clip in ComfyUI; queued jobs wait until the queue is resumed.
**Cancel** stops the run and starts the next queued job, leaving the run under Pending. Both require a second tap
to confirm. **Continue** brings the cancelled run back behind whatever is queued by then.

**Waiting to be rendered** lists plans not rendered yet and runs that were stopped, failed or cut off by a crash.
**Continue** picks up from the last finished clip, and only the clip that was cut off starts again.

**Fixing a shot:** press **Shots** on a film to watch each clip. Change a prompt if you like and press **Render
again**. In a continuous take every shot after it renders again too, because each starts on the last frame of the
one before.

**Open folder** shows the tab's folder in File Explorer. It's only there on this PC, not on your phone.

**Delete** removes the film along with its clips and prompts. **Shut down** force-stops everything and closes
ComfyUI if the panel started it. Queued jobs come back on hold next time.

### Two ways to connect clips

| | What each clip starts on | Between clips | Redoing a shot |
|---|---|---|---|
| **One continuous take** (default) | The last frame of the clip before | Nothing: it flows | Also redoes every shot after it |
| **Separate moments** | Your image | A short fade | Leaves the others alone |

If a continuous take drifts away from how your image looks, try separate moments.

**Workflow options**, under Generation, are the chosen workflow's own switches and settings, such as a sampler's steps
or a switched group, as the [Image side](image.md#workflow-options) has them. A run keeps them, so the rest of its clips and
any shot made again come out the same way. The MiniMax H3 workflows also offer **H3 Memory Optimization** and **H3 Sparse
Attention** (with its video attention budget) under Advanced, when the [H3-Optimizations](https://github.com/Zironic/H3-Optimizations)
custom node is installed.

**`video_Minimax_H3_T2V_I2V_R2V`** is the Motion Director workflow, which makes text, image and reference clips of H3 in one
workflow. The panel gives its Director one clip at a time: no image is text-to-video, one image is a start frame
(image-to-video) and more are references. Its prompt and nine image boxes are wired in when it's listed, and the Director's own
timeline and refine pass are left out. **Turbo (4-step LoRA)** switches the LoRA and the steps together. The saved workflow
has Sage attention off (it isn't installed) and the DiffAid sparse patch off (it fails under ComfyUI's dynamic VRAM; H3 Sparse
Attention above is the one to use). Reference clips (two or more images) currently fail inside the Director node itself, which
calls ComfyUI's reference node with its arguments in an older order, until that node is updated.
**Upscale a film.** Open a film in the Video tab and press **Upscale** to queue it through NVIDIA RTX Video Super Resolution
(the [RTX VSR Long Video](https://github.com/tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video) custom node; needs an RTX GPU). Pick a
scale (1.5 to 4×) and a mode. The result is saved beside it as `<name>_upscaled.mp4`, with the sound kept; ComfyUI's own copy of it
is removed. **Prompts** in the same viewer opens the shots of the run that made the film.

## Time and cost

Rendering takes about 30 minutes per minute of film on the GPU this was built on, so a 10-minute film takes around
5 hours. The ETA learns from each run, per model, workflow, workflow options and size. ComfyUI's queue is shared, so other work there takes its turn between
clips.

A film's prompts take one LLM request, however long the film is, and an upload's title and description take one
more. Claude's cost depends on the film's length and is shown on your account at platform.claude.com.

## The command line

```
python -m pipelines.video.main harbour.jpg "a storm rolls in over the harbour" --duration 3m --llm claude
```

| Option | What it does |
|---|---|
| `--duration` | Film length: `90`, `3m` or `2m30`. From 5 seconds to 10 minutes, default `2m`. It's split into clips as long as the video model makes them. |
| `--llm` | `gemini` or `claude`. Default: `LLM_PROVIDER`, else `gemini`. |
| `--start` | `chain` for one continuous take (default), `image` for separate moments. |
| `--reference IMAGE` | Another image, repeatable. The first one is image 2. |
| `--ref-video FILE`, `--ref-audio FILE` | A reference video or audio clip every clip gets, repeatable, on a workflow that takes them such as MiniMax H3. |
| `--ref-video-role`, `--ref-audio-role` | What each one is for, in order: `motion`, `character` or `place` for a video, `voice`, `music` or `sound` for audio. |
| `--image-role` | What an image after the first is, once per image in order from image 2: `view` (more of the subject), `character`, `place`, or `context` to let the LLM tell by looking (the default). `--image-2-role` still works. |
| `--size 1280x736` | Render size, rounded to multiples of 32. By default it takes the image's shape at the workflow's pixel count. |
| `--seed N` | Clip 1's seed, and the next number for each clip after it. Default: a random seed for every clip. |
| `--plan-only` | Writes the prompts and stops, so you can read them before spending the GPU time. |
| `--upload`, `--privacy` | Uploads when the film is done. Private by default. |
| `--resume RUN` | Continues a saved run. |
| `--redo SHOT` | With `--resume`: renders that shot again. In a chain, every shot after it too. |
| `--list` | Lists saved runs. |
| `--check` | Checks ffmpeg, the LLM key and ComfyUI, and makes nothing. |

Ctrl+C cancels the clip in ComfyUI, and `--resume` continues. The plan is in `runs/<run>/state.json`: edit any
`prompt` there and `--resume` uses it.

`python publish.py <film>` or `python publish.py --all` uploads what's in `output/`.

## Using another workflow

**Workflow**, at the top of the form, lists what you saved in ComfyUI, except those ending in `_shorts`, which are tuned for yt-auto-shorts, and those starting `image_`, which are for [Image](image.md). The panel converts the one you pick into API
format when it lists them, and finds its prompt, image, length and size inputs from the graph. **LTX23_I2V_T2V**
stays the default and works as before.

The form asks for what the chosen workflow can take:

| Workflow | Images | Without one |
| --- | --- | --- |
| I2V | 1, required, plus image 2 for the LLM | Can't run |
| I2V/T2V with a switch titled "text to video" | 0-1, plus image 2 for the LLM | The switch goes to text |
| TI2V, where the image plugs into an optional input | 0-1, plus image 2 for the LLM | The image comes out of the workflow |
| T2V, with no Load Image | None | Always from text |
| R2V, with several Load Image nodes | Up to that many, one box each | The empty ones come out, if they're optional |
| MiniMax H3 | Up to 9 images, 3 videos and 3 audio clips, 12 in all | The empty ones come out |

Without an image, the LLM describes everything in full in every prompt and the film is 1280x736, unless you set
a size. With references, every clip gets all of them and the LLM calls them `<Picture 1>`, `<Picture 2>`. A T2V or
R2V workflow can't start a clip on the last frame of the one before, so its clips are separate moments.

A workflow using MiniMax H3's reference-to-video node gets prompts written for H3 instead: its six labelled
fields, the subjects' details named and repeated word for word in every clip, actions sized to how long they really
take. The pipeline adds a Load Video (split into frames and sound) and a Load Audio for each of the node's
`ref_videos` and `ref_audios` that nothing in the saved workflow is wired to. `modules/h3_prompt.py` holds those instructions, rewritten from
[minimax-h3-skill](https://github.com/teskor-hub/minimax-h3-skill) (MIT). A reply without those fields is asked
for again rather than rendered.

A Wan workflow gets prompts written for Wan: a plain paragraph of how things move, with no sound, since it makes
none. Its negative prompt stays what you saved in the workflow. `modules/wan_prompt.py` holds those instructions.

It works out which input is the prompt by following the sampler's positive input back to its text, or from a
node's own `prompt` input. For length it wants a number node with "seconds" or "duration" in its title, or a
`length` input next to `width` and `height`. For size, integer nodes titled "width" and "height", or those same
inputs. A workflow that can't run stays in the list, greyed out, and the hint says why: a model or node that isn't
installed. Subgraphs are opened up as if you had unpacked them. A saved `SaveVideo` becomes VideoHelperSuite's Video Combine, the only way to keep its
clips out of ComfyUI's output folder.

A run keeps its workflow when you continue it or redo a shot. The conversions sit in `logs/workflows/`, so with
ComfyUI off the panel still lists the ones it had last time, and a run starts it.

From the command line, set `COMFYUI_WORKFLOW` to a workflow saved with ComfyUI's **Export (API)**. Next to it, put
a file with the same name ending in `.map.json`, which says which node inputs to fill (see
`workflows/video_workflow.map.json`). `prompt` and `image` are required.

Clips are kept out of ComfyUI's output folder: they render to its temp folder and are downloaded from there.

## Files

```
webapp.py             the panel
main.py               the run: plan, render, stitch, resume
publish.py            uploads what's in output/
start_webapp.pyw      what the shortcut runs
modules/planner.py    writes the shot list and upload text, with Gemini or Claude
modules/h3_prompt.py  what the LLM is told when a film renders on MiniMax H3
modules/wan_prompt.py what the LLM is told when a film renders on Wan
modules/comfy.py      talks to ComfyUI
modules/media.py      images, frames and joining clips, with ffmpeg
modules/run_state.py  what each run has finished
modules/run_eta.py    estimates, learned from earlier runs
modules/publisher.py  uploads to YouTube
modules/saved_workflows.py  turns the workflows saved in ComfyUI into what the pipeline runs
comfyui/              the workflow and which of its inputs get filled
runs/<run>/           a run's state, input image and clips, kept until you delete the film
output/               finished films
uploaded/             films that went to YouTube
inputs/               images the panel received, until a run has its own copy
logs/                 ETA timings, saved queues, the panel's log, converted workflows
```

## Tests

```
python -m unittest discover tests
```

The panel runs the real `main.py` as a real process, with real ffmpeg on tiny synthetic clips, against a fake
ComfyUI and a fake Claude on localhost. They use no GPU, no quota and no money, and everything they make goes to a
temporary folder. The page's script is checked with Node if it's installed.

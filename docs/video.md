# Video

The Video pipeline of [ComfyUI auto-gen](../README.md). It has no YouTube upload: films are downloaded, or found in `output/video`.

Turns one image and a short prompt into a film, from a single 5-second clip up to 10 minutes. An LLM looks at the image and writes a detailed
prompt per shot, ComfyUI renders each shot as a clip, and ffmpeg joins them. It runs from a web panel or the
command line.

```
image + "a storm rolls in over the harbour"
  → Gemini or Claude writes the shot list (one request for the whole film)
  → ComfyUI renders each shot (LTX-2.3: 5-10 seconds a clip, MiniMax H3: 5-15, Wan: 3-5)
  → ffmpeg joins the clips into output/<title>_<run>.mp4
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

**Every image has a box.** Image 1 is the first frame, or the main character in a reference workflow such as
MiniMax H3, which starts no clip on it. Each image after it is optional, and you say what it is:

| The image is | What the LLM does with it |
|---|---|
| **Same character, new view** (default) | More details of the subject to get right: another angle, the rest of an outfit. |
| **Another character** | It gives them a real part in the film, alongside the subject. |
| **Environment** | It's where the film happens: the subject goes there. Several places and the film moves between them. |

If your prompt says how to use an image, that wins. On H3 each character and place becomes a `<Subject>` of its own.

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
| LTX-2.3 (Built-in LTX, and any other) | 5-10 s | About as long as it holds up. |
| MiniMax H3 | 5-15 s | It was trained on 124-362 frames at 24 fps. |
| Wan 2.2 | 3-5 s | It was trained on 81 frames at 16 fps. |

The film length under the prompt says how many clips that comes to on the workflow you chose.

A workflow with one image input, like Built-in LTX, renders from image 1 alone. Only the LLM sees image 2 there, so
it reaches the clips as words in the prompts: expect a description of that place or person, not a copy of it.
A reference workflow gives image 2 to every clip.

**Resolution** offers 360p, 480p, 540p and 720p, in landscape 16:9 or portrait 9:16. The default and highest preset
is 720p (1280x736) in the shape of image 1, and **Custom size** takes anything else. The sizes are multiples of 32,
so they're as near 16:9 as the models allow. An image of another shape is fitted to the size, usually by cropping
its edges. The ETA scales with the pixel count.

**While it runs:** you see each step with its ETA, which clip is rendering, ComfyUI's own progress, GPU use and the
log. Generate adds more films to a queue, and each keeps the options it was queued with.

**Stop all** kills the run and cancels its clip in ComfyUI, leaving anything else you've queued there alone. What's
queued in the panel waits for you to press Resume. **Skip, resume later** does the same but starts the next film in
the queue straight away. **Continue** brings the skipped one back, behind whatever is queued by then.

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

## Time and cost

Rendering takes about 30 minutes per minute of film on the GPU this was built on, so a 10-minute film takes around
5 hours. The ETA learns from each run. ComfyUI's queue is shared, so other work there takes its turn between
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

**ComfyUI workflow** in the panel lists what you saved in ComfyUI, except those ending in `_ytautoshort`, which are tuned for yt-auto-shorts, and those starting `image_`, which are for [Image](image.md). The panel converts the one you pick into API
format when it lists them, and finds its prompt, image, length and size inputs from the graph. **Built-in LTX**
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

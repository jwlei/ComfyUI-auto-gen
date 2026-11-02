# ComfyUI auto-gen

Makes videos with ComfyUI from one panel, with two pipelines behind a **Shorts | Video** switch.

| | Shorts | Video |
|---|---|---|
| Makes | Vertical YouTube Shorts: AI script, narration, stock footage and a lip-synced presenter | A film of up to 10 minutes from an image and a prompt, whose shot prompts you can read first |
| Audio | Always narrated | None, an uploaded file, or text to speech |
| Workflows | Those saved in ComfyUI ending in `_ytautoshort`, which need a Load Audio node | Every other workflow saved in ComfyUI |
| Publishing | Uploads to YouTube | Download only |
| Details | [docs/shorts.md](docs/shorts.md) | [docs/video.md](docs/video.md) |

They share one queue, because there is one GPU, and a job started on either side shows on both. They also share one
record of API use: when a provider says a model is used up for the day, every run moves to the next model until the
reset that reply gave (`logs/exhausted_models.json`).

Nothing is paid for twice. **Generate** continues stopped work that matches: on Shorts the video on the topic you typed,
or the oldest with the topic empty; on Video the same brief, images, audio and length. The options on the form apply,
and only what they change is done again.

## Setup

You need Python 3.11+, ffmpeg on the PATH, and ComfyUI with the models your workflows use.

```
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`. `APP_PASSWORD` lets you reach the panel from a phone. `COMFYUI_LAUNCHER` lets the panel start ComfyUI.

## Running it

```
powershell -ExecutionPolicy Bypass -File scripts/create_shortcut.ps1
```

That makes a **ComfyUI auto-gen** shortcut, which starts the panel and opens it at http://localhost:8800. `/shorts` and
`/video` open each side directly. **Shut down** in the panel stops it. Or run `python webapp.py --host 0.0.0.0 --port 8800`.

**From your phone, anywhere:** install [Tailscale](https://tailscale.com/download) on this PC and your phone and sign
in to the same account on both. Starting the panel also starts Tailscale and reconnects it if it was off. **More** in
the panel shows the address to open on your phone, `http://100.x.y.z:8800`, which stays the same, so it's worth
bookmarking. Tailscale encrypts the connection, and only your own devices can reach it. `python webapp.py` on its own
only listens to this PC and leaves Tailscale alone: add `--host 0.0.0.0`.

Each pipeline also runs alone: `python -m pipelines.shorts.main --help` and `python -m pipelines.video.main --help`.

## Layout

```
core/         what both pipelines use: Gemini calls, API usage and quota, voice, saved workflows
pipelines/    shorts/ and video/, each with its own command line, runs and ETA
panel/        the web app: server.py is shared, shorts.py and video.py hold each side's options
workflows/    the built-in workflows
docs/         each pipeline in detail
logs/         usage, errors, topics and timings
runs/         saved progress, by pipeline
output/       shorts/ready, shorts/uploaded, shorts/bad and video
```

## Tests

```
python -m unittest discover tests
```

They use a fake ComfyUI and Claude on localhost, real ffmpeg, and a throwaway folder. Nothing costs anything.

## History

This joins two earlier projects, kept as archives: [auto-yt-shorts](https://github.com/jwlei/auto-yt-shorts) and auto-comfyui.

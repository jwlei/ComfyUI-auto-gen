import argparse
import os
import re
import shutil
import sys
import time

import ffmpeg
from dotenv import load_dotenv

from modules import media, paths, planner
from modules.comfy import ComfyClient
from modules.run_state import STAGES, RunState, list_runs

CLIP_ATTEMPTS = 2
# Clip render time per second of video, from the yt-auto-shorts runs on this GPU.
RENDER_SECONDS_PER_SECOND = 31


def parse_duration(text):
    """'90', '90s', '3m' or '2m30' as seconds, within the film limits."""
    match = re.fullmatch(r"(?:(\d+)m)?(\d+)?s?", text.strip().lower())
    if not match or not any(match.groups()):
        raise argparse.ArgumentTypeError(f"'{text}' is not a duration. Use 90, 3m or 2m30.")
    seconds = int(match.group(1) or 0) * 60 + int(match.group(2) or 0)
    if not planner.MIN_FILM_SECONDS <= seconds <= planner.MAX_FILM_SECONDS:
        raise argparse.ArgumentTypeError(f"The film has to be {planner.MIN_FILM_SECONDS}-{planner.MAX_FILM_SECONDS} seconds long, got {seconds}.")
    return seconds


def parse_size(text):
    match = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"'{text}' is not a size. Use 1280x736.")
    return media.on_grid(int(match.group(1))), media.on_grid(int(match.group(2)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Turns one image and a short prompt into a film of up to 10 minutes, made of ComfyUI clips.")
    parser.add_argument("image", nargs="?", help="The image the film opens on.")
    parser.add_argument("--no-image", action="store_true", help="Make the film from text alone, if the workflow can: give only the prompt.")
    parser.add_argument("--reference", action="append", default=[], metavar="IMAGE", help="Another image, for a workflow that takes several.")
    parser.add_argument("prompt", nargs="?", help="What should happen, in a sentence or two.")
    parser.add_argument("--duration", type=parse_duration, default="2m", help="Film length: 90, 3m or 2m30 (default 2m, from 5 seconds to 10 minutes).")
    parser.add_argument("--start", choices=("chain", "image"), default="chain",
                        help="Start each clip on the last frame of the one before (chain, default), or on the input image.")
    parser.add_argument("--llm", choices=[name for name, _ in planner.LLMS], default=None,
                        help="Who writes the prompts (default: LLM_PROVIDER in .env, else gemini).")
    parser.add_argument("--size", type=parse_size, help="Render size such as 1280x736 (default: the image's shape).")
    parser.add_argument("--plan-only", action="store_true", help="Write the prompts and stop before rendering.")
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube when the film is done.")
    parser.add_argument("--privacy", choices=("private", "unlisted", "public"), default="private", help="Privacy for --upload.")
    parser.add_argument("--resume", metavar="RUN", help="Continue a saved run, with its original options.")
    parser.add_argument("--redo", type=int, metavar="SHOT", help="With --resume: render this shot again (chained: and every shot after it).")
    parser.add_argument("--list", action="store_true", help="List saved runs.")
    parser.add_argument("--check", action="store_true", help="Check ffmpeg, the LLM key and ComfyUI, and make nothing.")
    args = parser.parse_args(argv)

    if args.redo is not None and not args.resume:
        parser.error("--redo needs --resume.")
    if args.plan_only and args.resume:
        parser.error("--plan-only writes a new plan: --resume already has one.")
    if args.no_image:
        args.image, args.prompt = None, args.image
    if not (args.resume or args.list or args.check or (args.prompt and (args.image or args.no_image))):
        parser.error("give an image and a prompt, --no-image and a prompt, or --resume RUN, --list or --check.")
    return args


def fmt_time(seconds):
    minutes = round(seconds / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes} min"


def print_stage(name, stages):
    print(f"▶ Stage {stages.index(name) + 1}/{len(stages)}: {name}")


def print_runs():
    runs = list_runs()
    if not runs:
        print("No saved runs.")
    for run in runs:
        done = "finished" if run.final else f"{len(run.clips)}/{len(run.shots or [])} clips"
        print(f"{run.id}  {run.seconds}s  {done}  {run.title or run.brief}")


def check_setup(args):
    """Each thing a run needs, without making anything."""
    llm = args.llm or planner.default_llm()
    problems = []
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("✅ ffmpeg found.")
    else:
        problems.append("ffmpeg isn't installed, or isn't on the PATH.")
    if planner.llm_configured(llm):
        print(f"✅ {llm} has credentials.")
    else:
        problems.append(f"{llm} has no credentials: see .env.example.")
    try:
        comfy = ComfyClient()
        print(f"✅ Workflow loaded: {os.path.basename(comfy.workflow_path)}")
        if comfy.wait_until_reachable():
            print(f"✅ ComfyUI is running at {comfy.url}.")
        else:
            problems.append(f"ComfyUI isn't running at {comfy.url}.")
    except (RuntimeError, ValueError) as e:
        problems.append(str(e))
    for problem in problems:
        print(f"❌ {problem}")
    return 1 if problems else 0


def preflight(needs_comfy):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg isn't installed, or isn't on the PATH.")
    if needs_comfy:
        comfy = ComfyClient()
        if not comfy.wait_until_reachable():
            raise RuntimeError(f"ComfyUI isn't running at {comfy.url}. Start it first: nothing has been spent yet.")


def plan_run(args, stages):
    images = [media.load_image(path) for path in ([args.image] if args.image else []) + args.reference]
    takes = ComfyClient().takes()
    if not takes["min_images"] <= len(images) <= takes["max_images"]:
        raise RuntimeError(f"This workflow takes {takes['min_images']}-{takes['max_images']} images, not {len(images)}.")
    # A clip can only start on the last one's final frame if an image is all it takes.
    start = args.start if takes["max_images"] == 1 else "image"
    width, height = args.size or media.render_size(*(images[0].size if images else media.DEFAULT_SHAPE))
    llm = args.llm or planner.default_llm()
    if not planner.llm_configured(llm):
        raise RuntimeError(f"{llm} has no credentials: see .env.example.")
    state = RunState.new(args.prompt, args.duration, start, width, height, llm, args.privacy if args.upload else None,
                         os.getenv("COMFYUI_WORKFLOW") or None, len(images))
    print(f"🗂️ Run folder: {state.id}")

    print_stage("Plan", stages)
    print(f"🧠 {llm} is writing {planner.shot_count(state.seconds)} prompts for a {state.seconds}s film at {width}x{height}...")
    title, shots = planner.plan_film([media.preview_jpeg(image) for image in images], args.prompt, state.seconds, state.start, llm)
    for image, path in zip(images, state.image_paths):
        media.save_input(image, path)
    state.update(title=title, shots=shots)

    print(f"🎞️ {title}")
    for number, shot in enumerate(shots, 1):
        print(f"\n{number}. {shot['title']} ({shot['seconds']}s)\n   {shot['prompt']}")
    print(f"\n⏳ Rendering should take about {fmt_time(state.seconds * RENDER_SECONDS_PER_SECOND)}.")
    return state


def clip_images(state, index):
    if state.start == "chain" and index > 0:
        return [media.last_frame(state.clip_path(index - 1), state.frame_path)]
    return state.image_paths


def render_shot(state, comfy, index, images):
    """Renders until the file holds up, on a new seed each time. Returns its path, or None."""
    shot = state.shots[index]
    save_as = os.path.join(state.clips_dir, f"shot_{index + 1:03d}")
    for attempt in range(1, CLIP_ATTEMPTS + 1):
        try:
            clip = comfy.render(shot["prompt"], images, shot["seconds"], state.width, state.height, save_as)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"      ❌ {e} (attempt {attempt}/{CLIP_ATTEMPTS}).")
            continue
        problem = media.clip_problem(clip, shot["seconds"])
        if not problem:
            return clip
        print(f"      ⚠️ Rejected: {problem} (attempt {attempt}/{CLIP_ATTEMPTS}).")
    return None


def render_clips(state, comfy):
    """Returns True once every shot has an accepted clip."""
    total = len(state.shots)
    time_spent = film_rendered = 0
    for index, shot in enumerate(state.shots):
        if index in state.clips:
            print(f"   ♻️ Clip {index + 1}/{total} kept from earlier.")
            continue

        started = time.time()
        print(f"   🎬 Clip {index + 1}/{total} · {shot['title']} · {shot['seconds']}s")
        try:
            images = [comfy.upload(path, f"autocomfyui_{state.id}{'' if number == 1 else f'_{number}'}.png")
                      for number, path in enumerate(clip_images(state, index), 1)]
        except (OSError, ffmpeg.Error) as e:
            print(f"   ❌ Could not give ComfyUI the start frame: {e}")
            return False
        clip = render_shot(state, comfy, index, images)
        if not clip:
            print(f"   ❌ Clip {index + 1} failed.")
            return False
        state.keep_clip(index, os.path.basename(clip))

        time_spent += time.time() - started
        film_rendered += shot["seconds"]
        left = sum(s["seconds"] for i, s in enumerate(state.shots) if i not in state.clips)
        print(f"      ✅ Done, about {fmt_time(left * time_spent / film_rendered)} left.")
    return True


def stitch_film(state):
    print("🧵 Stitching the film...")
    clips = [state.clip_path(i) for i in range(len(state.shots))]
    name = re.sub(r"[^\w-]+", "_", state.title).strip("_")[:60] or "film"
    output = os.path.join(paths.OUTPUT_DIR, f"{name}_{state.id}.mp4")
    media.stitch(clips, state.parts_dir, output, state.width, state.height, state.start == "chain")
    state.update(final=output)
    return output


def run(args):
    if args.list:
        print_runs()
        return 0
    if args.check:
        return check_setup(args)

    if args.resume:
        state = RunState.load(args.resume)
        if not state or not state.shots:
            raise RuntimeError(f"No saved run called '{args.resume}'. --list shows them.")
        # The rest of the film has to match the clips already made.
        if state.workflow:
            os.environ["COMFYUI_WORKFLOW"] = state.workflow
        if args.redo is not None:
            if not 1 <= args.redo <= len(state.shots):
                raise RuntimeError(f"This film has shots 1-{len(state.shots)}, not {args.redo}.")
            state.redo(args.redo - 1)
        if state.final:
            print(f"✅ Already finished: {state.final}")
            return 0
        # Only stitching left needs no ComfyUI. Saved after this, or a ComfyUI that's
        # down would cost a redo's clips without rendering anything.
        preflight(needs_comfy=state.next_stage() == "Clips")
        state.save()
        print(f"🗂️ Run folder: {state.id}")
        print(f"🎞️ Continuing {state.title}")
        stages = STAGES if state.upload else STAGES[:-1]
    else:
        preflight(needs_comfy=not args.plan_only)
        if args.plan_only:
            stages = STAGES[:1]
        else:
            stages = STAGES if args.upload else STAGES[:-1]
        state = plan_run(args, stages)
        if args.plan_only:
            print(f"📝 Change any prompt in the panel or {state.path}, then: python main.py --resume {state.id}")
            return 0

    print_stage("Clips", stages)
    if state.next_stage() == "Clips":
        comfy = ComfyClient()
        if args.resume:
            # A run that was killed outright leaves its last clip rendering.
            comfy.cancel()
        if not render_clips(state, comfy):
            print(f"💾 Kept what's done. Continue with: python main.py --resume {state.id}")
            return 1

    print_stage("Stitching", stages)
    output = stitch_film(state)
    print(f"✅ {media.duration(output):.0f}s film saved: {output}")

    if state.upload:
        print_stage("Upload", stages)
        # Imported here so a run that doesn't upload never needs the Google client library.
        from modules.publisher import publish_film
        if not publish_film(output, state.upload):
            print("⚠️ The film is fine and waits in output/. Upload it from there.")
            return 1
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(paths.PROJECT_ROOT, ".env"))
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. A saved run continues with --resume (python main.py --list shows them).")
        return 130
    except (RuntimeError, OSError) as e:
        print(f"❌ {e}")
        return 1
    except ffmpeg.Error as e:
        print(f"❌ ffmpeg failed: {(e.stderr or b'').decode('utf8', errors='replace')[-800:]}")
        print("💾 Kept what's done. --resume tries again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

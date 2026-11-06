import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

import ffmpeg
import requests
from dotenv import load_dotenv

from core import h3_refiner, seeds
from core import workflow_options
from core.audio import AudioEngine
from core.paths import LOGS_DIR
from core.saved_workflows import shorthand, title_words, unique_name
from pipelines.video import media, paths, planner, roles
from pipelines.video.comfy import ComfyClient
from pipelines.video.run_state import STAGES, RunState, list_runs

# The built-in workflow's short name in a film's file name.
BUILT_IN = "LTX23_I2V_T2V"

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


def parse_seed(text):
    try:
        return seeds.parse(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a seed: use a whole number from 0 to {seeds.MOST}, or random.")


def parse_size(text):
    match = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"'{text}' is not a size. Use 1280x736.")
    return media.on_grid(int(match.group(1))), media.on_grid(int(match.group(2)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Turns one image and a short prompt into a film of up to 10 minutes, made of ComfyUI clips.")
    parser.add_argument("image", nargs="?", help="The image the film opens on.")
    parser.add_argument("--no-image", action="store_true", help="Make the film from text alone, if the workflow can: give only the prompt.")
    parser.add_argument("--reference", action="append", default=[], metavar="IMAGE",
                        help="Another image. The first is image 2: if the clips only take one image, it goes to the LLM alone.")
    parser.add_argument("--image-role", "--image-2-role", dest="image_roles", action="append", choices=planner.IMAGE_ROLES, default=[],
                        help="What an image after the first is, once per image from image 2: view (more of the subject), "
                             "character (someone else), place (where it happens), or context (the LLM tells by looking, the default)."),
    parser.add_argument("--ref-video", action="append", default=[], metavar="FILE",
                        help="A reference video every clip gets (2-15 s), on a workflow that takes them such as MiniMax H3. Repeatable.")
    parser.add_argument("--ref-video-role", action="append", choices=roles.VIDEO_ROLES, default=[],
                        help="What a reference video is for, once per video in order: motion (movement and camera, the default), "
                             "character or place.")
    parser.add_argument("--ref-audio", action="append", default=[], metavar="FILE",
                        help="A reference audio clip every clip gets (2-15 s), on a workflow that takes them such as MiniMax H3. Repeatable.")
    parser.add_argument("--ref-audio-role", action="append", choices=roles.AUDIO_ROLES, default=[],
                        help="What a reference audio clip is for, once per clip in order: voice (the main character's), music, "
                             "or sound (ambience, the default).")
    parser.add_argument("prompt", nargs="?", help="What should happen, in a sentence or two.")
    parser.add_argument("--duration", type=parse_duration, default="2m",
                        help="Film length: 90, 3m or 2m30 (default 2m, from 5 seconds to 10 minutes). Each clip is as long as "
                             "the video model makes them: 5-10 s on LTX, 5-15 s on MiniMax H3, 3-5 s on Wan.")
    parser.add_argument("--start", choices=("chain", "image"), default="chain",
                        help="Start each clip on the last frame of the one before (chain, default), or on the input image.")
    parser.add_argument("--llm", choices=[name for name, _ in planner.LLMS], default=None,
                        help="Who writes the prompts (default: LLM_PROVIDER in .env, else gemini).")
    parser.add_argument("--size", type=parse_size, help="Render size such as 1280x736 (default: the image's shape).")
    parser.add_argument("--audio", metavar="FILE", help="An audio file for the film: its length sets the film's, and a workflow with Load Audio gets it per clip.")
    parser.add_argument("--narration", metavar="TEXT", help="Text to voice with Gemini TTS, used the same way as --audio.")
    parser.add_argument("--voice", help="Gemini TTS voice for --narration, such as en-gb-tutor-13.")
    parser.add_argument("--seed", type=parse_seed, help="The first clip's seed. Each clip after it gets the next one, so the film "
                                                       "can be made again. Default: a random seed for every clip.")
    parser.add_argument("--plan-only", action="store_true", help="Write the prompts and stop before rendering.")
    parser.add_argument("--direct", action="store_true", help="Every clip gets the prompt as it is: no LLM writes them.")
    parser.add_argument("--h3-enhance", choices=h3_refiner.TIMINGS, default="off",
                        help="Use OpenH3-IR before prompt review or before each render.")
    parser.add_argument("--h3-creativity", choices=h3_refiner.CREATIVITY, default="balanced",
                        help="OpenH3-IR creativity (default: balanced).")
    parser.add_argument("--clip-seconds", type=parse_clip_seconds, default=None, metavar="MIN-MAX",
                        help="How long each clip is, such as 5-8, within what the video model makes (its own range otherwise).")
    parser.add_argument("--first-only", action="store_true", help="Render the first clip and stop, to see it before rendering the rest.")
    parser.add_argument("--resume", metavar="RUN", help="Continue a saved run, with its original options.")
    parser.add_argument("--redo", type=int, metavar="SHOT", help="With --resume: render this shot again (chained: and every shot after it).")
    parser.add_argument("--list", action="store_true", help="List saved runs.")
    parser.add_argument("--check", action="store_true", help="Check ffmpeg, the LLM key and ComfyUI, and make nothing.")
    args = parser.parse_args(argv)

    if args.redo is not None and not args.resume:
        parser.error("--redo needs --resume.")
    if args.plan_only and args.resume:
        parser.error("--plan-only writes a new plan: --resume already has one.")
    if args.audio and args.narration:
        parser.error("--audio and --narration both give the film its audio: choose one.")
    if len(args.ref_video_role) > len(args.ref_video) or len(args.ref_audio_role) > len(args.ref_audio):
        parser.error("--ref-video-role and --ref-audio-role go with a --ref-video or --ref-audio each.")
    if args.no_image and args.reference:
        parser.error("--reference adds to an image, and --no-image has none.")
    if args.no_image:
        args.image, args.prompt = None, args.image
    if not (args.resume or args.list or args.check or (args.prompt and (args.image or args.no_image))):
        parser.error("give an image and a prompt, --no-image and a prompt, or --resume RUN, --list or --check.")
    return args


def fmt_time(seconds):
    minutes = round(seconds / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes} min"


def parse_clip_seconds(text):
    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", text or "")
    if not match or not 1 <= int(match.group(1)) <= int(match.group(2)) <= 60:
        raise argparse.ArgumentTypeError(f"'{text}' is not a clip length: use MIN-MAX in seconds, such as 5-8.")
    return int(match.group(1)), int(match.group(2))


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


def voiced_narration(args):
    """The text as audio. Kept by what it says and who reads it: an abort must not pay for the same voice twice."""
    name = media.audio_fingerprint(text=args.narration, voice=args.voice)
    path = os.path.join(LOGS_DIR, "voice_cache", f"{name}.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("🎙️ Voicing the narration...")
    return AudioEngine(voice=args.voice, output_dir=os.path.dirname(path)).voice_text(args.narration, path)


def reference_roles(args):
    """(path, role) for each reference video, then each reference audio clip. One without a role gets the first."""
    videos = [(path, (args.ref_video_role + [roles.MOTION] * len(args.ref_video))[number])
              for number, path in enumerate(args.ref_video)]
    audios = [(path, (args.ref_audio_role + [roles.SOUND] * len(args.ref_audio))[number])
              for number, path in enumerate(args.ref_audio)]
    return videos, audios


def check_references(takes, images, videos, audios):
    """Raises if the workflow can't take this many reference videos, audio clips or references of every kind."""
    if len(videos) > takes.get("max_videos", 0):
        raise RuntimeError(f"This workflow takes {takes.get('max_videos', 0) or 'no'} reference videos, not {len(videos)}.")
    if len(audios) > takes.get("max_audios", 0):
        raise RuntimeError(f"This workflow takes {takes.get('max_audios', 0) or 'no'} reference audio clips, not {len(audios)}.")
    most = takes.get("max_references", takes["max_images"])
    if images + len(videos) + len(audios) > most:
        raise RuntimeError(f"This workflow takes {most} references at most, of every kind together, not {images + len(videos) + len(audios)}.")


def keep_references(state, videos, audios):
    """Copies the reference videos and audio clips into the run's folder, in the shape the clips take them."""
    for number, (path, role) in enumerate(videos, 1):
        name = f"ref_video_{number}.mp4"
        sound = media.reference_video(path, os.path.join(state.folder, name))
        state.ref_videos.append({"file": name, "role": role, "sound": sound})
    for number, (path, role) in enumerate(audios, 1):
        name = f"ref_audio_{number}.wav"
        media.reference_audio(path, os.path.join(state.folder, name))
        state.ref_audios.append({"file": name, "role": role})


def plan_run(args, stages):
    images = [media.load_image(path) for path in ([args.image] if args.image else []) + args.reference]
    videos, audios = reference_roles(args)
    inputs = media.inputs_fingerprint(images, args.image_roles,
                                      media.audio_fingerprint(path=args.audio, text=args.narration, voice=args.voice),
                                      videos + audios)
    comfy = ComfyClient()
    takes = comfy.takes()
    # Clips with room for one image can't show the video model a second, so the LLM puts it into words.
    image_2_in_words = images.pop(1) if len(images) == 2 and takes["max_images"] == 1 else None
    if not takes["min_images"] <= len(images) <= takes["max_images"]:
        raise RuntimeError(f"This workflow takes {takes['min_images']}-{takes['max_images']} images, not {len(images)}.")
    check_references(takes, len(images), videos, audios)
    video_model = comfy.video_model()
    if args.h3_enhance != "off" and not h3_refiner.supports(comfy.workflow):
        raise RuntimeError("OpenH3-IR requires a supported MiniMax H3 workflow.")
    # A clip can only start on the last one's final frame if an image is all it takes.
    start = args.start if takes["max_images"] == 1 else "image"
    if args.size and media.size_problem(video_model, *args.size):
        raise RuntimeError(media.size_problem(video_model, *args.size))
    width, height = args.size or media.render_size(*(images[0].size if images else media.DEFAULT_SHAPE),
                                                   media.most_pixels(video_model) or media.RENDER_AREA)
    llm = args.llm or planner.default_llm()
    if not args.direct and not planner.llm_configured(llm):
        raise RuntimeError(f"{llm} has no credentials: see .env.example.")
    state = RunState.new(args.prompt, args.duration, start, width, height, llm,
                         os.getenv("COMFYUI_WORKFLOW") or None, len(images))
    state.inputs = inputs
    state.options = workflow_options.chosen_in_env() or None
    state.h3_mode, state.h3_creativity, state.image_roles = args.h3_enhance, args.h3_creativity, args.image_roles
    print_stage("Plan", stages)
    keep_references(state, videos, audios)
    audio = args.audio or (voiced_narration(args) if args.narration else None)
    if audio:
        media.to_wav(audio, state.audio_path)
        state.has_audio = True
        # The audio decides how long the film is.
        state.seconds = math.ceil(media.duration(state.audio_path))
        if not planner.MIN_FILM_SECONDS <= state.seconds <= planner.MAX_FILM_SECONDS:
            raise RuntimeError(f"The audio is {state.seconds}s, and a film has to be {planner.MIN_FILM_SECONDS}-{planner.MAX_FILM_SECONDS} seconds long.")
    brief = args.prompt + (f"\n\nThe film is narrated, and the shots follow this narration in order: {args.narration}" if args.narration else "")
    print(f"🗂️ Run folder: {state.id}")

    shortest, longest, usual = planner.clip_seconds(video_model, args.clip_seconds)
    if args.direct:
        # Your prompt as it is, for every clip, at lengths that add up to the film.
        count = planner.shot_count(state.seconds, video_model, args.clip_seconds)
        lengths = planner.fit_durations([usual] * count, state.seconds, video_model, args.clip_seconds)
        title = " ".join(args.prompt.split()[:6]) or "Untitled"
        shots = [{"title": f"Clip {n}", "seconds": length, "prompt": args.prompt} for n, length in enumerate(lengths, 1)]
        print(f"✍️ Your prompt for each of {count} clips, {shortest}-{longest}s a clip, at {width}x{height}.")
        for image, path in zip(images, state.image_paths):
            media.save_input(image, path)
        for index, shot in enumerate(shots):
            shot["seed"] = seeds.nth(args.seed, index)
        state.update(title=title, shots=shots)
        return state
    print(f"🧠 {llm} is writing {planner.shot_count(state.seconds, video_model, args.clip_seconds)} prompts for a {state.seconds}s film at "
          f"{width}x{height}, {shortest}-{longest}s a clip...")
    if image_2_in_words:
        print("🖼️ This workflow's clips take one image, so image 2 only reaches them through what the prompts say.")
    title, shots = planner.plan_film([media.preview_jpeg(image) for image in images], brief, state.seconds, state.start, llm,
                                     video_model, args.image_roles, media.preview_jpeg(image_2_in_words) if image_2_in_words else None,
                                     state.references(), chosen=args.clip_seconds)
    for image, path in zip(images, state.image_paths):
        media.save_input(image, path)
    for index, shot in enumerate(shots):
        shot["seed"] = seeds.nth(args.seed, index)
    state.update(title=title, shots=shots)

    print(f"🎞️ {title}")
    for number, shot in enumerate(shots, 1):
        print(f"\n{number}. {shot['title']} ({shot['seconds']}s)\n   {shot['prompt']}")
    print(f"\n⏳ Rendering should take about {fmt_time(state.seconds * RENDER_SECONDS_PER_SECOND * width * height / media.RENDER_AREA)}.")
    return state


def clip_images(state, index):
    if state.start == "chain" and index > 0:
        return [media.last_frame(state.clip_path(index - 1), state.frame_path)]
    return state.image_paths


def refine_shot(refiner, state, shot, images, workflow):
    videos = [{"path": state.reference_path(v), "role": v["role"], "sound": v.get("sound"),
               "seconds": media.duration(state.reference_path(v))} for v in state.ref_videos]
    audios = [{"path": state.reference_path(a), "role": a["role"],
               "seconds": media.duration(state.reference_path(a))} for a in state.ref_audios]
    return refiner.compile(shot["prompt"], shot["seconds"], state.width, state.height, state.h3_creativity,
                           images, state.image_roles, videos, audios, shot.get("seed"),
                           first_frame=h3_refiner.first_frame(workflow, images))


def refinement_key(state, shot, images):
    refs = [state.reference_path(ref) for ref in state.ref_videos + state.ref_audios]
    data = {"prompt": shot["prompt"], "seconds": shot["seconds"], "seed": shot.get("seed"),
            "size": [state.width, state.height], "creativity": state.h3_creativity,
            "image_roles": state.image_roles, "videos": state.ref_videos, "audios": state.ref_audios,
            "files": [h3_refiner.digest(path) for path in [*images, *refs]]}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def refine_review(state):
    pending = [shot for shot in state.shots if not shot.get("h3_reviewed")]
    if not pending:
        return
    url = (os.getenv("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
    try:
        online = requests.get(f"{url}/system_stats", timeout=5).ok
    except requests.RequestException:
        online = False
    with h3_refiner.Refiner(url if online else None) as refiner:
        workflow = ComfyClient().workflow
        for shot in pending:
            shot["prompt"] = refine_shot(refiner, state, shot, state.image_paths, workflow)
            shot["h3_reviewed"] = True
            state.save()


def clip_audio(state, comfy, index):
    """This shot's stretch of the film's audio, uploaded, if the workflow can use it."""
    if not state.has_audio or not comfy.takes_audio():
        return None
    start = sum(shot["seconds"] for shot in state.shots[:index])
    part = media.audio_slice(state.audio_path, start, state.shots[index]["seconds"], os.path.join(state.parts_dir, "shot_audio.wav"))
    return comfy.upload(part, f"autogen_{state.id}_audio.wav")


def upload_references(state, comfy):
    """The run's reference videos and audio clips, uploaded: (videos, audio clips)."""
    return ([comfy.upload(state.reference_path(video), f"autogen_{state.id}_{video['file']}") for video in state.ref_videos],
            [comfy.upload(state.reference_path(audio), f"autogen_{state.id}_{audio['file']}") for audio in state.ref_audios])


def render_shot(state, comfy, index, images, audio=None, references=((), ())):
    """Renders until the file holds up, on a new seed each time. Returns its path, or None."""
    shot = state.shots[index]
    save_as = os.path.join(state.clips_dir, f"shot_{index + 1:03d}")
    for attempt in range(1, CLIP_ATTEMPTS + 1):
        seed = seeds.pick(shot.get("seed"), attempt)
        print(f"      🌱 Seed {seed}")
        try:
            kwargs = {"h3_refined": True} if state.h3_mode != "off" else {}
            clip = comfy.render(shot["prompt"], images, shot["seconds"], state.width, state.height, save_as, audio, *references, seed=seed, **kwargs)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"      ❌ {e} (attempt {attempt}/{CLIP_ATTEMPTS}).")
            continue
        problem = media.clip_problem(clip, shot["seconds"])
        if not problem:
            shot["used_seed"] = seed
            return clip
        print(f"      ⚠️ Rejected: {problem} (attempt {attempt}/{CLIP_ATTEMPTS}).")
    return None


def render_clips(state, comfy, most=None):
    """Returns True once every shot has an accepted clip, or the most asked for are made."""
    total = len(state.shots)
    time_spent = film_rendered = 0
    references = None
    made = 0
    for index, shot in enumerate(state.shots):
        if index in state.clips:
            print(f"   ♻️ Clip {index + 1}/{total} kept from earlier.")
            continue
        if most is not None and made >= most:
            break
        made += 1

        started = time.time()
        print(f"   🎬 Clip {index + 1}/{total} · {shot['title']} · {shot['seconds']}s")
        image_paths = clip_images(state, index)
        if state.h3_mode == "render":
            key = refinement_key(state, shot, image_paths)
            if shot.get("h3_key") == key and shot.get("enhanced_prompt"):
                enhanced = shot["enhanced_prompt"]
                print("   ♻️ Keeping the OpenH3-IR prompt already written.")
            else:
                with h3_refiner.Refiner(comfy.url) as refiner:
                    enhanced = refine_shot(refiner, state, shot, image_paths, comfy.workflow)
                shot["enhanced_prompt"], shot["h3_key"] = enhanced, key
                state.save()
            print(f"   ✍️ OpenH3-IR prompt:\n{enhanced}")
        else:
            enhanced = None
        try:
            images = [comfy.upload(path, f"autocomfyui_{state.id}{'' if number == 1 else f'_{number}'}.png")
                      for number, path in enumerate(image_paths, 1)]
            # Once a run: they're the same for every clip.
            references = references or upload_references(state, comfy)
        except (OSError, ffmpeg.Error, requests.RequestException) as e:
            print(f"   ❌ Could not give ComfyUI the start frame or references: {e}")
            return False
        original = shot["prompt"]
        if enhanced is not None:
            shot["prompt"] = enhanced
        try:
            clip = render_shot(state, comfy, index, images, clip_audio(state, comfy, index), references)
        finally:
            shot["prompt"] = original
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
    # Made again (a shot redone), it keeps its name; a new film never takes another's.
    if not state.film_name:
        state.update(film_name=unique_name(paths.OUTPUT_DIR, f"{title_words(state.title, 'film')}_{shorthand(state.workflow, BUILT_IN)}", ".mp4"))
    output = os.path.join(paths.OUTPUT_DIR, f"{state.film_name}.mp4")
    media.stitch(clips, state.parts_dir, output, state.width, state.height, state.start == "chain")
    if state.has_audio:
        media.with_audio(output, state.audio_path)
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
        # The rest of the film is made with the options the first clips were.
        os.environ["COMFYUI_OPTIONS"] = json.dumps(state.options or {})
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
        stages = STAGES
    else:
        preflight(needs_comfy=not args.plan_only)
        stages = STAGES[:1] if args.plan_only else STAGES
        state = plan_run(args, stages)
        if state.h3_mode == "review":
            refine_review(state)
        if args.plan_only:
            print(f"📝 Change any prompt in the panel or {state.path}, then: python -m pipelines.video.main --resume {state.id}")
            return 0

    if args.resume and state.h3_mode == "review":
        refine_review(state)
    print_stage("Clips", stages)
    if state.next_stage() == "Clips":
        comfy = ComfyClient()
        if args.resume:
            # A run that was killed outright leaves its last clip rendering.
            comfy.cancel()
        if not render_clips(state, comfy, most=1 if args.first_only else None):
            print(f"💾 Kept what's done. Continue with: python -m pipelines.video.main --resume {state.id}")
            return 1
        if args.first_only and state.next_stage() == "Clips":
            print(f"👀 The first clip is made, to look at: continue for the rest. python -m pipelines.video.main --resume {state.id}")
            return 0

    print_stage("Stitching", stages)
    output = stitch_film(state)
    print(f"✅ {media.duration(output):.0f}s film saved: {output}")

    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(paths.PROJECT_ROOT, ".env"))
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. A saved run continues with --resume (python -m pipelines.video.main --list shows them).")
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

"""
Makes a set of images with ComfyUI: from text alone, or by editing the images you give it.
An LLM writes one prompt per image from a short brief, or your own prompt goes in word for word.
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

from pipelines.image import media, paths
from pipelines.image import planner as image_planner
from pipelines.image.comfy import ImageClient
from pipelines.image.run_state import STAGES, RunState, list_runs
from pipelines.video import media as video_media
from pipelines.video import planner

IMAGE_ATTEMPTS = 2


def parse_count(text):
    try:
        count = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number of images.")
    if not 1 <= count <= image_planner.MAX_IMAGES:
        raise argparse.ArgumentTypeError(f"A set has 1-{image_planner.MAX_IMAGES} images, not {count}.")
    return count


def parse_size(text):
    if not media.parse_size(text):
        raise argparse.ArgumentTypeError(f"'{text}' is not a size. Use 1920x1080, with each side {media.MIN_EDGE}-{media.MAX_EDGE}.")
    return text


def parse_aspect(text):
    if text != media.FROM_IMAGE and not media.parse_aspect(text):
        raise argparse.ArgumentTypeError(f"'{text}' is not an aspect ratio. Use 16:9, or '{media.FROM_IMAGE}'.")
    return text


def parse_megapixels(text):
    if text != media.FROM_IMAGE and not media.parse_megapixels(text):
        raise argparse.ArgumentTypeError(f"'{text}' is not a pixel count. Use up to {media.MAX_MEGAPIXELS:g}, or '{media.FROM_IMAGE}'.")
    return text


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Makes a set of images with ComfyUI, from text or by editing an image.")
    parser.add_argument("prompt", nargs="?", help="What you want, in a sentence or two. With --direct, the exact prompt.")
    parser.add_argument("--image", action="append", default=[], metavar="FILE", help="The image to edit, repeatable if the workflow takes several. Without one, it makes them from text.")
    parser.add_argument("--count", type=parse_count, default=1, help=f"How many images, 1-{image_planner.MAX_IMAGES} (default 1).")
    parser.add_argument("--direct", action="store_true", help="Use the prompt word for word. No LLM: each image only gets a new seed.")
    parser.add_argument("--llm", choices=[name for name, _ in planner.LLMS], default=None, help="Who writes the prompts (default: LLM_PROVIDER in .env, else gemini).")
    parser.add_argument("--aspect", type=parse_aspect, default=media.FROM_IMAGE, help=f"Aspect ratio such as 16:9, or '{media.FROM_IMAGE}' for the source image's (the default, square without one).")
    parser.add_argument("--megapixels", type=parse_megapixels, default=str(media.DEFAULT_MEGAPIXELS), help=f"Pixel count, such as 1.5 (the default is {media.DEFAULT_MEGAPIXELS:g}), or '{media.FROM_IMAGE}' for the source image's.")
    parser.add_argument("--size", type=parse_size, help="An exact size such as 1920x1080, in place of --aspect and --megapixels.")
    parser.add_argument("--plan-only", action="store_true", help="Write the prompts and stop before rendering.")
    parser.add_argument("--resume", metavar="RUN", help="Continue a saved run, with its original options.")
    parser.add_argument("--redo", type=int, metavar="IMAGE", help="With --resume: make this image again, by its place in the set.")
    parser.add_argument("--list", action="store_true", help="List saved runs.")
    parser.add_argument("--check", action="store_true", help="Check the LLM key and ComfyUI, and make nothing.")
    args = parser.parse_args(argv)

    if args.redo is not None and not args.resume:
        parser.error("--redo needs --resume.")
    if args.plan_only and (args.resume or args.direct):
        parser.error("--plan-only stops after writing prompts: with --resume or --direct none get written.")
    if not (args.resume or args.list or args.check or args.prompt):
        parser.error("give a prompt, or --resume RUN, --list or --check.")
    return args


def print_stage(name, stages):
    print(f"▶ Stage {stages.index(name) + 1}/{len(stages)}: {name}")


def print_runs():
    runs = list_runs()
    if not runs:
        print("No saved runs.")
    for run in runs:
        print(f"{run.id}  {len(run.done)}/{len(run.shots or [])} images  {run.title or run.brief}")


def check_setup(args):
    """Each thing a run needs, without making anything."""
    llm = args.llm or planner.default_llm()
    problems = []
    if planner.llm_configured(llm):
        print(f"✅ {llm} has credentials.")
    else:
        problems.append(f"{llm} has no credentials, so only your own prompts can run: see .env.example.")
    try:
        comfy = ImageClient()
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


def plan_run(args, stages):
    images = [video_media.load_image(path) for path in args.image]
    inputs = video_media.inputs_fingerprint(images, [])
    comfy = ImageClient()
    takes = comfy.takes()
    if not takes["min_images"] <= len(images) <= takes["max_images"]:
        raise RuntimeError(f"This workflow takes {takes['min_images']}-{takes['max_images']} images, not {len(images)}.")
    llm = None if args.direct else args.llm or planner.default_llm()
    if llm and not planner.llm_configured(llm):
        raise RuntimeError(f"{llm} has no credentials: see .env.example.")
    width, height = media.pick_size(args.size, args.aspect, args.megapixels, images[0].size if images else None)

    state = RunState.new(args.prompt, args.count, args.direct, width, height, llm, os.getenv("COMFYUI_WORKFLOW") or None, len(images))
    state.inputs = inputs
    for image, path in zip(images, state.input_paths):
        media.save_source(image, path)
    print_stage("Prompts", stages)
    print(f"🗂️ Run folder: {state.id}")
    if args.direct:
        title = " ".join(args.prompt.split()[:6])
        versions = [{"title": f"Version {number}", "prompt": args.prompt} for number in range(1, args.count + 1)]
    else:
        print(f"🧠 {llm} is writing {args.count} {'prompt' if args.count == 1 else 'prompts'}...")
        title, versions = image_planner.write_prompts([video_media.preview_jpeg(image) for image in images], args.prompt, args.count, llm)
    shots = [{"number": number, "file": None, **version} for number, version in enumerate(versions, 1)]
    # At once: the prompts are what was paid for.
    state.update(title=title, shots=shots)

    print(f"🖼️ {title}, {width}x{height}")
    for shot in shots:
        print(f"\n{shot['number']}. {shot['title']}\n   {shot['prompt']}")
    return state


def upload_images(state, comfy):
    """The source images in ComfyUI, the first at the size asked for, as a workflow that follows it comes out that size."""
    if not state.input_paths:
        return []
    first = media.fit(state.input_paths[0], *media.render_size(state.width, state.height), state.fitted_path)
    return [comfy.upload(path, f"autocomfyui_{state.id}{'' if number == 1 else f'_{number}'}.png")
            for number, path in enumerate([first, *state.input_paths[1:]], 1)]


def render_image(state, comfy, shot, images):
    """Renders until the file holds up, on a new seed each time. Returns its path, or None."""
    save_as = os.path.join(state.output_dir, f"image_{shot['number']:03d}")
    rendered = media.render_size(state.width, state.height)
    for attempt in range(1, IMAGE_ATTEMPTS + 1):
        try:
            path = comfy.render_image(shot["prompt"], images, *rendered, save_as)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"      ❌ {e} (attempt {attempt}/{IMAGE_ATTEMPTS}).")
            continue
        problem = media.image_problem(path)
        if not problem:
            media.trim(path, rendered, (state.width, state.height))
            return path
        print(f"      ⚠️ Rejected: {problem} (attempt {attempt}/{IMAGE_ATTEMPTS}).")
    return None


def render_images(state, comfy):
    """Returns how many it could not make. One that won't come out doesn't cost the rest of the set."""
    images = upload_images(state, comfy)
    failed = 0
    time_spent = made = 0
    total = len(state.shots)
    for position, shot in enumerate(state.shots, 1):
        if shot.get("file"):
            print(f"   ♻️ Image {position}/{total} kept from earlier.")
            continue
        started = time.time()
        print(f"   🖼️ Image {position}/{total} · {shot['title']}")
        path = render_image(state, comfy, shot, images)
        if not path:
            print(f"   ❌ Image {position} failed.")
            failed += 1
            continue
        shot["file"] = os.path.basename(path)
        state.save()
        time_spent += time.time() - started
        made += 1
        left = sum(1 for other in state.shots if not other.get("file")) - failed
        print(f"      ✅ Done{f', about {max(1, round(left * time_spent / made / 60))} min left' if left else ''}.")
    return failed


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
        # The rest of the set has to match what's been made.
        if state.workflow:
            os.environ["COMFYUI_WORKFLOW"] = state.workflow
        if args.redo is not None:
            if not 1 <= args.redo <= len(state.shots):
                raise RuntimeError(f"This set has images 1-{len(state.shots)}, not {args.redo}.")
            state.shots[args.redo - 1]["file"] = None
        print(f"🗂️ Run folder: {state.id}")
        print(f"🖼️ Continuing {state.title}")
        stages = STAGES
    else:
        stages = STAGES[:1] if args.plan_only else STAGES
        state = plan_run(args, stages)
        if args.plan_only:
            print(f"📝 Change any prompt in the panel or {state.path}, then: python -m pipelines.image.main --resume {state.id}")
            return 0

    print_stage("Images", stages)
    comfy = ImageClient()
    if not comfy.wait_until_reachable():
        raise RuntimeError(f"ComfyUI isn't running at {comfy.url}. Start it first: the prompts are saved.")
    if args.resume:
        # A run that was killed outright leaves its last image rendering.
        comfy.cancel()
    # Saved after this, or a ComfyUI that's down would cost a redo's image without rendering anything.
    state.save()
    failed = render_images(state, comfy)
    if failed:
        print(f"💾 {failed} of {len(state.shots)} failed, and the rest are kept. Continue with: python -m pipelines.image.main --resume {state.id}")
        return 1
    print(f"✅ {len(state.shots)} {'image' if len(state.shots) == 1 else 'images'} saved: {state.output_dir}")
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(paths.PROJECT_ROOT, ".env"))
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. A saved run continues with --resume (python -m pipelines.image.main --list shows them).")
        return 130
    except (RuntimeError, OSError) as e:
        print(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

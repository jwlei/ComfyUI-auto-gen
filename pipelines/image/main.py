"""
Makes a set of images with ComfyUI: from text alone, or by editing the images you give it.
An LLM writes one prompt per image from a short brief, or your own prompt goes in word for word.
"""
import argparse
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv

from core import batch_stop, h3_refiner, seeds
from core.saved_workflows import shorthand, title_words, unique_name
from pipelines.image import media, models, paths, regions
from pipelines.image import planner as image_planner
from pipelines.image.comfy import ImageClient
from pipelines.image.run_state import STAGES, RunState, list_runs
from pipelines.video import media as video_media
from pipelines.video import planner
from pipelines.video.main import parse_seed

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


def parse_option(text):
    name, _, value = text.partition("=")
    if not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError(f"'{text}' is not an option. Use NAME=VALUE, such as group_2nd_pass=0.")
    return name.strip(), value.strip()


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
    parser.add_argument("--seed", type=parse_seed, help="The first image's seed. Each image after it gets the next one, so the set "
                                                       "can be made again. Default: a random seed for every image.")
    parser.add_argument("--option", type=parse_option, action="append", default=[], metavar="NAME=VALUE",
                        help="One of the workflow's own options, repeatable, such as group_2nd_pass=0 or lora_knpv3_1=0.8. "
                             "Its map in logs/workflows lists them under _options.")
    parser.add_argument("--compose", action="store_true", help="The images are references for a new picture, made at the size asked for, "
                                                                "rather than image 1 edited. On a model that can (models.COMPOSE_SWITCH).")
    parser.add_argument("--transparent", action="store_true", help="With a transparent background, on a model that makes one.")
    parser.add_argument("--reference-detail", choices=("standard", "full"), default="standard",
                        help="How big the images besides the one edited are shown: standard, about a megapixel each (the default), or full.")
    parser.add_argument("--source-seed", type=parse_seed, action="append", default=[], metavar="SEED",
                        help="A seed that made one of the images, repeatable. An edit never uses it again: it would come out over-sharpened.")
    parser.add_argument("--areas", metavar="FILE", help='Change only painted parts of image 1, as JSON: {"mask": PNG painted in a colour per area, '
                                                       '"areas": [{"colour": "#e62828", "action": "change|remove|replace|add", "text": ...}]}.')
    parser.add_argument("--expand", metavar="L,T,R,B", help="Expand image 1 by these many pixels left, top, right and bottom. The prompt, "
                                                            "if any, says what the new parts show.")
    parser.add_argument("--plan-only", action="store_true", help="Write the prompts and stop before rendering.")
    parser.add_argument("--enhance-only", action="store_true", help="With the workflow's own prompt enhancer on: have it write each "
                                                                    "image's prompt, and stop there to read them. They're then made as written.")
    parser.add_argument("--first-only", action="store_true", help="Make the first image and stop, to see it before making the rest.")
    parser.add_argument("--resume", metavar="RUN", help="Continue a saved run, with its original options.")
    parser.add_argument("--redo", type=int, metavar="IMAGE", help="With --resume: make this image again, by its place in the set.")
    parser.add_argument("--list", action="store_true", help="List saved runs.")
    parser.add_argument("--check", action="store_true", help="Check the LLM key and ComfyUI, and make nothing.")
    args = parser.parse_args(argv)

    if args.redo is not None and not args.resume:
        parser.error("--redo needs --resume.")
    if args.plan_only and (args.resume or args.direct):
        parser.error("--plan-only stops after writing prompts: with --resume or --direct none get written.")
    if (args.areas or args.expand) and not args.image:
        parser.error("--areas and --expand work on an image: give it with --image.")
    if args.areas and args.expand:
        parser.error("Change painted areas or expand it, one at a time.")
    if args.expand:
        try:
            args.expand = [int(value) for value in args.expand.split(",")]
        except ValueError:
            args.expand = None
        if not args.expand or len(args.expand) != 4 or min(args.expand) < 0:
            parser.error("--expand takes four whole numbers of pixels: left, top, right and bottom, such as 0,0,400,0.")
    if (args.areas or args.expand) and (args.compose or args.plan_only):
        parser.error("--areas and --expand write their own prompt: leave out --compose and --plan-only.")
    if not (args.resume or args.list or args.check or args.prompt or args.areas or args.expand):
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
    if llm == "local":
        try:
            h3_refiner._paths(h3_refiner.selected_model(), require_compiler=False)
            print("✅ Local model and llama.cpp found.")
        except (RuntimeError, ValueError) as e:
            problems.append(str(e))
    elif planner.llm_configured(llm):
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
    region_edit = bool(args.areas or args.expand)
    inputs = video_media.inputs_fingerprint(images, [])
    if region_edit:
        inputs = f"{inputs}:{_region_fingerprint(args)}"
    options = dict(args.option)
    model = ImageClient(options).model()
    if args.compose:
        if model not in models.COMPOSE_SWITCH:
            raise RuntimeError(f"{models.label(model)} can't make a new picture from references: choose Qwen-Image 2.1's edit workflow.")
        options[models.COMPOSE_SWITCH[model]] = "1"
    if region_edit:
        if model not in models.REGION_EDITS:
            raise RuntimeError(f"{models.label(model)} can't change part of an image or expand it: choose Qwen-Image 2.1's edit workflow.")
        # The instruction is ours, and must reach the model as it is.
        if "prompt_enhancer" in options:
            options["prompt_enhancer"] = "0"
        images = images[:1]
    if args.transparent and model not in models.TRANSPARENT:
        raise RuntimeError(f"{models.label(model)} doesn't make transparent images: choose a Qwen-Image 2.1 workflow.")
    args.option = list(options.items())
    comfy = ImageClient(options)
    takes = comfy.takes()
    if not takes["min_images"] <= len(images) <= takes["max_images"]:
        raise RuntimeError(f"This workflow takes {takes['min_images']}-{takes['max_images']} images, not {len(images)}.")
    direct = args.direct or region_edit
    llm = None if direct else args.llm or planner.default_llm()
    if llm == "local" and images and h3_refiner.selected_model() == "qwen3-14b":
        raise RuntimeError("Qwen3 14B is text-only and cannot analyze attached images. Choose Gemma 4 in Settings.")
    if llm and not planner.llm_configured(llm):
        raise RuntimeError(f"{llm} has no credentials: see .env.example.")
    width, height = media.pick_size(args.size, args.aspect, args.megapixels, images[0].size if images else None,
                                    models.most_megapixels(model, args.compose))
    # Before the LLM is paid for anything.
    problem = (None if region_edit else models.size_problem_for(model, width, height, args.compose)) \
        or models.count_problem(model, args.count, direct)
    if problem:
        raise RuntimeError(problem)

    brief = args.prompt or ""
    state = RunState.new(brief, args.count, direct, width, height, llm, os.getenv("COMFYUI_WORKFLOW") or None, len(images))
    state.inputs = inputs
    state.options = options or None
    state.compose, state.transparent, state.detail = args.compose, args.transparent, args.reference_detail
    for image, path in zip(images, state.input_paths):
        media.save_source(image, path)
    taken = set(args.source_seed)
    for path in args.image:
        taken |= media.seeds_in(path)
    state.source_seeds = sorted(taken) or None
    print_stage("Prompts", stages)
    print(f"🗂️ Run folder: {state.id}")
    if region_edit:
        title, versions = plan_region(state, args, model)
    elif args.direct:
        title = " ".join(args.prompt.split()[:6])
        versions = [{"title": f"Version {number}", "prompt": args.prompt} for number in range(1, args.count + 1)]
    else:
        print(f"🧠 {llm} is writing {args.count} {'prompt' if args.count == 1 else 'prompts'}...")
        plan_args = ([video_media.preview_jpeg(image) for image in images], args.prompt, args.count, llm, model)
        if llm == "local":
            with h3_refiner.Refiner(comfy.url, require_compiler=False) as refiner:
                title, versions = image_planner.write_prompts(*plan_args, compose=args.compose,
                                                              transparent=args.transparent, refiner=refiner)
        else:
            title, versions = image_planner.write_prompts(*plan_args, compose=args.compose, transparent=args.transparent)
    shots = [{"number": number, "file": None, "seed": seeds.nth(args.seed, number - 1), **version}
             for number, version in enumerate(versions, 1)]
    # At once: the prompts are what was paid for.
    output_name = unique_name(paths.OUTPUT_DIR, f"{title_words(title, 'images')}_{shorthand(state.workflow, 'image')}")
    state.update(title=title, shots=shots, output_name=output_name)

    print(f"🖼️ {title}, {state.width}x{state.height}")
    for shot in shots:
        print(f"\n{shot['number']}. {shot['title']}\n   {shot['prompt']}")
    return state


def _region_fingerprint(args):
    """What a region edit changes, so a stopped one is only continued by the same edit."""
    if args.expand:
        return f"expand:{args.expand}:{args.prompt or ''}"
    try:
        with open(args.areas, "r", encoding="utf-8") as f:
            spec = json.load(f)
        with open(spec["mask"], "rb") as f:
            painted = hashlib.sha256(f.read()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Can't read the painted areas in {args.areas}: {e}")
    return f"areas:{json.dumps(spec.get('areas'), sort_keys=True)}:{painted}"


def plan_region(state, args, model):
    """Prepares a region edit (pipelines/image/regions.py) and its prompt, the same for every version: they differ by seed."""
    budget = models.TOKEN_BUDGET.get(model)
    try:
        if args.expand:
            region = regions.prepare_expand(state.input_paths[0], *args.expand, state.folder, args.prompt or "", budget)
            title = "Expanded" + (f" {' '.join(args.prompt.split()[:5])}" if args.prompt else "")
        else:
            with open(args.areas, "r", encoding="utf-8") as f:
                spec = json.load(f)
            region = regions.prepare_areas(state.input_paths[0], spec["mask"], spec.get("areas") or [], state.folder, budget)
            first = region["areas"][0]
            title = (f"{first.get('action', 'change').capitalize()} {' '.join((first.get('text') or '').split()[:5])}".strip()
                     if len(region["areas"]) == 1 else f"{len(region['areas'])} changes")
    except (OSError, KeyError, TypeError) as e:
        raise RuntimeError(f"Can't read the painted areas: {e}")
    except ValueError as e:
        raise RuntimeError(str(e))
    state.region = region
    state.width, state.height = region["size"]
    print(f"✍️ Working on {region['work'][0]}x{region['work'][1]}, for a {state.width}x{state.height} image.")
    versions = [{"title": f"Version {number}", "prompt": region["prompt"]} for number in range(1, args.count + 1)]
    return title, versions


def enhance_only(state, args):
    """
    Has the workflow's prompt enhancer write every image's prompt, each on its own seed, and stops so they can be read
    and changed. What it wrote then goes to the model as it is: the enhancer is switched off for the rest of the run.
    """
    options = dict(args.option)
    comfy = ImageClient(options)
    if not comfy.wait_until_reachable():
        raise RuntimeError(f"ComfyUI isn't running at {comfy.url}. Start it first: the prompts are saved.")
    images = upload_images(state, comfy)
    print(f"✍️ The workflow's prompt enhancer is writing {len(state.shots)} {'prompt' if len(state.shots) == 1 else 'prompts'}...")
    for position, shot in enumerate(state.shots, 1):
        seed = seeds.pick(shot.get("seed"))
        written = comfy.enhance(shot["prompt"], images, seed)
        if not written:
            raise RuntimeError("This workflow has no prompt enhancer switched on: choose it under What writes the prompts.")
        shot["enhanced_from"], shot["prompt"] = shot["prompt"], written
        print(f"\n{position}. {written}")
        state.save()
    if "prompt_enhancer" in options:
        options["prompt_enhancer"] = "0"
    state.options = options
    state.save()
    print(f"\n📝 Change any prompt in the panel or {state.path}, then: python -m pipelines.image.main --resume {state.id}")
    return 0


def upload_images(state, comfy):
    """
    The images in ComfyUI. Edited, the first is fitted to the size asked for, as a workflow that follows it comes out
    that size, and the rest are shown smaller (reference detail). As references for a new picture, all are. A region
    edit uploads what regions.py prepared instead.
    """
    name = lambda number: f"autocomfyui_{state.id}{'' if number == 1 else f'_{number}'}.png"
    if state.region:
        return [comfy.upload(os.path.join(state.folder, shown), name(number)) for number, shown in enumerate(state.region["shown"], 1)]
    if not state.input_paths:
        return []
    made = comfy.render_size(state.width, state.height)
    sources = list(state.input_paths)
    if not state.compose:
        sources[0] = media.fit(sources[0], *made, state.fitted_path)
    sizes = models.reference_sizes(comfy.model(), made, [media.size_of(path) for path in sources], state.detail, editing=not state.compose)
    shown = []
    for number, (path, size) in enumerate(zip(sources, sizes), 1):
        if tuple(size) != tuple(media.size_of(path)):
            path = media.resized(path, size, os.path.join(state.folder, f"shown_{number}.png"))
        shown.append(path)
    return [comfy.upload(path, name(number)) for number, path in enumerate(shown, 1)]


def render_image(state, comfy, shot, images):
    """Renders until the file holds up, on a new seed each time. Returns its path, or None."""
    save_as = os.path.join(state.output_dir, state.file_stem(shot))
    rendered = tuple(state.region["work"]) if state.region else comfy.render_size(state.width, state.height)
    taken = set(state.source_seeds or ())
    for attempt in range(1, IMAGE_ATTEMPTS + 1):
        seed = seeds.pick(shot.get("seed"), attempt)
        if seed in taken:
            avoided = seeds.avoid(seed, taken)
            print(f"      🌱 Seed {seed} made the image this starts from, and would come out over-sharpened: using {avoided} instead.")
            seed = avoided
        print(f"      🌱 Seed {seed}")
        try:
            path = comfy.render_image(shot["prompt"], images, *rendered, save_as, seed, transparent=state.transparent)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"      ❌ {e} (attempt {attempt}/{IMAGE_ATTEMPTS}).")
            continue
        problem = media.image_problem(path)
        if not problem:
            if state.region:
                path = regions.composite(path, state.region, state.input_paths[0], state.folder, path)
            else:
                media.trim(path, rendered, (state.width, state.height))
            shot["used_seed"] = seed
            # What the workflow's prompt enhancer made of the prompt, for the prompts view to show.
            if getattr(comfy, "enhanced", None):
                shot["enhanced_prompt"] = comfy.enhanced
            else:
                shot.pop("enhanced_prompt", None)
            return path
        print(f"      ⚠️ Rejected: {problem} (attempt {attempt}/{IMAGE_ATTEMPTS}).")
    return None


def render_images(state, comfy, most=None):
    """Returns how many it could not make. One that won't come out doesn't cost the rest of the set. most: stop after that many."""
    images = upload_images(state, comfy)
    failed = 0
    time_spent = made = tried = 0
    total = len(state.shots)
    for position, shot in enumerate(state.shots, 1):
        if most is not None and made >= most:
            break
        if shot.get("file") and not shot.get("redo"):
            print(f"   ♻️ Image {position}/{total} kept from earlier.")
            continue
        # The panel's Stop after this one: the image before is done, so the rest wait to be continued.
        if tried and batch_stop.requested():
            print("   🛑 Stopping after this image, as asked.")
            break
        tried += 1
        started = time.time()
        print(f"   🖼️ Image {position}/{total} · {shot['title']}")
        path = render_image(state, comfy, shot, images)
        redone = shot.pop("redo", None)
        if not path:
            print(f"   ❌ Image {position} failed{', so the one before stays' if redone and shot.get('file') else ''}.")
            state.save()
            failed += 1
            continue
        shot["file"] = os.path.basename(path)
        state.save()
        time_spent += time.time() - started
        made += 1
        left = sum(1 for other in state.shots if not other.get("file") or other.get("redo")) - failed
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
        # What's still to make is made as the rest were.
        args.option = list((state.options or {}).items())
        if args.redo is not None:
            if not 1 <= args.redo <= len(state.shots):
                raise RuntimeError(f"This set has images 1-{len(state.shots)}, not {args.redo}.")
            # The one there stays, in the set and on disk, until the new one takes its place: a redo that fails keeps it.
            state.shots[args.redo - 1]["redo"] = True
        print(f"🗂️ Run folder: {state.id}")
        print(f"🖼️ Continuing {state.title}")
        stages = STAGES
    else:
        stages = STAGES[:1] if args.plan_only else STAGES
        state = plan_run(args, stages)
        if args.plan_only:
            print(f"📝 Change any prompt in the panel or {state.path}, then: python -m pipelines.image.main --resume {state.id}")
            return 0

    if not args.resume and args.enhance_only:
        return enhance_only(state, args)
    print_stage("Images", stages)
    comfy = ImageClient(dict(args.option))
    if not comfy.wait_until_reachable():
        raise RuntimeError(f"ComfyUI isn't running at {comfy.url}. Start it first: the prompts are saved.")
    if args.resume:
        # A run that was killed outright leaves its last image rendering.
        comfy.cancel()
    # Saved after this, or a ComfyUI that's down would cost a redo's image without rendering anything.
    state.save()
    failed = render_images(state, comfy, most=1 if args.first_only else None)
    left = sum(1 for shot in state.shots if not shot.get("file"))
    if args.first_only and left and not failed:
        print(f"👀 The first one is made, to look at: continue for the {left} left. python -m pipelines.image.main --resume {state.id}")
        return 0
    if failed:
        print(f"💾 {failed} of {len(state.shots)} failed, and the rest are kept. Continue with: python -m pipelines.image.main --resume {state.id}")
        return 1
    if left and batch_stop.requested():
        print(f"💾 Stopped as asked, with {left} still to make. Continue with: python -m pipelines.image.main --resume {state.id}")
        return 0
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

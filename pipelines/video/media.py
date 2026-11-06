import hashlib
import io
import math
import os
import shutil

import ffmpeg
from PIL import Image, ImageOps

# The pixel count the workflow's models and memory use were tuned at (1280x736).
RENDER_AREA = 1280 * 736
DEFAULT_SHAPE = (1280, 736)  # With no image to take the shape from
# The workflow halves the size, then upscales: both have to stay on the model's grid.
SIZE_STEP = 32
# What the panel offers, landscape as (name, width, height), and portrait the other way round. On the
# grid, so as near 16:9 as it gets. The last is RENDER_AREA, which is as far as the panel's presets go.
SIZE_PRESETS = (("360p", 640, 352), ("480p", 864, 480), ("540p", 960, 544), ("720p", 1280, 736))
# The most pixels a model makes a clip of in a sensible time on this card: 960x544 (0.52 megapixels). A model left out
# takes any size up to RENDER_AREA.
MOST_PIXELS = {"minimax_h3": 960 * 544}
MAX_INPUT_EDGE = 2048
# Enough for the planner to see what's there, and small enough to send inline.
PREVIEW_EDGE = 1280
FPS = 24  # The workflow's frame rate
# A clip this much shorter than asked for was cut off.
MIN_LENGTH_SHARE = 0.8
# Fades through black between clips that start over on the input image.
FADE = 0.3
# Clips' sound doesn't line up where they join, which clicks without a tiny fade.
SEAM = 0.03
# How long a reference video or audio clip can be, in seconds: MiniMax H3 takes 2-15 s of either.
SHORTEST_REFERENCE, LONGEST_REFERENCE = 2, 15


def load_image(path):
    """The image the right way up (phones store it sideways plus a note), or raises if it isn't one."""
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except OSError as e:
        raise RuntimeError(f"Could not read the image {path}: {e}")


def audio_fingerprint(path=None, text=None, voice=None):
    """What identifies a film's audio: the file, or the text and who reads it. None without any."""
    if not path and not text:
        return None
    digest = hashlib.sha256(f"{text}{voice}".encode())
    if path:
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def inputs_fingerprint(images, image_roles, audio=None, references=()):
    """
    What identifies a film's inputs, so the same ones can be recognised however they were uploaded.
    references: (path, role) for each reference video or audio clip, in order.
    """
    digest = hashlib.sha256(str(audio).encode() if audio else b"")
    for path, role in references:
        with open(path, "rb") as f:
            digest.update(hashlib.sha256(f.read()).digest())
        digest.update(str(role).encode())
    for image in images:
        digest.update(f"{image.size}{image.mode}".encode())
        digest.update(image.tobytes())
    if len(images) > 1:
        # One role reads as it always did, so a run from before can still be continued.
        given = [image_roles] if isinstance(image_roles, str) else list(image_roles or [])
        digest.update(",".join(given[:len(images) - 1] or ["context"]).encode())
    return digest.hexdigest()


def save_input(image, path):
    copy = image.copy()
    copy.thumbnail((MAX_INPUT_EDGE, MAX_INPUT_EDGE))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    copy.save(path)


def preview_jpeg(image):
    copy = image.copy()
    copy.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def on_grid(value):
    return max(SIZE_STEP, round(value / SIZE_STEP) * SIZE_STEP)


def most_pixels(model):
    return MOST_PIXELS.get(model)


def size_problem(model, width, height):
    """Why a model won't make clips this size, or None."""
    most = most_pixels(model)
    if not most or width * height <= most:
        return None
    return (f"This workflow makes clips up to {most / 1e6:.2f} megapixels ({math.isqrt(most * 16 // 9) // SIZE_STEP * SIZE_STEP}"
            f"x{math.isqrt(most * 9 // 16) // SIZE_STEP * SIZE_STEP} at 16:9), and {width}x{height} is {width * height / 1e6:.2f}. "
            "Choose a smaller size.")


def render_size(width, height, most=RENDER_AREA):
    """The image's shape at the workflow's pixel count (or fewer, up to `most`), so its centre crop takes next to nothing off."""
    scale = math.sqrt(min(most, RENDER_AREA) / (width * height))
    size = on_grid(width * scale), on_grid(height * scale)
    if most < RENDER_AREA and size[0] * size[1] > most:
        # Rounded down, so the limit holds for any shape.
        size = tuple(max(SIZE_STEP, int(side * scale) // SIZE_STEP * SIZE_STEP) for side in (width, height))
    return size


def _probe(path):
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"]), {stream["codec_type"] for stream in probe["streams"]}


def clip_problem(path, seconds):
    """Returns what is wrong with the file, or None."""
    try:
        length, kinds = _probe(path)
    except (ffmpeg.Error, KeyError, ValueError) as e:
        return f"it can't be read ({e})"
    if "video" not in kinds:
        return "it has no video"
    if length < seconds * MIN_LENGTH_SHARE:
        return f"it is {length:.1f}s long instead of {seconds}s"
    return None


def last_frame(clip, path):
    """-update keeps overwriting the file, so what remains is the last frame the clip decodes to."""
    ffmpeg.input(clip, sseof=-0.5).output(path, update=1).run(overwrite_output=True, quiet=True)
    return path


def _make_part(clip, part, width, height, chained, fade):
    length, kinds = _probe(clip)
    source = ffmpeg.input(clip)
    video = source.video.filter("fps", fps=FPS)
    skip = 0.0
    if chained:
        # Its first frame is the last frame of the clip before: shown twice, it reads as a stutter.
        skip = 1 / FPS
        video = video.trim(start_frame=1).setpts("PTS-STARTPTS")
    length -= skip
    video = video.filter("scale", width, height).filter("setsar", 1)

    if "audio" in kinds:
        audio = source.audio.filter("atrim", start=skip).filter("asetpts", "PTS-STARTPTS")
    else:
        # Every part needs the same streams, or joining them drops the sound after this one.
        audio = ffmpeg.input("anullsrc=r=48000:cl=stereo", f="lavfi", t=length).audio
    seam = FADE if fade else SEAM
    audio = (
        audio.filter("aresample", 48000)
        .filter("aformat", channel_layouts="stereo")
        .filter("apad")
        .filter("atrim", end=length)
        .filter("afade", t="in", st=0, d=seam)
        .filter("afade", t="out", st=max(0.0, length - seam), d=seam)
    )
    if fade:
        video = video.filter("fade", t="in", st=0, d=FADE).filter("fade", t="out", st=max(0.0, length - FADE), d=FADE)

    ffmpeg.output(video, audio, part, vcodec="libx264", crf=18, preset="medium", pix_fmt="yuv420p",
                  acodec="aac", audio_bitrate="192k").run(overwrite_output=True, quiet=True)


def stitch(clips, parts_dir, output_path, width, height, chained):
    """
    Evens the clips out into same-format parts, then joins those without encoding
    again. Joining 40 clips in one filter graph would hold them all in memory at
    once, and a failure would start over from nothing.
    """
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    names = []
    for number, clip in enumerate(clips, 1):
        print(f"   ⚙️ Preparing clip {number}/{len(clips)}")
        name = f"part_{number:03d}.mp4"
        _make_part(clip, os.path.join(parts_dir, name), width, height, chained and number > 1, not chained)
        names.append(name)

    list_path = os.path.join(parts_dir, "parts.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.writelines(f"file '{name}'\n" for name in names)

    part_path = output_path + ".part"
    ffmpeg.input(list_path, f="concat", safe=0).output(part_path, c="copy", format="mp4", movflags="faststart") \
        .run(overwrite_output=True, quiet=True)
    os.replace(part_path, output_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return output_path


def to_wav(path, output):
    """Any audio or video file as the WAV a run keeps, so ComfyUI and ffmpeg can both read it."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    try:
        ffmpeg.input(path).output(output, vn=None, ac=2, ar=48000).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error:
        raise RuntimeError("That audio file can't be read.")
    return output


def audio_slice(path, start, seconds, output):
    """One shot's stretch of the audio, padded with silence to the clip's length."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    (ffmpeg.input(path, ss=start).filter("apad", whole_dur=seconds)
     .output(output, t=seconds, ac=2, ar=48000).run(overwrite_output=True, quiet=True))
    return output


def with_audio(video_path, audio_path):
    """Puts this audio on the film, in place of any the clips came with. Not encoding the picture again."""
    part_path = video_path + ".part"
    video, audio = ffmpeg.input(video_path).video, ffmpeg.input(audio_path).audio
    (ffmpeg.output(video, audio, part_path, vcodec="copy", acodec="aac", shortest=None, format="mp4", movflags="faststart")
     .run(overwrite_output=True, quiet=True))
    os.replace(part_path, video_path)


def duration(path):
    return _probe(path)[0]


def media_kind(path):
    """"video" for a file with moving pictures, "audio" for sound alone (an album cover doesn't count), or None."""
    try:
        streams = ffmpeg.probe(path)["streams"]
    except (ffmpeg.Error, KeyError, ValueError):
        return None
    if any(stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic") for stream in streams):
        return "video"
    return "audio" if any(stream.get("codec_type") == "audio" for stream in streams) else None


def _reference_length(path, what):
    try:
        length = _probe(path)[0]
    except (ffmpeg.Error, KeyError, ValueError):
        raise RuntimeError(f"That {what} can't be read.")
    if length < SHORTEST_REFERENCE:
        raise RuntimeError(f"A reference {what} has to be at least {SHORTEST_REFERENCE} seconds long, and this one is {length:.1f}s.")
    return min(length, LONGEST_REFERENCE)


def reference_video(path, output):
    """
    A reference video as the clips get it: 24 fps, the frame rate H3 reads references at, and its first
    15 seconds at most. Returns whether it has sound of its own, which H3 takes as a reference too.
    """
    length = _reference_length(path, "video")
    has_sound = "audio" in _probe(path)[1]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    source = ffmpeg.input(path, t=length)
    streams = [source.video.filter("fps", fps=FPS)] + ([source.audio] if has_sound else [])
    ffmpeg.output(*streams, output, vcodec="libx264", crf=18, preset="medium", pix_fmt="yuv420p",
                  **({"acodec": "aac", "audio_bitrate": "192k"} if has_sound else {})).run(overwrite_output=True, quiet=True)
    return has_sound


def reference_audio(path, output):
    """A reference audio clip as the clips get it: its first 15 seconds at most, as WAV."""
    length = _reference_length(path, "audio clip")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ffmpeg.input(path, t=length).output(output, vn=None, ac=2, ar=48000).run(overwrite_output=True, quiet=True)
    return output

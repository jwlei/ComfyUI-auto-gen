import io

import ffmpeg
from PIL import Image, ImageFilter

# Frames looked at per clip, spread over its length.
SAMPLES = 5
SHEET_FRAME = (270, 480)
MEASURED_FRAME = (180, 320)
# A pixel this close to its neighbours counts as flat, in brightness out of 255.
FLAT_EDGE = 6
# Share of a frame that's flat before a clip is worth asking about. Measured on 39 finished
# videos: their text and drawings on a colour reach 0.90-1.00, some real footage does too
# (night, sky), and 90% of all their frames stay under 0.80.
SUSPECT_FLATNESS = 0.89


def _frame_at(path, second):
    width, height = SHEET_FRAME
    raw, _ = (
        ffmpeg.input(path, ss=second)
        # As the composer crops it, so this measures what would be on screen.
        .filter('scale', width, height, force_original_aspect_ratio='increase')
        .filter('crop', width, height)
        .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    return Image.frombytes("RGB", SHEET_FRAME, raw) if len(raw) == width * height * 3 else None


def sample_frames(path):
    """Frames from across the clip. Empty if it can't be read."""
    try:
        seconds = float(ffmpeg.probe(path)['format']['duration'])
        frames = [_frame_at(path, seconds * (i + 0.5) / SAMPLES) for i in range(SAMPLES)]
    except (ffmpeg.Error, KeyError, ValueError):
        return []
    return [frame for frame in frames if frame]


def flatness(frame):
    """Share of the frame with almost no local contrast: flat colour and smooth gradients alike."""
    edges = frame.convert("L").resize(MEASURED_FRAME).filter(ImageFilter.FIND_EDGES)
    # The filter leaves the outermost row of pixels unfiltered.
    histogram = edges.crop((1, 1, MEASURED_FRAME[0] - 1, MEASURED_FRAME[1] - 1)).histogram()
    return sum(histogram[:FLAT_EDGE]) / sum(histogram)


def contact_sheet(frames):
    """The frames side by side as one JPEG, so a question about the clip takes one image."""
    width, height = SHEET_FRAME
    sheet = Image.new("RGB", (width * len(frames), height))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (width * i, 0))
    jpeg = io.BytesIO()
    sheet.save(jpeg, "JPEG", quality=80)
    return jpeg.getvalue()


def is_text_card(path, shows_only_text):
    """
    Whether a stock clip is just text on a colour. Pixels can't tell words on black from a
    city at night, so only a clip with a flat frame goes to shows_only_text (ContentBrain's),
    and nearly every clip costs no request. A clip that can't be read isn't called one.
    """
    frames = sample_frames(path)
    if not frames or max(map(flatness, frames)) < SUSPECT_FLATNESS:
        return False
    return shows_only_text(contact_sheet(frames))

import os
import re
import sys

import matplotlib

# Picked before pyplot is imported anywhere below: Agg has no window to open,
# so recording never steals focus and works with no display at all.
matplotlib.use("Agg")

from matplotlib.animation import FFMpegWriter  # noqa: E402

from run_simulation import DEFAULT_SETUP  # noqa: E402
from setup import read_setup_file  # noqa: E402
from visualization import make_scene  # noqa: E402


def video_name(title):
    # The title names the run, so it names the file.
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{slug or 'simulation'}.mp4"


def record(setup_file, output_file=None):
    setup_data = read_setup_file(setup_file)
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(setup_file), video_name(setup_data["title"])
        )

    fig, update, frames, frame_time = make_scene(setup_file)

    # One frame per frame_time seconds of simulated time, so the video plays
    # back at real speed. Taking fps from the config instead would be a little
    # off, since steps_per_frame has to be a whole number of steps.
    writer = FFMpegWriter(
        fps=1.0 / frame_time, metadata={"title": setup_data["title"]}
    )
    with writer.saving(fig, output_file, dpi=120):
        writer.grab_frame()
        for _ in range(frames):
            update()
            writer.grab_frame()

    return output_file, frames * frame_time


if __name__ == "__main__":
    setup_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETUP
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    path, duration = record(setup_file, output_file)
    print(f"wrote {path}  ({duration:.1f} s)")

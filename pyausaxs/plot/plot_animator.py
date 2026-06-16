# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""GIF animation support for the `plot` command.

The rigid-body backend can emit an enumerated sequence of plot files
(`name_0.png.plot`, `name_1.png.plot`, ...). The `plot` command renders every
`.plot` file to a `.png` as usual; afterwards, when a directory holds more than
`GIF_MIN_FRAMES` such frames sharing a stem, their rendered images are stitched
into one `name.gif`.

Sequences are detected from the `.png.plot` data files only (never the rendered
`.png`), so the result is stable when `plot` is run twice over the same directory.
"""

from __future__ import annotations

import re

from PIL import Image

# a directory needs more than this many matching frames before it is animated
GIF_MIN_FRAMES = 5

# matches "<stem>_<index>.png.plot"; the ".png" rendered output is intentionally not matched
_FRAME_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)\.png\.plot$")

# without --fps/--duration, a gif targets roughly this length: more frames simply play faster
_DEFAULT_TOTAL_SECONDS = 10.0
_MIN_FRAME_MS = 20    # cap at 50 fps so a long sequence never yields a 0 ms delay
_MAX_GIF_WIDTH = 1200  # standalone plots render at ~6000 px; downscale frames for a sane gif


def enumerated_groups(filenames: list[str]) -> dict[str, list[str]]:
    """Group `<stem>_<n>.png.plot` filenames by stem, ordered by the index n.

    Returns {stem: [filenames ordered by index]} for groups of any size; the
    caller decides which groups are large enough to animate.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for name in filenames:
        match = _FRAME_RE.match(name)
        if match:
            groups.setdefault(match.group("stem"), []).append((int(match.group("index")), name))
    return {stem: [name for _, name in sorted(items)] for stem, items in groups.items()}


def animation_groups(filenames: list[str]) -> dict[str, list[str]]:
    """Like `enumerated_groups`, but only the groups large enough to be animated."""
    return {stem: frames for stem, frames in enumerated_groups(filenames).items()
            if len(frames) >= GIF_MIN_FRAMES}


def frame_duration_ms(n_frames: int, fps: float | None = None, duration: float | None = None) -> int:
    """Per-frame delay in milliseconds.

    `fps` sets the rate directly; `duration` sets the total animation length in
    seconds; with neither, the gif targets a fixed default length.
    """
    if n_frames <= 0:
        return _MIN_FRAME_MS
    if fps is not None:
        per_frame = 1000.0 / fps
    else:
        total = duration if duration is not None else _DEFAULT_TOTAL_SECONDS
        per_frame = total * 1000.0 / n_frames
    return max(int(round(per_frame)), _MIN_FRAME_MS)


def assemble_gif(frame_paths: list[str], output: str, duration_ms: int, loop: int = 0) -> None:
    """Stitch already-rendered frame PNGs (in order) into a GIF, downscaling for size."""
    images = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        if image.width > _MAX_GIF_WIDTH:
            height = round(image.height * _MAX_GIF_WIDTH / image.width)
            image = image.resize((_MAX_GIF_WIDTH, height), Image.LANCZOS)
        images.append(image)
    images[0].save(
        output, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=loop, disposal=2, optimize=True,
    )

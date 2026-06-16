# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""GIF animation support for the `plot` command.

The rigid-body backend can emit an enumerated sequence of plot files
(`name_0.png.plot`, `name_1.png.plot`, ...). When a directory holds more than
`GIF_MIN_FRAMES` such frames sharing a stem, the `plot` command pulls them out of
the normal per-file rendering and instead renders them into one `name.gif`.

Pulling them out up front lets us enforce settings *across* the whole sequence
that a single frame cannot know about — in particular a shared residuals y-range,
so the residuals subpanel does not jump around from frame to frame.

Sequences are detected from the `.png.plot` data files only (never the rendered
`.png`), so the result is stable when `plot` is run twice over the same directory.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image

try:
    from .plot_helper import PlotType, read_dataset, render_plot
except ImportError:
    from plot_helper import PlotType, read_dataset, render_plot

# a directory needs more than this many matching frames before it is animated
GIF_MIN_FRAMES = 5

# matches "<stem>_<index>.png.plot"; the ".png" rendered output is intentionally not matched
_FRAME_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)\.png\.plot$")

# without --fps/--duration, a gif targets roughly this length: more frames simply play faster
_DEFAULT_TOTAL_SECONDS = 10.0
_MIN_FRAME_MS = 20   # cap at 50 fps so a long sequence never yields a 0 ms delay
_FRAME_DPI = 100     # frames are only ever seen as gif: keep them light


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


def _frame_arrays(path: str):
    """Return (I, Ierr, Imodel) of a residuals .plot frame, or None if it is not one."""
    with open(path) as f:
        if f.readline().rstrip() != PlotType.Residuals.value:
            return None
        data = read_dataset(f).data
    if data.ndim != 2 or data.shape[1] < 4:
        return None
    return data[:, 1], data[:, 2], data[:, 3]


def locked_residual_axes(frame_paths: list[str], res_margin: float = 0.05, int_margin: float = 0.1):
    """Compute shared (intensity_ylim, residual_ylim) spanning every frame in a sequence.

    Both panels are locked so neither the intensity (top) nor the residuals (bottom)
    drift between frames. Returns (None, None) if the frames are not residuals plots,
    in which case the backend's own axis limits are kept."""
    i_low = i_high = r_low = r_high = None
    for path in frame_paths:
        arrays = _frame_arrays(path)
        if arrays is None:
            return None, None
        I, Ierr, Imodel = arrays
        residuals = (I - Imodel) / Ierr
        r_low = np.min(residuals) if r_low is None else min(r_low, np.min(residuals))
        r_high = np.max(residuals) if r_high is None else max(r_high, np.max(residuals))
        positive = np.concatenate([I, Imodel])
        positive = positive[positive > 0]  # the intensity panel is typically log-scaled
        if positive.size:
            i_low = positive.min() if i_low is None else min(i_low, positive.min())
            i_high = positive.max() if i_high is None else max(i_high, positive.max())

    if r_low is None:
        return None, None
    r_pad = (r_high - r_low) * res_margin or 1.0
    residual_ylim = (float(r_low - r_pad), float(r_high + r_pad))
    intensity_ylim = None
    if i_low is not None:
        # multiplicative margin, so it reads well on a log axis
        intensity_ylim = (float(i_low / (1 + int_margin)), float(i_high * (1 + int_margin)))
    return intensity_ylim, residual_ylim


def render_frame_png(file: str, intensity_ylim=None, residual_ylim=None) -> bytes:
    """Render one .plot frame to in-memory PNG bytes (nothing written to disk).

    Module-level so it can be dispatched to a ProcessPoolExecutor. For a residuals
    frame the intensity (top) and residuals (bottom) panels are locked to the given
    ranges so both axes stay fixed across the animation."""
    plt.figure()
    render_plot(file)
    figure = plt.gcf()
    if len(figure.axes) >= 2:  # a residuals plot: [top intensity, bottom residuals]
        if intensity_ylim is not None:
            figure.axes[0].set_ylim(intensity_ylim)
        if residual_ylim is not None:
            figure.axes[1].set_ylim(residual_ylim)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=_FRAME_DPI)
    plt.close()
    return buffer.getvalue()


def assemble_gif(frames_png: list[bytes], output: str, duration_ms: int, loop: int = 0) -> None:
    """Stitch rendered frame PNGs (already in order) into a GIF."""
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames_png]
    images[0].save(
        output, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=loop, disposal=2, optimize=True,
    )

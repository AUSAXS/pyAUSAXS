# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Render AUSAXS output files (.fit and .plot) into matplotlib Figures for embedding."""

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from ..plot.plot_helper import (
    PlotType, Dataset, Hline, Vline,
    read_dataset, read_hline, read_vline, read_2dhist,
)
from .theme import PALETTE

# matplotlib rc overrides for a clean, modern look on embedded figures
PLOT_RC = {
    "figure.facecolor":  PALETTE["surface"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        PALETTE["border"],
    "grid.linewidth":    0.6,
    "grid.alpha":        0.6,
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "xtick.labelcolor":  PALETTE["text"],
    "ytick.labelcolor":  PALETTE["text"],
    "text.color":        PALETTE["text"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
}


def _new_figure() -> Figure:
    fig = Figure(facecolor=PALETTE["surface"])
    return fig

# pretty tab titles for known plot file stems
PRETTY_NAMES = {
    "p(r)": "p(r)",
    "profiles": "partial profiles",
    "chi2_evaluated_points_full": "χ² landscape",
    "chi2_evaluated_points_limited": "χ² reduced axes",
    "chi2_near_minimum": "χ² near minimum",
    "chi2_evaluated_points_limited_mass": "χ² reduced axes (mass)",
    "chi2_near_minimum_mass": "χ² near minimum (mass)",
}


def parse_fit_file(path: str) -> tuple[np.ndarray, dict[str, str]]:
    """Parse an ausaxs.fit file into a (n, 4) array of [q, I, Ierr, Imodel] plus header info."""
    info: dict[str, str] = {}
    rows = []
    with open(path) as f:
        for line in f:
            words = line.split()
            if len(words) < 4:
                for word in words:
                    if "=" in word:
                        key, _, value = word.partition("=")
                        info[key] = value
                continue
            try:
                rows.append([float(w) for w in words[:4]])
            except ValueError:
                continue
    data = np.array(rows)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unexpected fit file format: \"{path}\"")
    return data, info


def fit_figure(path: str, logx: bool = False) -> Figure:
    """Build the data + model + residuals figure from an ausaxs.fit file."""
    data, info = parse_fit_file(path)
    q, I, Ierr, Imodel = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    label = "AUSAXS fit"
    if "chi2" in info:
        try:
            label = f"$\\chi^2_r = {float(info['chi2']):.3f}$"
        except ValueError:
            pass

    with mpl.rc_context(PLOT_RC):
        fig = _new_figure()
        ax, ax_res = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        ax.errorbar(q, I, yerr=Ierr, fmt=".", color=PALETTE["muted"],
                    markersize=4, capsize=2, elinewidth=0.8, zorder=0)
        ax.plot(q, Imodel, color=PALETTE["accent"], lw=1.8, zorder=5, label=label)
        ax.set_yscale("log")
        ax.set_ylabel("I(q)")
        ax.legend()

        ax_res.axhline(0, color=PALETTE["muted"], lw=0.8)
        ax_res.plot(q, (I - Imodel)/Ierr, ".", color=PALETTE["accent"], markersize=4)
        ax_res.set_xlabel("q [$\\AA^{-1}$]")
        ax_res.set_ylabel("Residuals")
        if logx:
            ax.set_xscale("log")

        fig.set_layout_engine("tight")
    return fig


def _render_dataset(ax, d: Dataset, first: bool):
    o = d.options
    if o.dof != 0:
        d.data[:, 1] = d.data[:, 1]/o.dof
    if o.xshift != 0:
        d.data[:, 0] += o.xshift

    common = dict(label=o.legend if o.legend else None, zorder=o.zorder)
    if o.drawerror and 3 <= d.data.shape[1]:
        ax.errorbar(
            d.data[:, 0], d.data[:, 1], yerr=d.data[:, 2],
            color=o.color, linestyle="none", marker=o.markerstyle,
            markersize=2*o.markersize, capsize=2, **(common | {"zorder": 5})
        )
    elif o.drawline and o.drawmarker:
        ax.plot(
            d.data[:, 0], d.data[:, 1], color=o.color, linestyle=o.linestyle,
            linewidth=o.linewidth, marker=o.markerstyle, markersize=2*o.markersize, **common
        )
    elif o.drawmarker:
        ax.plot(
            d.data[:, 0], d.data[:, 1], color=o.color, linestyle="none",
            marker=o.markerstyle, markersize=2*o.markersize, **common
        )
    elif o.drawline:
        ax.plot(
            d.data[:, 0], d.data[:, 1], color=o.color,
            linestyle=o.linestyle, linewidth=o.linewidth, **common
        )

    if first:
        if o.title:
            ax.set_title(o.title)
        ax.set_xlabel(rf"{o.xlabel}")
        ax.set_ylabel(rf"{o.ylabel}")
        if o.xrange:
            ax.set_xlim(o.xrange)
        if o.yrange:
            ax.set_ylim(o.yrange)
        if o.xlog:
            ax.set_xscale("log")
        if o.ylog:
            ax.set_yscale("log")


def plot_file_figure(path: str) -> Figure:
    """Render a .plot file into a Figure. Mirrors plot_helper.plot_file, but draws
    on a standalone Figure instead of saving a png through pyplot."""
    with mpl.rc_context(PLOT_RC):
        fig = _new_figure()
        ax = None
        first = True
        has_legend = False

        with open(path) as f:
            while line := f.readline():
                try:
                    kind = PlotType(line.rstrip())
                except ValueError:
                    raise ValueError(f"Invalid plot type \"{line.rstrip()}\" in \"{path}\"")

                match kind:
                    case PlotType.Dataset | PlotType.Histogram:
                        d = read_dataset(f)
                        if d.data.size == 0:
                            continue
                        if ax is None:
                            ax = fig.add_subplot(111)
                        _render_dataset(ax, d, first)
                        first = False
                        has_legend = has_legend or bool(d.options.legend)

                    case PlotType.Hline:
                        h: Hline = read_hline(f)
                        if ax is None:
                            ax = fig.add_subplot(111)
                        ax.axhline(h.y, color=h.options.color, linestyle=h.options.linestyle,
                                   linewidth=h.options.linewidth, label=h.options.legend or None)
                        has_legend = has_legend or bool(h.options.legend)

                    case PlotType.Vline:
                        v: Vline = read_vline(f)
                        if ax is None:
                            ax = fig.add_subplot(111)
                        ax.axvline(v.x, color=v.options.color, linestyle=v.options.linestyle,
                                   linewidth=v.options.linewidth, label=v.options.legend or None)
                        has_legend = has_legend or bool(v.options.legend)

                    case PlotType.Landscape:
                        d = read_dataset(f)
                        ax3d = fig.add_subplot(111, projection="3d")
                        x, y, z = d.data[:, 0], d.data[:, 1], d.data[:, 2]
                        ax3d.scatter(x, y, z, c=z, cmap="coolwarm")
                        ax3d.set_xlabel(rf"{d.options.xlabel}")
                        ax3d.set_ylabel(rf"{d.options.ylabel}")
                        ax3d.set_zlabel(rf"{d.options.zlabel}")

                    case PlotType.Image:
                        x, y, z, _opts = read_2dhist(f)
                        if ax is None:
                            ax = fig.add_subplot(111)
                        X, Y = np.meshgrid(x, y)
                        cf = ax.contourf(X, Y, z, 100, cmap="coolwarm")
                        fig.colorbar(cf, ax=ax)

                    case _:
                        raise ValueError(f"Unsupported plot type \"{kind}\" in \"{path}\"")

        if ax is not None and has_legend:
            ax.legend()
        fig.set_layout_engine("tight")
    return fig


def pretty_plot_name(stem: str) -> str:
    # plot files are named e.g. "p(r).png.plot", so the stem may keep an image extension
    if stem.endswith(".png"):
        stem = stem[:-4]
    return PRETTY_NAMES.get(stem, stem)

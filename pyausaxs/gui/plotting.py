# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Render AUSAXS output files (.fit and .plot) into matplotlib Figures for embedding."""

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d  # noqa: F401  (registers the '3d' projection)

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


def _fit_figure(q, I, Ierr, Imodel, label: str, logx: bool) -> Figure:
    """Build a data + model + residuals figure from arrays."""
    with mpl.rc_context(PLOT_RC):
        fig = _new_figure()
        ax, ax_res = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        ax.errorbar(q, I, yerr=Ierr, fmt=".", color=PALETTE["muted"], markersize=4, capsize=2, elinewidth=0.8, zorder=0)
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


def fit_figure(path: str, logx: bool = False) -> Figure:
    """Build the data + model + residuals figure from an ausaxs.fit file."""
    data, info = parse_fit_file(path)
    label = "AUSAXS fit"
    if "chi2" in info:
        try:
            label = f"$\\chi^2_r = {float(info['chi2']):.3f}$"
        except ValueError:
            pass
    return _fit_figure(data[:, 0], data[:, 1], data[:, 2], data[:, 3], label, logx)


def fit_figure_from_curves(data: np.ndarray, logx: bool = False) -> Figure:
    """Build a fit figure from an (n, 4) array of [q, I, Ierr, I_model] curves."""
    q, I, Ierr, Imodel = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(((I - Imodel)/Ierr)**2)
    label = f"$\\chi^2_r = {chi2/max(len(q), 1):.3f}$"
    return _fit_figure(q, I, Ierr, Imodel, label, logx)


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


# distinct body colours, deliberately excluding red (reserved for the split residues)
_BODY_COLORS = ["#4a7dbd", "#e89a3c", "#46a86c", "#9467bd", "#17becf", "#8c564b", "#bcbd22", "#7f7f7f"]


def draw_structure(ax, data: dict, split_residues: list[int], *,
                   show_atoms: bool = False, show_copies: bool = True,
                   show_constraints: bool = True, highlight: set[tuple[int, int | None]] | None = None,
                   color_by: str = "body", body_names: dict[int, str] | None = None):
    """Draw a rigid-body structure preview on a 3D axis from a backend preview-structure dict (see Rigidbody.preview_structure).
    The Cα backbone is drawn per body (one colour each) with symmetry copies faded, and the split residues marked in red.
    Authoritative body/Cα/residue metadata comes from the backend, so it works for wildcards, multi-file loads and symmetry alike.

    Options (all default to the rigid-body pane's original behaviour):
        show_atoms       — also draw every atom as a faint cloud (atomic detail), not just the Cα trace
        show_copies      — draw symmetry copies (copy > 0); when False only the originals are shown
        show_constraints — draw the constraint tethers / attractor-repulsor arrows
        highlight         — a set of (body, copy) selectors to keep lit while everything else is dimmed
                           (copy=None selects the whole body, i.e. every one of its copies); empty/None
                           means nothing is dimmed
        color_by         — "body" (a colour per body) or "copy" (a colour per symmetry copy)
        body_names       — body index -> display name, used to label bodies with no Cα atoms (falls back
                           to "b{index+1}" for any body missing from the mapping)
    """
    body_names = body_names or {}
    highlight = highlight or set()
    coords = data["coords"]
    body, copy, res, is_ca = data["body"], data["copy"], data["residue_seq"], data["is_ca"]
    splits = sorted({int(s) for s in split_residues})

    def _dimmed(b: int, c: int | None = None) -> bool:
        if not highlight:
            return False
        if c is None:  # copy-agnostic check (e.g. the all-atom cloud): lit if any selector names this body
            return not any(bb == b for bb, _cc in highlight)
        return (b, None) not in highlight and (b, c) not in highlight

    def _colour(b: int, c: int) -> str:
        idx = c if color_by == "copy" else b
        return _BODY_COLORS[idx % len(_BODY_COLORS)]

    # optional faint all-atom cloud, drawn beneath the backbone; copies included only if shown
    if show_atoms:
        cloud = (copy == 0) if not show_copies else np.ones(len(coords), dtype=bool)
        for b in sorted(set(body[cloud].tolist())):
            m = cloud & (body == b)
            if not m.any():
                continue
            ax.scatter(
                coords[m, 0], coords[m, 1], coords[m, 2], s=2, color=_colour(b, 0),
                alpha=0.04 if _dimmed(b) else 0.22, edgecolors="none", depthshade=False, zorder=0
            )

    # Cα backbone, drawn separately per (body, copy) so traces never bridge bodies or copies
    for b in sorted(set(body[is_ca].tolist())):
        for c in sorted(set(copy[is_ca & (body == b)].tolist())):
            if not show_copies and c != 0:
                continue
            pts = coords[is_ca & (body == b) & (copy == c)]
            if len(pts) == 0:
                continue
            original = (c == 0)
            if _dimmed(b, c):
                lw, alpha, z = 0.8, 0.12, 1
            else:
                lw, alpha, z = (1.0, 1.0, 2) if original else (0.8, 0.65, 1)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=_colour(b, c), lw=lw, alpha=alpha, zorder=z)

    # bodies with no Cα atoms at all (e.g. a non-protein hetero group) draw nothing in the trace above
    # and would otherwise be completely invisible; mark each with a small sphere at the centroid
    # instead. The sphere is real geometry sized in data units (Angstrom), not a screen-space
    # scatter marker, so it pans/zooms consistently with the rest of the structure instead of
    # visually dominating once the view zooms out.
    ca_bodies = set(body[is_ca].tolist())
    no_ca_bodies = sorted(set(body.tolist()) - ca_bodies)
    MIN_RADIUS = 25  # floor so single-atom bodies (Rg=0) stay visible
    for b in no_ca_bodies:
        pts0 = coords[(body == b) & (copy == 0)]
        if len(pts0) > 1:
            rg = float(np.sqrt(np.mean(np.sum((pts0 - pts0.mean(axis=0)) ** 2, axis=1))))
        else:
            rg = 0.0
        # approximate the group's physical extent from its Rg, treating it as a solid sphere
        # (Rg = sqrt(3/5)*R  =>  R = sqrt(5/3)*Rg)
        radius = 0.1*max(rg * np.sqrt(5 / 3), MIN_RADIUS)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 6))
        sx, sy, sz = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)

        for c in sorted(set(copy[body == b].tolist())):
            if not show_copies and c != 0:
                continue
            pts = coords[(body == b) & (copy == c)]
            if len(pts) == 0:
                continue
            centroid = pts.mean(axis=0)
            original = (c == 0)
            if _dimmed(b, c):
                alpha = 0.15
            else:
                alpha = 1.0 if original else 0.65
            colour = _colour(b, c)
            ax.plot_surface(
                centroid[0] + radius * sx, centroid[1] + radius * sy, centroid[2] + radius * sz,
                color=colour, alpha=alpha, linewidth=0, shade=True, zorder=2
            )
            if original:
                label = body_names.get(b, f"b{b + 1}")
                ax.text(*centroid, f" {label}", color=colour, alpha=alpha, fontsize=7, zorder=2)

    # split-residue markers on the originals, in red
    highlight = is_ca & (copy == 0) & np.isin(res, splits)
    if highlight.any():
        ax.scatter(
            coords[highlight, 0], coords[highlight, 1], coords[highlight, 2], s=80,
            color="red", edgecolors="black", linewidths=0.6, depthshade=False, zorder=3
        )

    # active constraints, all in black: a dashed tether for backbone (0) / centre-of-mass (1)
    # constraints (told apart by length — CM ones span much further), and a solid line with
    # directional arrowheads for attractors (2, pointing inward) and repulsors (3, outward).
    # indices reference copy 0, so they map straight onto the rows of `coords`.
    constraints = data.get("constraints") if show_constraints else None
    if constraints is not None and len(constraints):
        n = len(coords)
        scale = float((coords.max(0) - coords.min(0)).max()) or 1.0
        for idx1, idx2, ctype in constraints:
            if not (0 <= idx1 < n and 0 <= idx2 < n):
                continue
            p1, p2 = coords[idx1], coords[idx2]
            if ctype in (0, 1):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color="black", ls="--", lw=1.0, zorder=4)
                continue
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="black", lw=1.2, zorder=4)
            d = p2 - p1
            dist = float(np.linalg.norm(d))
            if dist == 0:
                continue
            d /= dist
            length = min(0.15 * scale, 0.4 * dist)
            inward = (ctype == 2)  # attractor pulls together; repulsor pushes apart
            for tail, direction in ((p1, d if inward else -d), (p2, -d if inward else d)):
                ax.quiver(tail[0], tail[1], tail[2], direction[0], direction[1], direction[2],
                          length=length, normalize=True, color="black",
                          arrow_length_ratio=0.45, lw=1.2, zorder=5)

    # equal aspect over every atom so symmetry copies are never clipped out of view
    span = float((coords.max(0) - coords.min(0)).max()) / 2 or 1.0
    mid = (coords.max(0) + coords.min(0)) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

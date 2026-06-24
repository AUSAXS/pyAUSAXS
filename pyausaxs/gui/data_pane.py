# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .panes import QMIN, QMAX, _read_saxs_data
from .theme import PALETTE
from .widgets import RangeSlider


class SaxsDataPane(ttk.Frame):
    """Interactive data-inspection pane for a single SAXS dataset.

    Shown as a sibling notebook tab named after the file stem. Hosts an I(q)
    plot whose x-axis is replaced by a RangeSlider overlaid on the figure's
    reserved bottom margin: two axvlines mark the selected q-range and continue
    as dashed risers into the slider's handles. Call qrange() to read (qmin, qmax)."""

    def __init__(self, parent, file_path: str):
        super().__init__(parent)
        self.file_path = file_path
        self.title = os.path.splitext(os.path.basename(file_path))[0]
        self._vlines = [None, None]
        self._track_pads = (None, None)
        self._bottom_frac = None

        # read data first so we can set the slider bounds from the actual q-range
        data = _read_saxs_data(file_path)
        if data:
            qs, _, _ = data
            vmin, vmax = min(qs), max(qs)
        else:
            vmin, vmax = QMIN, QMAX

        # --- figure ---
        self._fig = Figure(facecolor=PALETTE["surface"])
        self._ax = self._fig.add_subplot(111)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._mpl_widget = self._mpl_canvas.get_tk_widget()
        self._mpl_widget.pack(fill="both", expand=True, padx=6, pady=6)

        # --- slider, overlaid on the figure's reserved bottom margin ---
        # the slider replaces the plot's x-axis: it is laid over the bottom strip of
        # the figure (blended to the surface colour) and the plot's axvlines continue
        # as dashed risers down to its handles, so dragging feels like grabbing the plot.
        self.q_slider = RangeSlider(
            self, vmin, vmax,
            log=True, fmt="{:.4g}", stem=True,
            on_change=self._on_range_changed,
        )
        self.q_slider.configure(style="Card.TFrame")
        self.q_slider.canvas.configure(background=PALETTE["surface"])
        self.q_slider.place(in_=self._mpl_widget, relx=0, rely=1.0,
                            relwidth=1.0, anchor="sw")

        self._draw_data(data, vmin, vmax)

        # Keep the slider track aligned with the plot's x-axis and the figure's bottom
        # margin matched to the slider height. Both depend on final widget geometry, which
        # settles in stages: the figure widget reaches full size and draws while the
        # overlaid slider canvas is still 1px, and no further draw fires once the canvas
        # catches up. So drive the layout from every settling point — the figure's draws
        # and both widgets' <Configure>/<Map> — each guarded to no-op until ready. (Don't
        # bind <Configure> on the figure widget to drive its resize: that would replace
        # matplotlib's own resize handler and freeze the figure size.)
        self._fig.canvas.mpl_connect("draw_event", lambda _e: self._align_slider())
        self._mpl_widget.bind("<Configure>", lambda _e: self._relayout(), add="+")
        self.q_slider.canvas.bind("<Configure>", lambda _e: self._relayout(), add="+")
        self.bind("<Map>", lambda _e: self._relayout(), add="+")

    # ------------------------------------------------------------------
    def qrange(self) -> tuple[float, float]:
        return self.q_slider.values()

    # ------------------------------------------------------------------
    def _relayout(self):
        """Re-reserve the figure's bottom margin and re-align the slider. Safe to call
        from any of the staggered geometry events; each step no-ops until ready."""
        self._reserve_bottom()
        self._align_slider()

    # ------------------------------------------------------------------
    def _reserve_bottom(self):
        """Reserve a bottom strip of the figure so the axes stop exactly where the
        overlaid slider's canvas begins — then the axvlines (clipped at the axes bottom)
        meet the slider's risers with no seam. The strip is measured from the slider's
        actual on-screen position, and re-run on resize because it is a figure fraction
        while the slider keeps a fixed pixel height."""
        fig_h = self._mpl_widget.winfo_height()
        canvas = self.q_slider.canvas
        if fig_h <= 1 or not canvas.winfo_ismapped():
            return
        canvas_top = canvas.winfo_rooty() - self._mpl_widget.winfo_rooty()  # from figure top
        frac = min(max((fig_h - canvas_top) / fig_h, 0.05), 0.6)
        if self._bottom_frac is None or abs(frac - self._bottom_frac) > 0.004:
            self._bottom_frac = frac
            self._fig.get_layout_engine().set(rect=(0, frac, 1, 1))
            self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _align_slider(self):
        """Match the slider's track padding to the axes' horizontal extent so the
        handles line up with the plotted q-range. Driven by matplotlib's draw_event,
        so it follows tight-layout and resizes automatically.

        The slider canvas and the figure widget need not share a width or left
        offset, so the axes' figure-fraction box is mapped through the figure
        widget's own width and translated into the slider's frame via absolute
        screen x — multiplying the fractions by the slider width instead skews the
        far (right) handle in proportion to any width mismatch."""
        mpl_widget = self._mpl_widget
        slider_canvas = self.q_slider.canvas
        fig_w = mpl_widget.winfo_width()
        slider_w = slider_canvas.winfo_width()
        if fig_w <= 1 or slider_w <= 1:  # widgets not realised yet; a later draw retries
            return
        pos = self._ax.get_position()  # axes rect in figure-fraction coords
        dx = mpl_widget.winfo_rootx() - slider_canvas.winfo_rootx()  # figure left in slider frame
        pads = (round(dx + pos.x0 * fig_w),
                round(slider_w - dx - pos.x1 * fig_w))
        if pads != self._track_pads:  # avoid redundant slider redraws on every draw
            self._track_pads = pads
            self.q_slider.set_track_pads(*pads)

    # ------------------------------------------------------------------
    def _draw_data(self, data, vmin: float, vmax: float):
        ax = self._ax
        p = PALETTE
        ax.clear()
        self._fig.patch.set_facecolor(p["surface"])
        ax.set_facecolor(p["surface"])
        ax.tick_params(colors=p["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(p["border"])
        ax.set_ylabel("I(q)", color=p["muted"], fontsize=9)

        if data:
            qs, Is, sigs = data
            has_sigma = any(s > 0 for s in sigs)
            if has_sigma:
                ax.errorbar(qs, Is, yerr=sigs, fmt=".", color=p["accent"],
                            ecolor=p["border"], ms=3, lw=0.8, capsize=0)
            else:
                ax.plot(qs, Is, ".", color=p["accent"], ms=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # pin x-limits to the data range so the axes edges coincide with the slider's
            # vmin/vmax — otherwise matplotlib's default margins inset the data and the
            # axvlines no longer line up with the handles directly below them
            ax.set_xlim(vmin, vmax)
        else:
            ax.text(0.5, 0.5, "Could not read data file",
                    transform=ax.transAxes, ha="center", va="center",
                    color=p["muted"])

        # the slider below stands in for the x-axis, so strip the axes' own one and let
        # the axvlines run to the bottom edge where the slider's risers pick them up.
        # (done after set_xscale, which would otherwise re-enable the bottom ticks)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)

        self._vlines[0] = ax.axvline(vmin, color=p["accent"], lw=1.5, ls="--", alpha=0.7)
        self._vlines[1] = ax.axvline(vmax, color=p["accent"], lw=1.5, ls="--", alpha=0.7)

        self._fig.set_layout_engine("tight")
        self._mpl_canvas.draw_idle()

    def _on_range_changed(self, qmin: float, qmax: float):
        if self._vlines[0] is not None:
            self._vlines[0].set_xdata([qmin, qmin])
            self._vlines[1].set_xdata([qmax, qmax])
            self._mpl_canvas.draw_idle()


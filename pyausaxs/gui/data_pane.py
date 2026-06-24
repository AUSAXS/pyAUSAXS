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
    plot with two axvlines driven by a RangeSlider whose track is aligned to
    the plot's x-axis. Call qrange() to read the selected (qmin, qmax)."""

    def __init__(self, parent, file_path: str):
        super().__init__(parent)
        self.file_path = file_path
        self.title = os.path.splitext(os.path.basename(file_path))[0]
        self._vlines = [None, None]

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
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # --- slider ---
        slider_frame = ttk.Frame(self, padding=(6, 2, 6, 6))
        slider_frame.pack(fill="x")
        self.q_slider = RangeSlider(
            slider_frame, vmin, vmax,
            log=True, fmt="{:.4g}",
            on_change=self._on_range_changed,
        )
        self.q_slider.pack(fill="x")

        self._draw_data(data, vmin, vmax)

        # align slider track with axes margins after every draw/resize
        self._fig.canvas.mpl_connect("draw_event", lambda _e: self._align_slider())
        self._mpl_canvas.get_tk_widget().bind("<Configure>",
                                              lambda _e: self._mpl_canvas.draw_idle())

    # ------------------------------------------------------------------
    def qrange(self) -> tuple[float, float]:
        return self.q_slider.values()

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
        ax.set_xlabel("q [1/Å]", color=p["muted"], fontsize=9)
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
        else:
            ax.text(0.5, 0.5, "Could not read data file",
                    transform=ax.transAxes, ha="center", va="center",
                    color=p["muted"])

        self._vlines[0] = ax.axvline(vmin, color=p["accent"], lw=1.5, ls="--", alpha=0.7)
        self._vlines[1] = ax.axvline(vmax, color=p["accent"], lw=1.5, ls="--", alpha=0.7)

        self._fig.set_layout_engine("tight")
        self._mpl_canvas.draw_idle()

    def _on_range_changed(self, qmin: float, qmax: float):
        if self._vlines[0] is not None:
            self._vlines[0].set_xdata([qmin, qmin])
            self._vlines[1].set_xdata([qmax, qmax])
            self._mpl_canvas.draw_idle()

    def _align_slider(self):
        """Match the slider track margins to the matplotlib axes left/right margins."""
        try:
            bbox = self._ax.get_position()
            w = self._mpl_canvas.get_tk_widget().winfo_width()
            self.q_slider.set_track_pads(int(bbox.x0 * w), int((1.0 - bbox.x1) * w))
        except Exception:
            pass

# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import math
import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .panes import QMIN, QMAX, _read_saxs_data
from .theme import PALETTE


class SaxsDataPane(ttk.Frame):
    """Interactive data-inspection pane for a single SAXS dataset."""

    # factor that converts a raw q value in the selected unit to Å⁻¹
    UNIT_FACTORS = {"Å⁻¹": 1.0, "nm⁻¹": 0.1}
    DEFAULT_UNIT = "Å⁻¹"

    DATA_RATIO = 8      # data-axes : slider-strip height ratio
    Y_TRACK = 0.64      # handle row, as a fraction of the slider strip's height
    Y_LABEL = 0.48      # decade-label row, just beneath the handles
    HANDLE_MS = 11      # handle marker diameter (points)
    LABEL_SIZE = 10     # decade-label font size (points)
    UNIT_ANIM_MS = 900  # how long both interpretations are shown on a unit change

    def __init__(self, parent, file_path: str):
        super().__init__(parent)
        self.file_path = file_path
        self.title = "Dataset: " + os.path.splitext(os.path.basename(file_path))[0]
        self._qs_raw = None                       # q as read from file (selected unit)
        self._qs = self._Is = self._sigs = None   # q converted to Å⁻¹
        self._has_sigma = False
        self._data_artists: list = []
        self._unit_var = tk.StringVar(value=self.DEFAULT_UNIT)
        self._unit_factor = self.UNIT_FACTORS[self.DEFAULT_UNIT]
        self._drag = None                         # handle being dragged (0 / 1 / None)
        self._vline_top = None                    # cached slider-fraction of the plot top
        self._decade_labels: list = []
        self._ghost_artists: list = []            # previous interpretation during a unit change
        self._unit_anim_job = None                # pending end-of-transition callback

        # the axis (and slider track) span the data's own q-range, in Å⁻¹; the unit
        # selector reinterprets the file's q and the range rescales with it
        data = _read_saxs_data(file_path)
        if data:
            qs, _, _ = data
            self._vmin = min(qs) * self._unit_factor
            self._vmax = max(qs) * self._unit_factor
        else:
            self._vmin, self._vmax = QMIN, QMAX
        self._qmin, self._qmax = self._vmin, self._vmax

        # --- figure: data axes on top, a thin slider strip sharing its x below ---
        self._fig = Figure(facecolor=PALETTE["surface"])
        self._ax, self._sax = self._fig.subplots(
            2, 1, sharex=True,
            gridspec_kw=dict(height_ratios=[self.DATA_RATIO, 1], hspace=0.0))
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self)

        # --- editable readouts + unit selector, in a tk row beneath the plot ---
        ctrl = ttk.Frame(self, padding=(6, 0, 6, 4))
        ctrl.pack(side="bottom", fill="x")
        self.lo_var, self.hi_var = tk.StringVar(), tk.StringVar()
        lo_entry = ttk.Entry(ctrl, textvariable=self.lo_var, width=9, justify="center")
        hi_entry = ttk.Entry(ctrl, textvariable=self.hi_var, width=9, justify="center")
        center = ttk.Frame(ctrl)
        lo_entry.grid(row=0, column=0, sticky="w")
        center.grid(row=0, column=1)
        hi_entry.grid(row=0, column=2, sticky="e")
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(2, weight=1)
        # the plot is always Å⁻¹; picking nm⁻¹ declares the file's q to be in nm⁻¹, so the
        # data is converted (×0.1) to keep the displayed axis in Å⁻¹.
        ttk.Label(center, text="q [Å⁻¹]", style="Muted.TLabel").pack()
        unit_box = ttk.Combobox(center, textvariable=self._unit_var,
                                values=list(self.UNIT_FACTORS), state="readonly", width=5)
        unit_box.pack()
        unit_box.bind("<<ComboboxSelected>>", lambda _e: self._on_unit_changed())
        for entry in (lo_entry, hi_entry):
            entry.bind("<Return>", lambda _e: self._on_entry())
            entry.bind("<FocusOut>", lambda _e: self._on_entry())

        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(6, 0))

        self._draw_data(data)
        self._draw_selector()
        self._sync_entries()

        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        # the dashed risers run from the strip up to the plot top; that span (in strip
        # fractions) shifts when tight-layout re-places the axes, so refresh it per draw
        self._fig.canvas.mpl_connect("draw_event", lambda _e: self._sync_vline_top())

        self._fig.set_layout_engine("tight")
        self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    def qrange(self) -> tuple[float, float]:
        return self._qmin, self._qmax

    # ------------------------------------------------------------------
    def _draw_data(self, data):
        ax, p = self._ax, PALETTE
        ax.clear()
        self._data_artists = []
        ax.set_facecolor(p["surface"])
        ax.tick_params(colors=p["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(p["border"])
        ax.spines["bottom"].set_visible(False)  # blend into the slider strip below
        ax.set_ylabel("I(q)", color=p["muted"], fontsize=9)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)  # the strip is the x-axis

        if data:
            qs, Is, sigs = data
            self._qs_raw = qs
            self._qs = [q * self._unit_factor for q in qs]
            self._Is, self._sigs = Is, sigs
            self._has_sigma = any(s > 0 for s in sigs)
            ax.set_xscale("log")
            ax.set_yscale("log")
            self._redraw_data_artists()
            # y-limits from the data; x is the data's own q-range
            ax.relim()
            ax.autoscale_view(scalex=False)
            ax.set_xlim(self._vmin, self._vmax)
            ax.autoscale(enable=False)
        else:
            self._qs_raw = self._qs = self._Is = self._sigs = None
            ax.text(0.5, 0.5, "Could not read data file", transform=ax.transAxes,
                    ha="center", va="center", color=p["muted"])

        self._style_strip()

    def _style_strip(self):
        """The slider strip hosts the track/handles/risers and the decade labels (drawn
        as text in _draw_selector); its own axis furniture is hidden."""
        sax, p = self._sax, PALETTE
        sax.set_ylim(0, 1)
        sax.set_yticks([])
        sax.set_facecolor(p["surface"])
        for spine in sax.spines.values():
            spine.set_visible(False)
        sax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # ------------------------------------------------------------------
    def _redraw_data_artists(self):
        """Re-add the data split into included (blue) and excluded (muted red)."""
        for a in self._data_artists:
            a.remove()
        self._data_artists = []
        if self._qs is None:
            return
        ax, p = self._ax, PALETTE
        qs, Is, sigs = self._qs, self._Is, self._sigs
        qmin, qmax = self._qmin, self._qmax

        def _pick(seq, mask):
            return [v for v, m in zip(seq, mask) if m]

        def _add(qsel, Isel, ssel, color, ecolor):
            if not qsel:
                return
            if self._has_sigma:
                artist = ax.errorbar(qsel, Isel, yerr=ssel, fmt=".", color=color,
                                     ecolor=ecolor, ms=3, lw=0.8, capsize=0)
            else:
                artist, = ax.plot(qsel, Isel, ".", color=color, ms=3)
            self._data_artists.append(artist)

        ex = [not (qmin <= q <= qmax) for q in qs]
        inc = [not e for e in ex]
        _add(_pick(qs, ex), _pick(Is, ex), _pick(sigs, ex), p["bad_border"], p["bad_border"])
        _add(_pick(qs, inc), _pick(Is, inc), _pick(sigs, inc), p["accent"], p["border"])

    # ------------------------------------------------------------------
    def _draw_selector(self):
        """Create the track, selected span, handles and dashed risers (drawn once;
        later moves update their data in place). """
        sax, p = self._sax, PALETTE
        yt = self.Y_TRACK
        tr = sax.get_xaxis_transform()

        self._track_line = Line2D([self._vmin, self._vmax], [yt, yt], transform=tr,
                                  color=p["track"], lw=4, solid_capstyle="round",
                                  clip_on=False, zorder=1)
        self._span_line = Line2D([self._qmin, self._qmax], [yt, yt], transform=tr,
                                 color=p["accent"], lw=4, solid_capstyle="round",
                                 clip_on=False, zorder=2)
        sax.add_line(self._track_line)
        sax.add_line(self._span_line)

        self._vlines = []
        for q in (self._qmin, self._qmax):
            vl = Line2D([q, q], [yt, 1.0], transform=tr, color=p["danger"],
                        lw=1.5, ls="--", alpha=0.8, clip_on=False, zorder=3)
            sax.add_line(vl)
            self._vlines.append(vl)

        self._handles = Line2D([self._qmin, self._qmax], [yt, yt], transform=tr,
                               ls="none", marker="o", ms=self.HANDLE_MS, mfc=p["accent"],
                               mec=p["surface"], mew=1.5, clip_on=False, zorder=5)
        self._handle_cores = Line2D([self._qmin, self._qmax], [yt, yt], transform=tr,
                                    ls="none", marker="o", ms=3, mfc=p["surface"],
                                    mec="none", clip_on=False, zorder=6)
        sax.add_line(self._handles)
        sax.add_line(self._handle_cores)
        self._draw_decade_labels()

    def _draw_decade_labels(self):
        """(Re)draw the decade labels just beneath the handles; redrawn on a unit change
        since the q-range — and thus which decades fall in it — rescales."""
        for t in self._decade_labels:
            t.remove()
        self._decade_labels = []
        tr = self._sax.get_xaxis_transform()
        d0, d1 = math.ceil(math.log10(self._vmin)), math.floor(math.log10(self._vmax))
        for d in range(d0, d1 + 1):
            self._decade_labels.append(self._sax.text(
                10.0 ** d, self.Y_LABEL, rf"$10^{{{d}}}$", transform=tr,
                ha="center", va="top", color=PALETTE["muted"], fontsize=self.LABEL_SIZE,
                clip_on=False))

    def _update_selector(self):
        self._span_line.set_xdata([self._qmin, self._qmax])
        self._handles.set_xdata([self._qmin, self._qmax])
        self._handle_cores.set_xdata([self._qmin, self._qmax])
        self._vlines[0].set_xdata([self._qmin, self._qmin])
        self._vlines[1].set_xdata([self._qmax, self._qmax])

    def _sync_vline_top(self):
        """Match each riser's top to the plot's top edge, expressed in strip fractions
        (>1, since the strip is short). Recomputed per draw so it tracks tight-layout."""
        ax_pos, sax_pos = self._ax.get_position(), self._sax.get_position()
        if sax_pos.height <= 0:
            return
        y_top = (ax_pos.y1 - sax_pos.y0) / sax_pos.height
        if self._vline_top is None or abs(y_top - self._vline_top) > 1e-3:
            self._vline_top = y_top
            for vl in self._vlines:
                vl.set_ydata([self.Y_TRACK, y_top])
            self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_press(self, event):
        if event.inaxes is not self._sax or event.xdata is None:
            return
        self._drag = self._nearest_handle(event.xdata)
        self._drag_to(event.xdata)

    def _on_motion(self, event):
        # x is shared, so dragging stays valid even if the cursor strays up into the plot
        if self._drag is None or event.xdata is None:
            return
        self._drag_to(event.xdata)

    def _on_release(self, _event):
        self._drag = None

    def _nearest_handle(self, q: float) -> int:
        lq = math.log10(max(q, 1e-30))
        return 0 if abs(lq - math.log10(self._qmin)) <= abs(lq - math.log10(self._qmax)) else 1

    def _drag_to(self, q: float):
        q = min(max(q, self._vmin), self._vmax)
        if self._drag == 0:
            self._qmin = min(q, self._qmax)
        else:
            self._qmax = max(q, self._qmin)
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        self._sync_entries()
        self._update_selector()
        self._redraw_data_artists()
        self._mpl_canvas.draw_idle()

    def _sync_entries(self):
        self.lo_var.set(f"{self._qmin:.4g}")
        self.hi_var.set(f"{self._qmax:.4g}")

    def _on_entry(self):
        try:
            lo, hi = float(self.lo_var.get()), float(self.hi_var.get())
        except ValueError:
            self._sync_entries()
            return
        lo = min(max(lo, self._vmin), self._vmax)
        hi = min(max(hi, self._vmin), self._vmax)
        self._qmin, self._qmax = min(lo, hi), max(lo, hi)
        self._refresh()

    def _on_unit_changed(self):
        """Reinterpret the file's q in the selected unit and convert to Å⁻¹. The data's
        q-range — and so the axis, slider track and decade labels — rescales with it,
        while the selection keeps its position relative to the data. A short transition
        keeps the previous interpretation on screen so the shift is visible."""
        if self._qs_raw is None:
            return
        new_factor = self.UNIT_FACTORS[self._unit_var.get()]
        if new_factor == self._unit_factor:
            return
        old_qs, old_lim = self._qs, (self._vmin, self._vmax)
        ratio = new_factor / self._unit_factor
        self._unit_factor = new_factor
        self._qs = [q * new_factor for q in self._qs_raw]
        self._vmin, self._vmax = min(self._qs), max(self._qs)
        self._qmin *= ratio
        self._qmax *= ratio

        # ghost the previous interpretation at its old positions, and widen the axis to
        # span both ranges so the data is seen sliding across before the view settles
        self._clear_ghosts()
        ghost, = self._ax.plot(old_qs, self._Is, ".", color=PALETTE["muted"], ms=3, alpha=0.35, zorder=0)
        self._ghost_artists = [ghost]
        self._track_line.set_xdata([self._vmin, self._vmax])
        self._draw_decade_labels()
        self._refresh()
        self._ax.set_xlim(min(self._vmin, old_lim[0]), max(self._vmax, old_lim[1]))
        self._mpl_canvas.draw_idle()

        if self._unit_anim_job is not None:
            self.after_cancel(self._unit_anim_job)
        self._unit_anim_job = self.after(self.UNIT_ANIM_MS, self._settle_unit_change)

    def _settle_unit_change(self):
        """End the unit-change transition: drop the ghost and zoom onto the new range."""
        self._unit_anim_job = None
        self._clear_ghosts()
        self._ax.set_xlim(self._vmin, self._vmax)
        self._mpl_canvas.draw_idle()

    def _clear_ghosts(self):
        for a in self._ghost_artists:
            a.remove()
        self._ghost_artists = []

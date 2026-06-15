# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import math
import re
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Optional

from .theme import FONTS, PALETTE


class FileField(ttk.Frame):
    """A labelled file path entry with a browse button and background validation coloring.

    The validator is called with the entered path and must return True/False.
    on_valid fires with the path whenever it passes validation (including on focus-out,
    so it is suited to passive reactions like coloring or autodetection). on_commit fires
    only on explicit user commits — pressing Enter or picking a file via Browse — and is
    suited to actions that should not be re-triggered by merely tabbing away.
    """

    def __init__(
        self, parent,
        label: str,
        validator: Callable[[str], bool],
        on_valid: Optional[Callable[[str], None]] = None,
        on_commit: Optional[Callable[[str], None]] = None,
        filetypes: Optional[list[tuple[str, str]]] = None,
        directory: bool = False,
    ):
        super().__init__(parent)
        self._validator = validator
        self._on_valid = on_valid
        self._on_commit = on_commit
        self._filetypes = filetypes or []
        self._directory = directory
        self.valid = False
        self.touched = False  # whether the user has manually edited the field

        ttk.Label(self, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.var = tk.StringVar()
        self.entry = tk.Entry(
            self, textvariable=self.var,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            disabledbackground=PALETTE["surface"], readonlybackground=PALETTE["surface"],
            insertbackground=PALETTE["text"], font=FONTS["base"],
            relief="flat", highlightthickness=1,
            highlightbackground=PALETTE["border"], highlightcolor=PALETTE["accent"],
        )
        self.entry.grid(row=1, column=0, sticky="ew", ipady=4)
        ttk.Button(self, text="Browse", style="Icon.TButton", command=self._browse).grid(
            row=1, column=1, padx=(6, 0))
        self.columnconfigure(0, weight=1)

        self.entry.bind("<Return>", lambda _e: self._commit())
        self.entry.bind("<FocusOut>", lambda _e: self.validate())
        self.entry.bind("<Key>", self._on_keypress)

    def _set_state_color(self, fill: str, border: str):
        self.entry.configure(background=fill, readonlybackground=fill,
                             disabledbackground=fill, highlightbackground=border)

    def _on_keypress(self, event):
        if event.keysym in ("Return", "Tab"):
            return
        self.touched = True
        self.valid = False
        self._set_state_color(PALETTE["surface"], PALETTE["border"])

    def _browse(self):
        if self._directory:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=self._filetypes + [("All files", "*")])
        if path:
            self.set(path, touched=True)
            self._commit()

    def _commit(self):
        """Validate and, if valid, fire on_commit. Triggered by Enter and Browse only."""
        if self.validate() and self._on_commit:
            self._on_commit(self.get())

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, path: str, touched: bool = False):
        self.var.set(path)
        self.touched = self.touched or touched
        self.validate()

    def validate(self) -> bool:
        path = self.get()
        if not path:
            self.valid = False
            self._set_state_color(PALETTE["surface"], PALETTE["border"])
            return False
        self.valid = bool(self._validator(path))
        if self.valid:
            self._set_state_color(PALETTE["ok"], PALETTE["ok_border"])
        else:
            self._set_state_color(PALETTE["bad"], PALETTE["bad_border"])
        if self.valid and self._on_valid:
            self._on_valid(path)
        return self.valid


class RangeSlider(ttk.Frame):
    """A double-ended slider with linked min/max entry boxes.

    Supports linear and logarithmic scales. Values are reported through
    on_change(vmin, vmax) and can be read via .values().
    """

    HANDLE_R = 8          # handle radius
    TRACK_H = 5           # track thickness
    PAD = 14
    CANVAS_H = 42

    def __init__(
        self, parent,
        vmin: float, vmax: float,
        start: Optional[tuple[float, float]] = None,
        log: bool = False,
        fmt: str = "{:.4g}",
        on_change: Optional[Callable[[float, float], None]] = None,
    ):
        super().__init__(parent)
        self._min, self._max = vmin, vmax
        self._log = log
        self._fmt = fmt
        self._on_change = on_change
        lo, hi = start if start else (vmin, vmax)
        self._pos = [self._to_pos(lo), self._to_pos(hi)]
        self._drag = None
        self._hover = None

        self.canvas = tk.Canvas(
            self, height=self.CANVAS_H, highlightthickness=0,
            background=PALETTE["bg"], bd=0)
        self.canvas.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(2, 2))

        self.lo_var, self.hi_var = tk.StringVar(), tk.StringVar()
        lo_entry = ttk.Entry(self, textvariable=self.lo_var, width=9, justify="center")
        hi_entry = ttk.Entry(self, textvariable=self.hi_var, width=9, justify="center")
        lo_entry.grid(row=1, column=0, sticky="w")
        hi_entry.grid(row=1, column=1, sticky="e")
        lo_entry.bind("<Return>", lambda _e: self._entry_changed())
        lo_entry.bind("<FocusOut>", lambda _e: self._entry_changed())
        hi_entry.bind("<Return>", lambda _e: self._entry_changed())
        hi_entry.bind("<FocusOut>", lambda _e: self._entry_changed())

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", self._hover_motion)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))
        self._sync_entries()

    # value <-> normalized [0, 1] position
    def _to_pos(self, value: float) -> float:
        value = min(max(value, self._min), self._max)
        if self._log:
            return (math.log10(value) - math.log10(self._min)) / (math.log10(self._max) - math.log10(self._min))
        return (value - self._min) / (self._max - self._min)

    def _to_value(self, pos: float) -> float:
        pos = min(max(pos, 0.0), 1.0)
        if self._log:
            return 10 ** (pos*(math.log10(self._max) - math.log10(self._min)) + math.log10(self._min))
        return self._min + pos*(self._max - self._min)

    def values(self) -> tuple[float, float]:
        return self._to_value(self._pos[0]), self._to_value(self._pos[1])

    def set_values(self, lo: float, hi: float):
        self._pos = sorted((self._to_pos(lo), self._to_pos(hi)))
        self._sync_entries()
        self._redraw()
        if self._on_change:
            self._on_change(*self.values())

    def _track_x(self, pos: float) -> float:
        w = self.canvas.winfo_width()
        return self.PAD + pos*(w - 2*self.PAD)

    def _x_to_pos(self, x: float) -> float:
        w = self.canvas.winfo_width()
        return (x - self.PAD) / max(w - 2*self.PAD, 1)

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        cy = self.CANVAS_H // 2 - 4
        r = self.HANDLE_R
        ht = self.TRACK_H

        # background track with rounded caps
        c.create_line(self.PAD, cy, w - self.PAD, cy,
                      width=ht, fill=PALETTE["track"], capstyle="round")
        # selected span
        x0, x1 = self._track_x(self._pos[0]), self._track_x(self._pos[1])
        c.create_line(x0, cy, x1, cy, width=ht, fill=PALETTE["accent"], capstyle="round")

        # tick marks: decades for log scale, quarters for linear
        if self._log:
            d0, d1 = math.ceil(math.log10(self._min)), math.floor(math.log10(self._max))
            ticks = [10.0**d for d in range(d0, d1 + 1)]
        else:
            ticks = [self._min + i*(self._max - self._min)/4 for i in range(5)]
        for t in ticks:
            x = self._track_x(self._to_pos(t))
            c.create_text(x, cy + r + 7, text=self._fmt.format(t),
                          font=FONTS["small"], fill=PALETTE["muted"])

        # circular handles, accent-filled with a white core and a hover ring
        for i, pos in enumerate(self._pos):
            x = self._track_x(pos)
            if self._hover == i or self._drag == i:
                c.create_oval(x - r - 3, cy - r - 3, x + r + 3, cy + r + 3,
                              fill=PALETTE["accent_soft"], outline="")
            c.create_oval(x - r, cy - r, x + r, cy + r,
                          fill=PALETTE["accent"], outline=PALETTE["surface"], width=2)
            c.create_oval(x - 2, cy - 2, x + 2, cy + 2, fill=PALETTE["surface"], outline="")

    def _nearest_handle(self, x: float) -> int:
        d0 = abs(self._track_x(self._pos[0]) - x)
        d1 = abs(self._track_x(self._pos[1]) - x)
        return 0 if d0 <= d1 else 1

    def _set_hover(self, value):
        if value != self._hover:
            self._hover = value
            self._redraw()

    def _hover_motion(self, event):
        if self._drag is not None:
            return
        x = self._track_x(self._pos[self._nearest_handle(event.x)])
        self._set_hover(self._nearest_handle(event.x) if abs(x - event.x) <= self.HANDLE_R + 6 else None)

    def _press(self, event):
        self._drag = self._nearest_handle(event.x)
        self._motion(event)

    def _motion(self, event):
        if self._drag is None:
            return
        pos = min(max(self._x_to_pos(event.x), 0.0), 1.0)
        if self._drag == 0:
            self._pos[0] = min(pos, self._pos[1])
        else:
            self._pos[1] = max(pos, self._pos[0])
        self._sync_entries()
        self._redraw()
        if self._on_change:
            self._on_change(*self.values())

    def _release(self, _event):
        self._drag = None

    def _sync_entries(self):
        lo, hi = self.values()
        self.lo_var.set(self._fmt.format(lo))
        self.hi_var.set(self._fmt.format(hi))

    def _entry_changed(self):
        try:
            lo = float(self.lo_var.get())
            hi = float(self.hi_var.get())
        except ValueError:
            self._sync_entries()
            return
        self.set_values(lo, hi)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class ConsolePane(ttk.Frame):
    """A read-only scrolling text pane for subprocess output."""

    def __init__(self, parent, height: int = 8):
        super().__init__(parent)
        self.text = tk.Text(
            self, height=height, wrap="word", state="disabled",
            background=PALETTE["console_bg"], foreground=PALETTE["console_fg"],
            insertbackground=PALETTE["console_fg"], selectbackground=PALETTE["accent"],
            font=FONTS["mono"], relief="flat", borderwidth=0,
            padx=10, pady=8,
        )
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def append(self, line: str):
        line = _ANSI_RE.sub("", line)
        self.text.configure(state="normal")
        self.text.insert("end", line)
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

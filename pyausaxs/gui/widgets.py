# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import math
import re
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Optional

from tkinterdnd2 import DND_FILES

from .theme import ANSI_COLORS, FONTS, PALETTE, SYNTAX


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
        ttk.Button(self, text="Browse", style="Icon.TButton", command=self._browse).grid(row=1, column=1, padx=(6, 0))
        self.columnconfigure(0, weight=1)

        self.entry.bind("<Return>", lambda _e: self._commit())
        self.entry.bind("<FocusOut>", lambda _e: self.validate())
        self.entry.bind("<Key>", self._on_keypress)

    def _set_state_color(self, fill: str, border: str):
        self.entry.configure(background=fill, readonlybackground=fill, disabledbackground=fill, highlightbackground=border)

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

    def accept_drop(self, path: str) -> bool:
        """Accept a drag-and-dropped path if it passes this field's validator, mirroring the
        Browse flow (set + commit). Returns whether the path was accepted."""
        if not self._validator(path):
            return False
        self.set(path, touched=True)
        self._commit()
        return True


# one dropped-file token: either a brace-quoted path (used by tkdnd when the path contains
# spaces) or a bare run of non-space characters
_DND_TOKEN_RE = re.compile(r"\{[^}]*\}|\S+")


def _split_dropped_paths(data: str) -> list[str]:
    """Split a tkinterdnd2 <<Drop>> event's raw `data` string into individual file paths. Parsed by hand rather than via Tcl's splitlist, 
    which would mis-parse backslashes in Windows paths as escape sequences."""
    return [token.strip("{}") for token in _DND_TOKEN_RE.findall(data)]


class _DropOverlay(tk.Frame):
    """A hint shown over its parent while a dragged file hovers over it, cueing where it will land."""

    # A boundary crossing at the very edge of the parent can still produce a DropLeave that
    # isn't a genuine departure. hide() is delayed by this long so a following DropEnter can
    # cancel it before it takes effect; only a DropLeave with nothing following within the
    # delay actually hides the overlay.
    _HIDE_DELAY_MS = 400

    def __init__(self, parent):
        super().__init__(parent, background=PALETTE["accent_soft"],
                          highlightthickness=3, highlightbackground=PALETTE["accent"])
        tk.Label(
            self, text="⤓", background=PALETTE["accent_soft"], foreground=PALETTE["accent"],
            font=(FONTS["base"][0], 42),
        ).place(relx=0.5, rely=0.46, anchor="center")
        tk.Label(
            self, text="Drop file to load", background=PALETTE["accent_soft"], foreground=PALETTE["accent"],
            font=FONTS["heading"],
        ).place(relx=0.5, rely=0.56, anchor="center")
        self._hide_job = None

    def show(self):
        if self._hide_job is not None:
            self.after_cancel(self._hide_job)
            self._hide_job = None
        if not self.winfo_ismapped():
            self.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.lift()

    def hide(self, immediate: bool = False):
        if self._hide_job is not None:
            self.after_cancel(self._hide_job)
            self._hide_job = None
        if immediate:
            self.place_forget()
        else:
            self._hide_job = self.after(self._HIDE_DELAY_MS, self._hide_now)

    def _hide_now(self):
        self._hide_job = None
        self.place_forget()


# Set once any <<Drop>> succeeds anywhere in the process. tkdnd has a known quirk (confirmed
# via a live diagnostic, unrelated to this app's code) where the very first cross-process drag
# of a session can silently resolve to a DropLeave with no file data instead of a Drop; every
# attempt after that first one works normally. Tracked process-wide, not per-pane, since the
# quirk is in the shared underlying XDND machinery, not any one widget.
_drop_ever_succeeded = False


def enable_file_drop(
    widget,
    fields: list[FileField],
    on_unmatched: Optional[Callable[[str], None]] = None,
    on_leave_without_drop: Optional[Callable[[], None]] = None,
):
    """Register `widget` as a drag-and-drop target for files.

    Each dropped path is routed to the first of `fields` whose validator accepts it, so a file lands in the right FileField no matter where
    over `widget` it was actually dropped (e.g. a SAXS file dropped on the structure field still ends up in the SAXS field). Paths that no
    field accepts are reported to `on_unmatched`, if given, and otherwise ignored.

    While a file is dragged over `widget`, a hint overlay (see `_DropOverlay`) — a child of `widget`, so it can never disrupt `widget`'s own
    drop-target resolution — is shown, and hidden again on drop or once the drag leaves.

    `on_leave_without_drop`, if given, fires whenever a drag leaves without any Drop having
    ever succeeded yet in this process (see `_drop_ever_succeeded`) — a cheap nudge to retry
    for the known first-drop-of-the-session quirk described above.
    """
    overlay = _DropOverlay(widget)

    def handle_enter(_event):
        overlay.show()
        return "copy"

    def handle_leave(_event):
        overlay.hide()
        if not _drop_ever_succeeded and on_leave_without_drop:
            on_leave_without_drop()

    def handle_drop(event):
        global _drop_ever_succeeded
        _drop_ever_succeeded = True
        overlay.hide(immediate=True)
        for path in _split_dropped_paths(event.data):
            if not any(field.accept_drop(path) for field in fields) and on_unmatched:
                on_unmatched(path)
        return "copy"

    widget.drop_target_register(DND_FILES)
    widget.dnd_bind("<<DropEnter>>", handle_enter)
    widget.dnd_bind("<<DropLeave>>", handle_leave)
    widget.dnd_bind("<<Drop>>", handle_drop)


class Tooltip:
    """A small hover tooltip for any widget (Tk has none of its own). Bindings are added with
    add="+" so they don't clobber the widget's own <Enter>/<Leave> handlers."""

    def __init__(self, widget, text: str, delay: int = 450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self.text, justify="left",
            background=PALETTE["text"], foreground="#ffffff",
            font=FONTS["small"], padx=8, pady=4, borderwidth=0,
        ).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class RangeSlider(ttk.Frame):
    """A double-ended slider with linked min/max entry boxes.

    Supports linear and logarithmic scales. Values are reported through
    on_change(vmin, vmax) and can be read via .values().
    """

    HANDLE_R = 8          # handle radius
    TRACK_H = 5           # track thickness
    PAD = 14
    CANVAS_H = 42

    def set_track_pads(self, left: int, right: int):
        """Override the symmetric PAD with per-side values, e.g. to align with a plot's x-axis."""
        self._left_pad = max(left, self.HANDLE_R + 2)
        self._right_pad = max(right, self.HANDLE_R + 2)
        self._redraw()

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
        self._left_pad = self.PAD
        self._right_pad = self.PAD

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
        return self._left_pad + pos * (w - self._left_pad - self._right_pad)

    def _x_to_pos(self, x: float) -> float:
        w = self.canvas.winfo_width()
        return (x - self._left_pad) / max(w - self._left_pad - self._right_pad, 1)

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        cy = self.CANVAS_H // 2 - 4
        r = self.HANDLE_R
        ht = self.TRACK_H

        # background track with rounded caps
        c.create_line(self._left_pad, cy, w - self._right_pad, cy, width=ht, fill=PALETTE["track"], capstyle="round")
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
            c.create_text(x, cy + r + 7, text=self._fmt.format(t), font=FONTS["small"], fill=PALETTE["muted"])

        # circular handles, accent-filled with a white core and a hover ring
        for i, pos in enumerate(self._pos):
            x = self._track_x(pos)
            if self._hover == i or self._drag == i:
                c.create_oval(x - r - 3, cy - r - 3, x + r + 3, cy + r + 3, fill=PALETTE["accent_soft"], outline="")
            c.create_oval(x - r, cy - r, x + r, cy + r, fill=PALETTE["accent"], outline=PALETTE["surface"], width=2)
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


# one ANSI escape sequence: group 1 = parameters, group 2 = final byte (e.g. 'm' for SGR)
_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


class ConsolePane(ttk.Frame):
    """A read-only scrolling text pane for subprocess output.

    Renders the backend's ANSI SGR colour codes by mapping them to per-colour text
    tags; non-colour escape sequences are dropped. Colour state does not carry across
    append() calls, matching the line-buffered output the backend emits.
    """

    def __init__(self, parent, height: int = 8):
        super().__init__(parent)
        self.text = tk.Text(
            self, height=height, wrap="word", state="disabled",
            background=PALETTE["console_bg"], foreground=PALETTE["console_fg"],
            insertbackground=PALETTE["console_fg"], selectbackground=PALETTE["accent"],
            font=FONTS["mono"], relief="flat", borderwidth=0,
            padx=10, pady=8,
        )
        for code, color in ANSI_COLORS.items():
            self.text.tag_configure(f"ansi{code}", foreground=color)
        # tags for GUI-side status lines (Python errors / success notes), which carry
        # no ANSI codes of their own; reuse the ANSI red/green so they match backend output
        mono = FONTS["mono"]
        self.text.tag_configure("error", foreground=ANSI_COLORS[31], font=(mono[0], mono[1], "bold"))
        self.text.tag_configure("success", foreground=ANSI_COLORS[32])
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def append(self, line: str, tag: str | None = None):
        """Append text, rendering any ANSI colour codes. `tag` colours text that no ANSI
        code applies to (e.g. "error" for a Python-side message); ANSI codes still win."""
        self.text.configure(state="normal")
        pos = 0
        active = None
        for match in _ANSI_RE.finditer(line):
            if match.start() > pos:
                self.text.insert("end", line[pos:match.start()], active or tag or ())
            if match.group(2) == "m":  # SGR colour/reset; other sequences are dropped
                active = self._apply_sgr(active, match.group(1))
            pos = match.end()
        if pos < len(line):
            self.text.insert("end", line[pos:], active or tag or ())
        self.text.see("end")
        self.text.configure(state="disabled")

    @staticmethod
    def _apply_sgr(tag, params: str):
        """Fold an SGR parameter list into the active colour tag (None = default)."""
        for code in params.split(";"):
            if code in ("", "0"):
                tag = None
            elif code.isdigit() and int(code) in ANSI_COLORS:
                tag = f"ansi{code}"
        return tag

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class RigidbodyHighlighter:
    """Syntax highlighting for the rigid-body sequencer script in a tk.Text editor.

    Colours the first token of each line — scope keywords by nesting depth, line
    operations, argument keywords, and unrecognised tokens flagged as errors — plus
    trailing # comments, and highlights the matching scope opener / `end` pair around
    the cursor. A tkinter port of the Qt RigidBodyHighlighter; since tkinter has no
    per-line parser state, depth is recomputed by scanning the whole (short) script.
    """

    SCOPE_KEYWORDS = ("loop", "optimize_once", "on_improvement")
    _FIRST_TOKEN_RE = re.compile(r"\S+")

    def __init__(self, editor: tk.Text, operations=None, keywords=None):
        self.editor = editor
        self.operations = set(operations or ())
        self.keywords = set(keywords or ())
        self._scope_tags = [f"scope{i}" for i in range(len(SYNTAX["scope"]))]
        self._token_tags = ("op", "keyword", "comment", "error", "error_line", *self._scope_tags)

        mono = FONTS["mono"]
        mono_bold = (mono[0], mono[1], "bold")
        editor.tag_configure("op", foreground=SYNTAX["operation"], font=mono_bold)
        editor.tag_configure("keyword", foreground=SYNTAX["keyword"])
        editor.tag_configure("comment", foreground=SYNTAX["comment"])
        # unrecognised tokens stand out with bold red text on a soft red line tint
        editor.tag_configure("error", foreground=SYNTAX["error"], font=mono_bold)
        editor.tag_configure("error_line", background=SYNTAX["error_bg"])
        for tag, color in zip(self._scope_tags, SYNTAX["scope"]):
            editor.tag_configure(tag, foreground=color, font=mono_bold)
        editor.tag_configure("bracket", background=SYNTAX["bracket_bg"])
        # keep the full-line background tints beneath the foreground token tags
        editor.tag_lower("bracket")
        editor.tag_lower("error_line")

    def set_vocabulary(self, operations, keywords):
        """Replace the known operations/keywords (e.g. once the backend reports them)."""
        self.operations = set(operations or ())
        self.keywords = set(keywords or ())
        self.highlight()

    def _scope_tag(self, depth: int) -> str:
        return self._scope_tags[depth % len(self._scope_tags)]

    def highlight(self):
        ed = self.editor
        for tag in self._token_tags:
            ed.tag_remove(tag, "1.0", "end")

        depth = 0
        lines = ed.get("1.0", "end-1c").split("\n")
        for i, line in enumerate(lines):
            lineno = i + 1
            hash_col = line.find("#")
            if hash_col != -1:  # green comment wins over any keyword inside it
                ed.tag_add("comment", f"{lineno}.{hash_col}", f"{lineno}.end")
                scannable = line[:hash_col]
            else:
                scannable = line

            match = self._FIRST_TOKEN_RE.search(scannable)
            if not match:
                continue
            token = match.group()
            start, end = f"{lineno}.{match.start()}", f"{lineno}.{match.end()}"
            if token == "end":  # decrease first so `end` shares its opener's colour
                depth = max(0, depth - 1)
                ed.tag_add(self._scope_tag(depth), start, end)
            elif token in self.SCOPE_KEYWORDS:
                ed.tag_add(self._scope_tag(depth), start, end)
                depth += 1
            elif token in self.operations:
                ed.tag_add("op", start, end)
            elif token in self.keywords:
                ed.tag_add("keyword", start, end)
            elif token != "}" and (self.operations or self.keywords):
                # only flag unknown tokens when we actually know the valid vocabulary;
                # tint the whole line so the error is visible while scanning the script
                ed.tag_add("error", start, end)
                ed.tag_add("error_line", f"{lineno}.0", f"{lineno + 1}.0")

    def highlight_brackets(self):
        """Tint the scope opener/`end` line pair around the cursor's line."""
        ed = self.editor
        ed.tag_remove("bracket", "1.0", "end")
        lines = ed.get("1.0", "end-1c").split("\n")
        lineno = int(ed.index("insert").split(".")[0])
        token = self._first_token(lines, lineno)
        if token in self.SCOPE_KEYWORDS:
            pair = self._match_forward(lines, lineno)
        elif token == "end":
            pair = self._match_backward(lines, lineno)
        else:
            return
        if pair is None:
            return
        for n in (lineno, pair):
            ed.tag_add("bracket", f"{n}.0", f"{n + 1}.0")

    @staticmethod
    def _first_token(lines, n: int) -> str:
        tokens = lines[n - 1].split() if 1 <= n <= len(lines) else []
        return tokens[0] if tokens else ""

    def _match_forward(self, lines, lineno: int):
        stack = 0
        for n in range(lineno + 1, len(lines) + 1):
            token = self._first_token(lines, n)
            if token in self.SCOPE_KEYWORDS:
                stack += 1
            elif token == "end":
                if stack == 0:
                    return n
                stack -= 1
        return None

    def _match_backward(self, lines, lineno: int):
        stack = 0
        for n in range(lineno - 1, 0, -1):
            token = self._first_token(lines, n)
            if token == "end":
                stack += 1
            elif token in self.SCOPE_KEYWORDS:
                if stack == 0:
                    return n
                stack -= 1
        return None


class ScrollableFrame(ttk.Frame):
    """A vertically scrolling container. Pack/grid children into `.body`; the scrollbar
    appears only when the content overflows. Suited to lists that can grow long (e.g. the
    per-body rows of a badly-split structure). Mouse-wheel scrolling is active while the
    pointer is over the frame."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, background=PALETTE["surface"], highlightthickness=0, bd=0)
        self._scroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scroll_set)
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=e.width))
        # only scroll while the pointer is inside, so wheel events elsewhere are untouched
        self._canvas.bind("<Enter>", lambda _e: self._bind_wheel())
        self._canvas.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _on_scroll_set(self, lo, hi):
        # hide the scrollbar entirely when everything fits
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self._scroll.grid_remove()
        else:
            self._scroll.grid()
        self._scroll.set(lo, hi)

    def _bind_wheel(self):
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)      # Windows / macOS
        self._canvas.bind_all("<Button-4>", self._on_wheel)        # X11 scroll up
        self._canvas.bind_all("<Button-5>", self._on_wheel)        # X11 scroll down

    def _unbind_wheel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        if event.num == 5 or event.delta < 0:
            self._canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self._canvas.yview_scroll(-1, "units")

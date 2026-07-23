# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Structure-inspection / management pane.

Opened from a "View structure" button once a PDB is loaded. It shows a large 3D view of the
structure and lets the user inspect and re-organise its bodies without touching the run:

  * toggle atomic detail (all-atom cloud), symmetry copies, and constraints;
  * see and highlight the individual bodies via a scrolling, backend-fed list;
  * merge or delete bodies, and convert a set of bodies to a symmetry, by adding the
    corresponding setup elements (`merge` / `delete` / `convert_to_symmetry`);
  * preview the resulting script changes as a red/green diff and send them to the editor.

The body list is authoritative: every edit is applied by rebuilding the setup script through the
backend and re-reading the bodies that remain, so a merge or delete really does make bodies
disappear from the list (and the plot) exactly as it will at run time.
"""

import re
import difflib
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .plotting import draw_structure, _BODY_COLORS
from .theme import FONTS, PALETTE
from .widgets import CollapsibleSection, ScrollableFrame

# the load block whose bodies we manage; setup elements are inserted just after it
_LOAD_BLOCK_RE = re.compile(r"load\s*\{.*?\}", re.DOTALL)


def _synth_load_block(pdb_path: str, splits: str) -> str:
    inner = [f"    pdb {pdb_path}"]
    if splits.strip():
        inner.append(f"    split {splits.strip()}")
    return "load {\n" + "\n".join(inner) + "\n}"


def _insert_elements(base: str, elements: list[str]) -> str:
    """Return `base` with the setup elements inserted right after its load block."""
    if not elements:
        return base
    block = "".join(f"{e}\n" for e in elements)
    match = _LOAD_BLOCK_RE.search(base)
    if match is None:  # no load block to anchor to: prepend
        return block + base
    end = match.end()
    sep = "" if base[:end].endswith("\n") else "\n"
    return base[:end] + sep + block + base[end:]


class StructurePane(ttk.Frame):
    """Interactive structure-inspection and body-management pane for a single PDB."""

    def __init__(
        self, parent, pdb_path: str, *,
        splits: str = "",
        base_script: Optional[Callable[[], str]] = None,
        on_apply_script: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        print(f"path is \"{pdb_path}\"")
        self.pdb_path = pdb_path
        self._splits = splits
        self._base_script = base_script          # target script to diff/patch, or None
        self._on_apply_script = on_apply_script   # apply a confirmed new script, or None
        self.title = "Structure: " + pdb_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        self._elements: list[str] = []            # setup elements the user has applied, in order
        self._data: dict | None = None            # last good preview-structure dict
        self._names: list[str] = []               # body names aligned to body indices
        self._bodies: list[dict] = []             # per-body summary rows (index/name/atoms/res)
        self._highlight: int | None = None        # body index isolated in the view, or None
        self._row_frames: list[tk.Widget] = []
        self._rows: list[tuple] = []              # (body index, recolourable row widgets)

        self._show_atoms = tk.BooleanVar(value=False)
        self._show_copies = tk.BooleanVar(value=True)
        self._show_constraints = tk.BooleanVar(value=True)
        self._colour_by_copy = tk.BooleanVar(value=False)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_plot(paned)
        controls = ttk.Frame(paned, padding=(6, 2, 2, 2), width=320)
        controls.pack_propagate(False)
        paned.add(controls, weight=0)
        self._build_controls(controls)

        ok, msg = self._rebuild(self._elements)  # initial draw of the untouched structure
        if not ok:
            self._set_status(f"Could not load the structure: {msg}", ok=False)
            self._redraw()

    # ----- layout -------------------------------------------------------------
    def _build_plot(self, paned):
        frame = tk.Frame(paned, background=PALETTE["surface"])
        paned.add(frame, weight=1)
        self._fig = Figure(facecolor=PALETTE["surface"])
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_axis_off()
        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        toolbar = NavigationToolbar2Tk(self._canvas, frame, pack_toolbar=False)
        toolbar.configure(background=PALETTE["surface"])
        for child in toolbar.winfo_children():
            try:
                child.configure(background=PALETTE["surface"])
            except tk.TclError:
                pass
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_controls(self, parent):
        self._merge_var = tk.StringVar()
        self._delete_var = tk.StringVar()
        self._sym_add_var = tk.StringVar()
        self._sym_convert_var = tk.StringVar()

        # The applied-elements list, status line, and Send button stay pinned to the bottom so
        # they're visible no matter which sections are open; packed first with side="bottom" so
        # the collapsible sections above can grow and shrink freely without displacing them.
        if self._on_apply_script is not None:
            ttk.Button(parent, text="Send to script…", style="Accent.TButton",
                       command=self._send_to_script).pack(side="bottom", fill="x", pady=(8, 0))
        self._status = ttk.Label(parent, text="", style="Muted.TLabel", wraplength=290, justify="left")
        self._status.pack(side="bottom", fill="x", pady=(6, 0))
        self._applied = ttk.Frame(parent)
        self._applied.pack(side="bottom", fill="x", pady=(10, 0))

        # --- collapsible control sections, run as an accordion (one open at a time) so every
        # section header stays visible and the column fits a small pane ---
        self._sections: list[CollapsibleSection] = []

        # --- display toggles ---
        display = self._section(parent, "Display", expanded=False)
        for text, var in (
            ("Atomic detail", self._show_atoms),
            ("Symmetry copies", self._show_copies),
            ("Constraints", self._show_constraints),
            ("Colour by symmetry copy", self._colour_by_copy),
        ):
            ttk.Checkbutton(display.body, text=text, variable=var, command=self._redraw).pack(anchor="w")

        # --- body list (scrolls when long, so the section keeps a bounded height) ---
        bodies = self._section(parent, "Bodies", expanded=True)
        self._body_list = ScrollableFrame(bodies.body, height=220)
        self._body_list.pack(fill="both", expand=True)

        # --- merge / delete ---
        actions = self._section(parent, "Manage bodies", expanded=False)
        self._action_row(actions.body, "Merge", self._merge_var, "first others…", self._apply_merge)
        self._action_row(actions.body, "Delete", self._delete_var, "bodies…", self._apply_delete)

        # --- symmetry: two distinct operations, the more common one (adding a symmetry to a
        # single body) on top, decomposing several bodies into a shared symmetry below ---
        sym = self._section(parent, "Symmetry", expanded=False).body
        self._action_row(sym, "Add symmetry to a body", self._sym_add_var,
                         "a body then a type, e.g. b1 c4", self._apply_add_symmetry, button="Apply")
        self._action_row(sym, "Decompose bodies into a symmetry", self._sym_convert_var,
                         "bodies then a type, e.g. b1 b2 b3 b4 c4", self._apply_convert_symmetry, button="Convert")

    def _section(self, parent, title, *, expanded: bool) -> CollapsibleSection:
        """A collapsible controls section, spaced from the one above it and wired into the accordion."""
        section = CollapsibleSection(parent, title, expanded=expanded, on_toggle=self._accordion)
        section.pack(fill="x", pady=(0, 6))
        self._sections.append(section)
        return section

    def _accordion(self, opened: CollapsibleSection):
        """Keep the sections mutually exclusive: expanding one collapses the rest, so the header
        bars are always in view. Collapsing a section leaves the others untouched."""
        if not opened.expanded:
            return
        for section in self._sections:
            if section is not opened and section.expanded:
                section.set_expanded(False)

    def _action_row(self, parent, label, var, hint, command, button="Apply"):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 2))
        ttk.Label(row, text=label, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = ttk.Entry(row, textvariable=var)
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(row, text=button, style="Icon.TButton", command=command).grid(row=1, column=1)
        row.columnconfigure(0, weight=1)
        entry.bind("<Return>", lambda _e: command())
        self._hint(row, hint, row=2)

    def _hint(self, parent, text, row):
        ttk.Label(parent, text=text, style="Muted.TLabel", font=FONTS["small"]).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))

    # ----- backend rebuild ----------------------------------------------------
    def _compose(self, elements: list[str]) -> str:
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        if _LOAD_BLOCK_RE.search(base) is None:  # target has no load block: fall back to a synthetic one
            base = _synth_load_block(self.pdb_path, self._splits)
        return _insert_elements(base, elements)

    def _rebuild(self, elements: list[str]) -> tuple[bool, str]:
        """Load the composed setup script through the backend and refresh the view/body list
        from the bodies that remain. Returns (ok, message); on failure nothing is mutated."""
        script = self._compose(elements)
        try:
            from ..wrapper.Rigidbody import Rigidbody
            rb = Rigidbody(script)
            data = rb.preview_structure()
            if not len(data["coords"]):
                return False, "the structure is empty"
            names = rb.body_names()
        except Exception as e:
            return False, str(e)

        self._data, self._names = data, names
        self._compute_bodies()
        if self._highlight is not None and self._highlight not in {b["index"] for b in self._bodies}:
            self._highlight = None  # the highlighted body was merged/deleted away
        self._rebuild_body_list()
        self._redraw()
        return True, ""

    def _compute_bodies(self):
        import numpy as np
        data = self._data
        body, copy, res = data["body"], data["copy"], data["residue_seq"]
        self._bodies = []
        for k, idx in enumerate(sorted(set(body.tolist()))):
            orig = (body == idx) & (copy == 0)
            resids = res[orig & (res >= 0)]
            self._bodies.append({
                "index": idx,
                "name": self._names[k] if k < len(self._names) else f"b{idx + 1}",
                "atoms": int(orig.sum()),
                "res": (int(resids.min()), int(resids.max())) if len(resids) else None,
                "colour": _BODY_COLORS[idx % len(_BODY_COLORS)],
            })

    # ----- body list ----------------------------------------------------------
    def _rebuild_body_list(self):
        for w in self._row_frames:
            w.destroy()
        self._row_frames = []
        self._rows = []  # (index, [widgets to recolour on highlight])
        for b in self._bodies:
            row = tk.Frame(self._body_list.body, background=PALETTE["surface"], cursor="hand2")
            row.pack(fill="x")

            swatch = tk.Frame(row, background=b["colour"], width=12, height=12)
            swatch.pack(side="left", padx=(2, 6), pady=4)
            swatch.pack_propagate(False)

            res = f"res {b['res'][0]}–{b['res'][1]}" if b["res"] else "no residues"
            label = tk.Label(
                row, text=f"{b['name']}", foreground=PALETTE["text"],
                font=FONTS["base"], anchor="w")
            label.pack(side="left")
            meta = tk.Label(
                row, text=f"{b['atoms']} atoms · {res}", foreground=PALETTE["muted"],
                font=FONTS["small"], anchor="e")
            meta.pack(side="right", padx=(0, 4))

            for w in (row, swatch, label, meta):
                w.bind("<Button-1>", lambda _e, i=b["index"]: self._toggle_highlight(i))
            # double-click the name (or the row) to rename the body
            for w in (row, label):
                w.bind("<Double-Button-1>", lambda _e, bb=b, lbl=label: self._start_rename(lbl, bb))
            self._row_frames.append(row)
            self._rows.append((b["index"], (row, label, meta)))
        self._refresh_row_highlight()

    def _refresh_row_highlight(self):
        """Recolour the rows to reflect the highlighted body, in place — so a click doesn't tear
        down the row widgets (which would make double-click-to-rename impossible)."""
        for index, widgets in self._rows:
            bg = PALETTE["accent_soft"] if index == self._highlight else PALETTE["surface"]
            for w in widgets:
                w.configure(background=bg)

    def _toggle_highlight(self, index: int):
        self._highlight = None if self._highlight == index else index
        self._refresh_row_highlight()
        self._redraw()

    def _start_rename(self, label: tk.Label, b: dict):
        """Replace a body's name label with an inline entry so the user can rename it. Committing
        applies a `rename <old> <new>` element; the backend keeps the body's default name too, so a
        rename can always be undone by renaming back."""
        old = b["name"]
        var = tk.StringVar(value=old)
        entry = tk.Entry(
            label.master, textvariable=var, font=FONTS["base"], width=14,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            insertbackground=PALETTE["text"], relief="flat", highlightthickness=1,
            highlightbackground=PALETTE["accent"], highlightcolor=PALETTE["accent"])
        label.pack_forget()
        entry.pack(side="left")
        entry.focus_set()
        entry.select_range(0, "end")

        state = {"done": False}

        def finish(apply: bool):
            if state["done"]:
                return
            state["done"] = True
            new = var.get().strip()
            entry.destroy()
            self._rebuild_body_list()  # restore the row to its normal label state
            if not apply or not new or new == old:
                return
            if any(c.isspace() for c in new):
                self._set_status("A body name cannot contain spaces.", ok=False)
                return
            self._apply_element(f"rename {old} {new}")

        entry.bind("<Return>", lambda _e: finish(True))
        entry.bind("<FocusOut>", lambda _e: finish(True))
        entry.bind("<Escape>", lambda _e: finish(False))

    # ----- drawing ------------------------------------------------------------
    def _redraw(self):
        ax = self._ax
        lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) if self._data is not None else None
        ax.clear()
        ax.set_axis_off()
        if self._data is None:
            ax.text2D(0.5, 0.5, "Could not read the structure", transform=ax.transAxes,
                      ha="center", va="center", color=PALETTE["muted"])
        else:
            draw_structure(
                ax, self._data, self._parse_splits(),
                show_atoms=self._show_atoms.get(),
                show_copies=self._show_copies.get(),
                show_constraints=self._show_constraints.get(),
                highlight_body=self._highlight,
                color_by="copy" if self._colour_by_copy.get() else "body",
            )
            if lims is not None and self._preserve_view:
                ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        self._preserve_view = True
        self._fig.set_layout_engine("tight")
        self._canvas.draw_idle()

    _preserve_view = False

    def _parse_splits(self) -> list[int]:
        return [int(t) for t in re.split(r"[,\s]+", self._splits.strip()) if t.isdigit()]

    # ----- setup actions ------------------------------------------------------
    def _apply_element(self, element: str):
        candidate = self._elements + [element]
        ok, msg = self._rebuild(candidate)
        if ok:
            self._elements = candidate
            self._rebuild_applied_list()
            self._set_status(f"Applied: {element}", ok=True)
        else:
            self._set_status(f"Rejected “{element}”: {msg}", ok=False)

    def _apply_merge(self):
        tokens = self._merge_var.get().split()
        if len(tokens) < 2:
            self._set_status("Merge needs a target body and at least one other.", ok=False)
            return
        self._apply_element("merge " + " ".join(tokens))
        self._merge_var.set("")

    def _apply_delete(self):
        tokens = self._delete_var.get().split()
        if not tokens:
            self._set_status("Delete needs at least one body.", ok=False)
            return
        self._apply_element("delete " + " ".join(tokens))
        self._delete_var.set("")

    def _apply_add_symmetry(self):
        """Add a symmetry to a single body: `symmetry <body> <type>` (e.g. b1 c4). A lone type is
        allowed for a single-body system, where the backend infers the body."""
        tokens = self._sym_add_var.get().split()
        if not tokens:
            self._set_status("Add symmetry needs a body and a type, e.g. b1 c4.", ok=False)
            return
        if len(tokens) > 2:
            self._set_status("Add symmetry takes one body and one type, e.g. b1 c4.", ok=False)
            return
        self._apply_element("symmetry " + " ".join(tokens))
        self._sym_add_var.set("")

    def _apply_convert_symmetry(self):
        """Decompose several bodies into one shared symmetry, collapsing the copies into the first
        body plus a fitted symmetry: `convert_to_symmetry { type <type> bodies <b…> }`."""
        tokens = self._sym_convert_var.get().split()
        if len(tokens) < 3:
            self._set_status("Decomposing needs at least two bodies and a type, e.g. b1 b2 c2.", ok=False)
            return
        *bodies, sym = tokens
        element = "convert_to_symmetry {\n    type " + sym + "\n    bodies " + " ".join(bodies) + "\n}"
        self._apply_element(element)
        self._sym_convert_var.set("")

    def _rebuild_applied_list(self):
        for w in self._applied.winfo_children():
            w.destroy()
        if not self._elements:
            return
        ttk.Label(self._applied, text="Applied elements", style="Muted.TLabel").pack(anchor="w")
        for i, element in enumerate(self._elements):
            row = ttk.Frame(self._applied)
            row.pack(fill="x", pady=1)
            # collapse any multi-line element (e.g. a convert_to_symmetry block) to one tidy line
            summary = " ".join(element.split())
            ttk.Label(row, text=summary, font=FONTS["mono"]).pack(side="left")
            ttk.Button(row, text="✕", width=2, style="Icon.TButton",
                       command=lambda i=i: self._remove_element(i)).pack(side="right")

    def _remove_element(self, i: int):
        candidate = self._elements[:i] + self._elements[i + 1:]
        ok, msg = self._rebuild(candidate)
        if ok:
            self._elements = candidate
            self._rebuild_applied_list()
            self._set_status("Removed an element.", ok=True)
        else:
            self._set_status(f"Could not remove: {msg}", ok=False)

    def _set_status(self, text: str, *, ok: bool):
        self._status.configure(text=text, foreground=PALETTE["ok_border"] if ok else PALETTE["danger"])

    # ----- send to script -----------------------------------------------------
    def _send_to_script(self):
        if not self._elements:
            self._set_status("No changes to send.", ok=False)
            return
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        new = _insert_elements(base, self._elements)
        ScriptDiffDialog(self, base, new, on_confirm=lambda: self._on_apply_script(new))


class ScriptDiffDialog(tk.Toplevel):
    """A modal side-by-side diff: the original script on the left with removed lines tinted red,
    the new script on the right with added lines tinted green. Confirm applies the change."""

    def __init__(self, parent, old: str, new: str, on_confirm: Callable[[], None]):
        super().__init__(parent)
        self.title("Preview script changes")
        self.configure(background=PALETTE["bg"])
        self.transient(parent.winfo_toplevel())
        self._on_confirm = on_confirm

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        left = self._make_text(body, "Original")
        right = self._make_text(body, "With changes")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self._fill(left, right, old, new)

        buttons = ttk.Frame(self, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply changes", style="Accent.TButton",
                   command=self._confirm).pack(side="right", padx=(0, 8))

        self.geometry("820x520")
        self.grab_set()

    def _make_text(self, parent, heading) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=heading, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        text = tk.Text(
            frame, wrap="none", font=FONTS["mono"], relief="flat", borderwidth=0,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            padx=8, pady=6, height=24, width=48)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        text.tag_configure("del", background=PALETTE["bad"], foreground=PALETTE["danger"])
        text.tag_configure("add", background="#e3f4e9", foreground="#2f7d4f")
        frame.text = text  # type: ignore[attr-defined]
        return frame

    def _fill(self, left, right, old: str, new: str):
        lt, rt = left.text, right.text  # type: ignore[attr-defined]
        old_lines, new_lines = old.splitlines(), new.splitlines()
        sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            a, b = old_lines[i1:i2], new_lines[j1:j2]
            if tag == "equal":
                for line in a:
                    lt.insert("end", line + "\n")
                for line in b:
                    rt.insert("end", line + "\n")
                continue
            # keep the two sides row-aligned by padding the shorter side with blank lines
            rows = max(len(a), len(b))
            for k in range(rows):
                lt.insert("end", (a[k] if k < len(a) else "") + "\n", "del" if k < len(a) else ())
                rt.insert("end", (b[k] if k < len(b) else "") + "\n", "add" if k < len(b) else ())
        lt.configure(state="disabled")
        rt.configure(state="disabled")

    def _confirm(self):
        self._on_confirm()
        self.destroy()

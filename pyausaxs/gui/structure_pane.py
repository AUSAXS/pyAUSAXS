# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Structure-inspection / management pane.

Opened from a "View structure" button once a PDB is loaded. It shows a large 3D view of the
structure and lets the user inspect and re-organise its bodies without touching the run:

  * toggle atomic detail (all-atom cloud), symmetry copies, and constraints;
  * re-split the structure, and see/highlight the individual bodies via a scrolling, backend-fed
    list; symmetry replicas fold out beneath their base body and can be highlighted one at a time;
  * merge or delete bodies, convert a set of bodies to a symmetry, and add constraints between
    bodies, by adding the corresponding setup elements (`merge` / `delete` /
    `convert_to_symmetry` / `constrain` / `autoconstrain`);
  * preview the resulting script changes as a red/green diff and send them to the editor.

The body list is authoritative: every edit is applied by rebuilding the setup script through the
backend and re-reading the bodies that remain, so a merge or delete really does make bodies
disappear from the list (and the plot) exactly as it will at run time.
"""

import re
import difflib
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .plotting import draw_structure, _BODY_COLORS
from .theme import FONTS, PALETTE
from .widgets import CollapsibleSection, PlaceholderEntry, ScrollableFrame, ellipsize_label

# the load block whose bodies we manage; setup elements are inserted just after it
_LOAD_BLOCK_RE = re.compile(r"load\s*\{.*?\}", re.DOTALL)

# Every element the structure pane reads or writes, so that an external edit to any of them (in the main editor) marks the view stale: the
# load block plus all body-affecting setup elements. The tail captures either a whole brace block or the rest of the inline line, so an edit 
# inside a symmetry/constraint block counts too. Longer keywords precede the prefixes they contain.
_STALE_TAIL = r"(?:[ \t]*\{.*?\}|[^\n]*)"
_STALE_RE = re.compile(
    r"(?m)^[ \t]*(?:load[ \t]*\{.*?\}|"
    r"(?:merge|delete|rename|convert_to_symmetry|symmetry|constraint|constrain"
    r"|autoconstraints|autoconstrain|copy_body|copy)\b" + _STALE_TAIL + r")",
    re.DOTALL,
)


def _structure_signature(script: str) -> tuple:
    """A fingerprint of the parts of `script` the structure pane cares about, so staleness is flagged when — and only when — one of them 
    changes in the main editor. Broader than the rigid-body pane's preview signature: it also covers rename/merge/copy, which change the 
    body list this pane shows even when they leave the drawn geometry untouched."""
    return tuple(m.group(0) for m in _STALE_RE.finditer(script))


def _synth_load_block(pdb_path: str, splits: str) -> str:
    inner = [f"    pdb {pdb_path}"]
    if splits.strip():
        inner.append(f"    split {splits.strip()}")
    return "load {\n" + "\n".join(inner) + "\n}"


def _norm_splits(value: str) -> str:
    """Normalise a splits string to a canonical space-separated form, so equal splits written with
    different spacing/commas compare equal (and don't churn the load block)."""
    return " ".join(t for t in re.split(r"[,\s]+", value.strip()) if t)


def _load_split(base: str) -> str:
    """The split directive currently in the base script's load block, as a raw string ("" if none)."""
    match = _LOAD_BLOCK_RE.search(base)
    if not match:
        return ""
    inner = re.match(r"load\s*\{(.*)\}", match.group(0), re.DOTALL)
    for line in (inner.group(1).splitlines() if inner else []):
        tokens = line.split(None, 1)
        if tokens and tokens[0] == "split":
            return tokens[1].strip() if len(tokens) == 2 else ""
    return ""


def _with_split(base: str, splits: str) -> str:
    """Return `base` with its load block's split directive set to `splits` (added if missing, removed if empty), touching only the split line so 
    unrelated formatting is preserved. A no-op when the split is already `splits`, so a diff shows a change only when the user actually re-split."""
    match = _LOAD_BLOCK_RE.search(base)
    if not match or _norm_splits(_load_split(base)) == _norm_splits(splits):
        return base
    block = match.group(0)
    splits = splits.strip()
    line_re = re.compile(r"^([ \t]*)split\b[^\n]*", re.MULTILINE)
    if line_re.search(block):
        if splits:
            new_block = line_re.sub(lambda m: f"{m.group(1)}split {splits}", block, count=1)
        else:  # drop the split line (and its own line break) entirely
            new_block = re.sub(r"[ \t]*split\b[^\n]*\n?", "", block, count=1)
    elif splits:  # insert a split line before the closing brace, matching pdb's indentation
        pm = re.search(r"^([ \t]*)pdb\b", block, re.MULTILINE)
        indent = pm.group(1) if pm else "    "
        new_block = re.sub(r"\n?\}$", f"\n{indent}split {splits}\n}}", block, count=1)
    else:
        return base
    return base[:match.start()] + new_block + base[match.end():]


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
        base_signature: Optional[Callable[[str], object]] = None,
    ):
        super().__init__(parent)
        self.pdb_path = pdb_path
        self._splits = splits
        self._base_script = base_script          # target script to diff/patch, or None
        self._on_apply_script = on_apply_script   # apply a confirmed new script, or None
        # reduce the base script to a structural fingerprint, so a later edit to the same body/split
        # setup is detected as "stale". Defaults to the structure-pane signature, which covers every
        # element this pane reads or writes (rename/merge/split/symmetry/constraint/...).
        self._base_signature = base_signature or _structure_signature
        self._built_sig = None                    # base fingerprint the current view was built from
        self.title = "Structure: " + pdb_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        self._elements: list[str] = []            # setup elements the user has applied, in order
        self._data: dict | None = None            # last good preview-structure dict
        self._names: list[str] = []               # body names aligned to body indices
        self._bodies: list[dict] = []             # per-body summary rows (index/name/atoms/res/copies)
        self._replica_info: dict[tuple[int, int], dict] = {}  # (body, copy) -> {"type", "name"}
        # the view selection isolated in the plot, as a (body, copy) pair: copy None highlights the
        # whole body (all its symmetry copies), an int highlights just that one replica. None = nothing.
        self._highlight: tuple[int, int | None] | None = None
        self._expanded_bodies: set[int] = set()   # body indices whose replica children are unfolded
        self._row_frames: list[tk.Widget] = []
        self._rows: list[tuple] = []              # ((body, copy) selector, recolourable row widgets)
        self._body_row_frames: dict[int, tk.Frame] = {}       # body index -> its row (anchor to pack replicas after)
        self._body_chevrons: dict[int, tk.Label] = {}         # body index -> its fold chevron, flipped in place
        self._replica_row_frames: dict[int, list[tk.Frame]] = {}  # body index -> its currently-built replica rows
        self._redraw_job: Optional[str] = None    # pending after_idle handle for a deferred _redraw()

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

    def destroy(self):
        # cancel a pending deferred redraw so it doesn't fire against a torn-down figure/canvas
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
            self._redraw_job = None
        super().destroy()

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
        # The applied-elements list, status line, and Send button stay pinned to the bottom so they're visible no matter which sections 
        # are open; packed first with side="bottom" so the collapsible sections above can grow and shrink freely without displacing them.
        if self._on_apply_script is not None:
            ttk.Button(parent, text="Send to script…", style="Accent.TButton",
                       command=self._send_to_script).pack(side="bottom", fill="x", pady=(8, 0))
        self._status = ttk.Label(parent, text="", style="Muted.TLabel", wraplength=290, justify="left")
        self._status.pack(side="bottom", fill="x", pady=(6, 0))

        # An amber action bar styled like a section header, but it triggers a refresh instead of
        # expanding. It is packed above the sections only while the view is stale (see _set_stale).
        self._build_refresh_bar(parent)

        # --- collapsible control sections; each opens and closes independently, so the body list
        # can stay open for reference (e.g. body names) while another section is in use
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

        # --- body list (scrolls when long, so the section keeps a bounded height), with a splits
        # editor above it so the structure can be re-split here without leaving the pane ---
        bodies = self._section(parent, "Bodies", expanded=True)
        splits_row = ttk.Frame(bodies.body)
        splits_row.pack(fill="x", pady=(0, 6))
        ttk.Label(splits_row, text="Split at residues", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w")
        self._splits_var = tk.StringVar(value=self._splits)
        splits_entry = ttk.Entry(splits_row, textvariable=self._splits_var)
        splits_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(splits_row, text="Apply", style="Icon.TButton", command=self._apply_splits).grid(row=1, column=1)
        splits_row.columnconfigure(0, weight=1)
        splits_entry.bind("<Return>", lambda _e: self._apply_splits())
        self._body_list = ScrollableFrame(bodies.body, max_height=220)
        self._body_list.pack(fill="both", expand=True)

        # --- merge / delete ---
        actions = self._section(parent, "Manage bodies", expanded=False)
        self._rename_entry = self._action_row(actions.body, "Rename", "old new", self._apply_rename, button="Rename")
        self._merge_entry = self._action_row(actions.body, "Merge", "first others...", self._apply_merge)
        self._delete_entry = self._action_row(actions.body, "Delete", "body", self._apply_delete)

        # --- symmetry: two distinct operations, the more common one (adding a symmetry to a single body) on top, decomposing several bodies 
        # into a shared symmetry below
        sym = self._section(parent, "Symmetry", expanded=False).body
        self._sym_add_entry = self._action_row(
            sym, "Add symmetry to a body", "body type", self._apply_add_symmetry, button="Apply"
        )
        self._sym_convert_entry = self._action_row(
            sym, "Decompose bodies into a symmetry", "bodies... type", self._apply_convert_symmetry, button="Convert"
        )

        # --- constraints: auto-generate a set (backbone) on top, then add an individual constraint between two bodies below. Existing constraints
        # are edited by removing/re-adding via the applied-elements list, and shown in the view via the "Constraints" display toggle.
        con = self._section(parent, "Constraints", expanded=False).body
        self._autoconstrain_entry = self._action_row(
            con, "Auto-generate constraints", "backbone", self._apply_autoconstrain, button="Generate"
        )
        self._constraint_entry = self._action_row(
            con, "Constrain two bodies", "body1 body2 type", self._apply_add_constraint, button="Add"
        )

        # --- staged elements: kept as its own accordion section, always last, so it can be collapsed
        # like any other section when the user needs more room for e.g. Bodies + Constraints at once
        self._applied = self._section(parent, "Applied elements", expanded=True).body
        self._rebuild_applied_list()  # seed the initial "no changes" placeholder

    def _section(self, parent, title, *, expanded: bool) -> CollapsibleSection:
        """A collapsible controls section, spaced from the one above it. Sections are independent,
        so any number can be open at once."""
        section = CollapsibleSection(parent, title, expanded=expanded)
        section.pack(fill="x", pady=(0, 6))
        self._sections.append(section)
        return section

    # amber "the script changed" attention colours, distinct from the neutral section headers
    _WARN_BG = "#f0ad4e"
    _WARN_FG = "#4a3208"

    def _build_refresh_bar(self, parent):
        bar = tk.Frame(parent, background=self._WARN_BG, cursor="hand2")
        # the reload glyph only exists in the regular weight of this font, not the bold heading one
        icon = tk.Label(bar, text="↻", background=self._WARN_BG, foreground=self._WARN_FG,
                        font=(FONTS["base"][0], 14), width=2)
        icon.pack(side="left", padx=(6, 0), pady=6)
        title = tk.Label(bar, text="Script changed - refresh", background=self._WARN_BG,
                         foreground=self._WARN_FG, font=FONTS["heading"])
        title.pack(side="left", pady=6)
        for w in (bar, icon, title):
            w.bind("<Button-1>", lambda _e: self._do_refresh())
        self._refresh_bar = bar  # created unpacked; _set_stale packs it above the sections

    def _action_row(self, parent, label, hint, command, button="Apply") -> PlaceholderEntry:
        """An action row: a short label, a text entry whose greyed placeholder carries the format hint (so no separate hint line is needed),
        and a button. Returns the entry so the caller can read it with .get() and reset it with .clear()."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text=label, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = PlaceholderEntry(row, hint)
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(row, text=button, style="Icon.TButton", command=command).grid(row=1, column=1)
        row.columnconfigure(0, weight=1)
        entry.bind("<Return>", lambda _e: command())
        return entry

    # ----- backend rebuild ----------------------------------------------------
    def _compose(self, elements: list[str]) -> str:
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        if _LOAD_BLOCK_RE.search(base) is None:  # target has no load block: fall back to a synthetic one
            base = _synth_load_block(self.pdb_path, self._splits)
        else:  # honour the pane's (possibly edited) splits over whatever the base load block carries
            base = _with_split(base, self._splits)
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
            layout = rb.symmetry_layout()
        except Exception as e:
            return False, str(e)

        self._data, self._names = data, names
        self._replica_info = {
            (int(body), int(copy)): {"type": t, "name": nm}
            for body, copy, t, nm in zip(layout["body"], layout["copy"], layout["type"], layout["name"])
        }
        self._compute_bodies()
        valid = {(b["index"], None) for b in self._bodies}
        valid |= {(b["index"], c) for b in self._bodies for c in b["copies"]}
        if self._highlight is not None and self._highlight not in valid:
            self._highlight = None  # the highlighted body/replica was merged/deleted/de-symmetrised away
        self._rebuild_body_list()
        self._schedule_redraw()
        # the view now matches this base, so it is no longer stale
        self._built_sig = self._base_sig()
        self._set_stale(False)
        return True, ""

    # ----- staleness / refresh ------------------------------------------------
    def _base_sig(self):
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        return self._base_signature(base)

    def check_stale(self):
        """Flag the view as stale when the base script has changed since it was built (e.g. the user
        edited the load block in the main editor). Called when the pane is switched back to."""
        self._set_stale(self._built_sig is not None and self._base_sig() != self._built_sig)

    def _set_stale(self, stale: bool):
        if not hasattr(self, "_refresh_bar") or not self._sections:
            return
        if stale and not self._refresh_bar.winfo_ismapped():
            self._refresh_bar.pack(fill="x", pady=(0, 6), before=self._sections[0])
        elif not stale and self._refresh_bar.winfo_ismapped():
            self._refresh_bar.pack_forget()

    def _do_refresh(self):
        """Reload the view from the current script, discarding all staged edits (they are the only
        thing lost, so confirm only when there are any)."""
        if self._elements:
            n = len(self._elements)
            if not messagebox.askyesno(
                "Refresh from script",
                f"Discard {n} staged change{'' if n == 1 else 's'} and reload from the current script?",
                parent=self):
                return
        self._elements = []
        self._rebuild_applied_list()
        # re-read the splits from the (possibly re-edited) script, so the field mirrors it again
        base = self._base_script() if self._base_script else ""
        if _LOAD_BLOCK_RE.search(base):
            self._splits = _load_split(base)
            self._splits_var.set(self._splits)
        ok, msg = self._rebuild(self._elements)
        self._set_status("Reloaded from the current script." if ok
                         else f"Could not reload from the script: {msg}", ok=ok)

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
                # every copy index present for this body; copy 0 is the base, >0 are symmetry replicas
                "copies": sorted(set(copy[body == idx].tolist())),
            })

    # ----- body list ----------------------------------------------------------
    def _rebuild_body_list(self):
        """Full teardown and rebuild of every row. Used only when the underlying body data itself changes (a backend rebuild) or a 
        renamed row's inline entry needs restoring — those are the only cases where every row actually needs replacing. """
        for w in self._row_frames:
            w.destroy()
        self._row_frames = []
        self._rows = []  # ((body, copy) selector, [widgets to recolour on highlight])
        self._body_row_frames = {}
        self._body_chevrons = {}
        self._replica_row_frames = {}
        for b in self._bodies:
            self._build_body_row(b)
            # symmetry replicas (copy > 0), foldable beneath the base body they belong to
            if b["index"] in self._expanded_bodies:
                self._replica_row_frames[b["index"]] = [
                    self._build_replica_row(b, c) for c in b["copies"][1:]
                ]
        self._refresh_row_highlight()

    def _build_body_row(self, b: dict):
        replicas = b["copies"][1:]
        row = tk.Frame(self._body_list.body, background=PALETTE["surface"], cursor="hand2")
        row.pack(fill="x")

        # a fold chevron for bodies that have replicas, or a matching-width spacer for those that don't
        # so the swatches line up. The chevron toggles the fold without
        # changing the highlight.
        if replicas:
            chevron = tk.Label(
                row, text="▾" if b["index"] in self._expanded_bodies else "▸",
                background=PALETTE["surface"], foreground=PALETTE["muted"], font=FONTS["small"], width=2)
            chevron.pack(side="left")
            chevron.bind("<Button-1>", lambda _e, i=b["index"]: self._toggle_body_expand(i))
            self._body_chevrons[b["index"]] = chevron
        else:
            tk.Label(row, text="", background=PALETTE["surface"], font=FONTS["small"], width=2).pack(side="left")

        swatch = tk.Frame(row, background=b["colour"], width=12, height=12)
        swatch.pack(side="left", padx=(0, 6), pady=4)
        swatch.pack_propagate(False)

        res = f"res {b['res'][0]}–{b['res'][1]}" if b["res"] else "no residues"
        label = tk.Label(row, text=b["name"], foreground=PALETTE["text"], font=FONTS["base"], anchor="w")
        label.pack(side="left")
        extra = f" · {len(replicas)} copies" if replicas else ""
        meta = tk.Label(
            row, text=f"{b['atoms']} atoms · {res}{extra}", foreground=PALETTE["muted"],
            font=FONTS["small"], anchor="e")
        meta.pack(side="right", padx=(0, 4))

        for w in (row, swatch, label, meta):  # click anywhere on the row highlights the whole body
            w.bind("<Button-1>", lambda _e, i=b["index"]: self._toggle_highlight(i, None))
        for w in (row, label):  # double-click the name (or the row) to rename the body
            w.bind("<Double-Button-1>", lambda _e, nm=b["name"], lbl=label: self._start_rename(lbl, nm))
        self._row_frames.append(row)
        self._rows.append(((b["index"], None), (row, label, meta)))
        self._body_row_frames[b["index"]] = row

    def _build_replica_row(self, b: dict, copy: int, *, after: tk.Widget | None = None) -> tk.Frame:
        """A single symmetry-replica child row, indented under its base body. Clicking it isolates just that replica in the view. Name and
        type badge come straight from the backend's symmetry layout, keyed to this (body, copy) pair. Packed right after `after` (an
        existing row) when given, so a single body's replicas can be spliced into their exact spot in the list without disturbing any
        other row — see _toggle_body_expand."""
        info = self._replica_info[(b["index"], copy)]
        row = tk.Frame(self._body_list.body, background=PALETTE["surface"], cursor="hand2")
        if after is not None:
            row.pack(fill="x", after=after)
        else:
            row.pack(fill="x")
        tk.Frame(row, background=PALETTE["surface"], width=28).pack(side="left")  # indent past the chevron
        swatch = tk.Frame(row, background=b["colour"], width=8, height=8)
        swatch.pack(side="left", padx=(0, 6))
        swatch.pack_propagate(False)
        label = tk.Label(row, text=info["name"], foreground=PALETTE["muted"], font=FONTS["small"], anchor="w")
        label.pack(side="left")
        badge = tk.Label(row, text=f"[{info['type']}]", foreground=PALETTE["accent"], font=FONTS["small"], anchor="w")
        badge.pack(side="left", padx=(4, 0))
        widgets = [row, swatch, label, badge]
        for w in widgets:
            w.bind("<Button-1>", lambda _e, i=b["index"], c=copy: self._toggle_highlight(i, c))
        for w in (row, label):  # double-click the name (or the row) to rename the replica, same as a base body
            w.bind("<Double-Button-1>", lambda _e, nm=info["name"], lbl=label: self._start_rename(lbl, nm))
        self._row_frames.append(row)
        self._rows.append(((b["index"], copy), tuple(widgets)))
        return row

    def _refresh_row_highlight(self):
        """Recolour the rows to reflect the highlighted body/replica, in place — so a click doesn't
        tear down the row widgets (which would make double-click-to-rename impossible)."""
        for selector, widgets in self._rows:
            bg = PALETTE["accent_soft"] if selector == self._highlight else PALETTE["surface"]
            for w in widgets:
                w.configure(background=bg)

    def _toggle_body_expand(self, body: int):
        """Fold/unfold a body's symmetry-replica rows in place: splice its replica rows in or out
        right after its own row, and flip its chevron — without touching any other row. """
        expanding = body not in self._expanded_bodies
        if expanding:
            self._expanded_bodies.add(body)
        else:
            self._expanded_bodies.discard(body)
        chevron = self._body_chevrons.get(body)
        if chevron is not None:
            chevron.configure(text="▾" if expanding else "▸")

        if expanding:
            b = next(bb for bb in self._bodies if bb["index"] == body)
            after = self._body_row_frames[body]
            rows = []
            for c in b["copies"][1:]:
                after = self._build_replica_row(b, c, after=after)
                rows.append(after)
            self._replica_row_frames[body] = rows
        else:
            stale = set(self._replica_row_frames.pop(body, []))
            if stale:
                self._rows = [(sel, widgets) for sel, widgets in self._rows if widgets[0] not in stale]
                self._row_frames = [w for w in self._row_frames if w not in stale]
                for row in stale:
                    row.destroy()
        self._refresh_row_highlight()

    def _toggle_highlight(self, body: int, copy: int | None):
        selector = (body, copy)
        self._highlight = None if self._highlight == selector else selector
        # isolating a specific replica is pointless while copies are hidden, so reveal them
        if copy not in (None, 0) and self._highlight is not None and not self._show_copies.get():
            self._show_copies.set(True)
        self._refresh_row_highlight()
        self._schedule_redraw()

    def _start_rename(self, label: tk.Label, old: str):
        """Replace a body or replica name label with an inline entry so the user can rename it in place. Committing applies a 
        `rename <old> <new>` element; the backend keeps the default name too, so a rename can always be undone by renaming back. Works the 
        same for a base body's name and a replica's addressable name (e.g. "b1s1r1"), since both are just names the backend accepts."""
        # match the label's own font (replica labels use a smaller font than base bodies), and take the label's spot in the 
        # row's left-to-right pack order so a sibling packed after it (e.g. the replica's type badge) doesn't visually jump 
        # to its left while the entry is showing
        siblings = label.master.pack_slaves()
        after_idx = siblings.index(label) + 1
        before = siblings[after_idx] if after_idx < len(siblings) else None
        var = tk.StringVar(value=old)
        entry = tk.Entry(
            label.master, textvariable=var, font=label.cget("font"), width=14,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            insertbackground=PALETTE["text"], relief="flat", highlightthickness=1,
            highlightbackground=PALETTE["accent"], highlightcolor=PALETTE["accent"])
        label.pack_forget()
        if before is not None:
            entry.pack(side="left", before=before)
        else:
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
                self._set_status("A name cannot contain spaces.", ok=False)
                return
            self._apply_element(f"rename {old} {new}")

        entry.bind("<Return>", lambda _e: finish(True))
        entry.bind("<FocusOut>", lambda _e: finish(True))
        entry.bind("<Escape>", lambda _e: finish(False))

    # ----- drawing ------------------------------------------------------------
    def _schedule_redraw(self):
        """Defer the actual (comparatively expensive) matplotlib redraw to the next idle pass of the Tk event loop, instead of running it inline. """
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after_idle(self._run_scheduled_redraw)

    def _run_scheduled_redraw(self):
        self._redraw_job = None
        self._redraw()

    def _redraw(self):
        ax = self._ax
        view = self._get_orientation()
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
                highlight_body=self._highlight[0] if self._highlight else None,
                highlight_copy=self._highlight[1] if self._highlight else None,
                color_by="copy" if self._colour_by_copy.get() else "body",
            )
            if lims is not None and self._preserve_view:
                ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        if self._preserve_view:  # keep the camera angle across redraws, not just the zoom
            self._set_orientation(view)
        self._preserve_view = True
        self._fig.set_layout_engine("tight")
        self._canvas.draw_idle()

    _preserve_view = False

    # ----- camera --------------------------------------------------------------
    # The rigid-body pane and this pane show the same structure in two separate figures; rather than truly sync them, each adopts the other's 
    # camera angle when the user switches to it, which is enough to feel continuous. Only the orientation is carried, not the zoom, since the 
    # two structures can diverge (after a merge/delete) and shared limits would then clip.
    def get_camera_orientation(self) -> tuple:
        return self._get_orientation()

    def set_camera_orientation(self, cam: Optional[tuple]):
        if cam is None:
            return
        self._set_orientation(cam)
        self._canvas.draw_idle()

    def _get_orientation(self) -> tuple:
        ax = self._ax
        return (ax.elev, ax.azim, getattr(ax, "roll", 0.0))

    def _set_orientation(self, cam: tuple):
        elev, azim, roll = cam
        try:
            self._ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:  # matplotlib < 3.5 has no roll
            self._ax.view_init(elev=elev, azim=azim)

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

    def _apply_splits(self):
        """Re-split the structure at the residue numbers in the splits field and rebuild the view. The split lives in the load block, so it is 
        applied by recomposing (see _with_split); on a bad value the field is reverted so it always mirrors the splits actually in force."""
        new = self._splits_var.get().strip()
        if _norm_splits(new) == _norm_splits(self._splits):
            self._splits_var.set(new)
            return
        prev = self._splits
        self._splits = new
        ok, msg = self._rebuild(self._elements)
        if ok:
            self._splits_var.set(new)
            self._set_status(f"Re-split at {new}." if new else "Removed all splits.", ok=True)
        else:
            self._splits = prev
            self._splits_var.set(prev)
            self._set_status(f"Could not re-split: {msg}", ok=False)

    def _apply_rename(self):
        """Rename a body: `rename <old> <new>`, the same element the inline double-click-to-rename
        (see _start_rename) applies, just entered as two tokens instead of typed in place."""
        tokens = self._rename_entry.get().split()
        if len(tokens) != 2:
            self._set_status("Rename needs the current name and the new name, e.g. b1 core.", ok=False)
            return
        old, new = tokens
        if old == new:
            self._set_status("The new name is the same as the current one.", ok=False)
            return
        self._apply_element(f"rename {old} {new}")
        self._rename_entry.clear()

    def _apply_merge(self):
        tokens = self._merge_entry.get().split()
        if len(tokens) < 2:
            self._set_status("Merge needs a target body and at least one other.", ok=False)
            return
        self._apply_element("merge " + " ".join(tokens))
        self._merge_entry.clear()

    def _apply_delete(self):
        tokens = self._delete_entry.get().split()
        if not tokens:
            self._set_status("Delete needs at least one body.", ok=False)
            return
        self._apply_element("delete " + " ".join(tokens))
        self._delete_entry.clear()

    def _apply_add_symmetry(self):
        """Add a symmetry to a single body: `symmetry <body> <type>` (e.g. b1 c4). A lone type is
        allowed for a single-body system, where the backend infers the body."""
        tokens = self._sym_add_entry.get().split()
        if not tokens:
            self._set_status("Add symmetry needs a body and a type, e.g. b1 c4.", ok=False)
            return
        if len(tokens) > 2:
            self._set_status("Add symmetry takes one body and one type, e.g. b1 c4.", ok=False)
            return
        self._apply_element("symmetry " + " ".join(tokens))
        self._sym_add_entry.clear()

    def _apply_convert_symmetry(self):
        """Decompose several bodies into one shared symmetry, collapsing the copies into the first
        body plus a fitted symmetry: `convert_to_symmetry { type <type> bodies <b…> }`."""
        tokens = self._sym_convert_entry.get().split()
        if len(tokens) < 3:
            self._set_status("Decomposing needs at least two bodies and a type, e.g. b1 b2 c2.", ok=False)
            return
        *bodies, sym = tokens
        element = "convert_to_symmetry {\n    type " + sym + "\n    bodies " + " ".join(bodies) + "\n}"
        self._apply_element(element)
        self._sym_convert_entry.clear()

    def _apply_autoconstrain(self):
        """Auto-generate a set of constraints: `autoconstrain <backbone|none>`. Defaults to
        backbone, the usual choice; `none` clears any auto-generated set."""
        choice = self._autoconstrain_entry.get().strip() or "backbone"
        self._apply_element(f"autoconstrain {choice}")
        self._autoconstrain_entry.clear()

    def _apply_add_constraint(self):
        """Add a distance constraint between two bodies: `<body1> <body2> <type> [distance]`.
        `bond` and `cm` need only the two bodies; `attract` and `repel` also take a target
        distance (e.g. b1 b2 attract 30). Built as a `constrain { … }` block for the backend."""
        tokens = self._constraint_entry.get().split()
        if len(tokens) < 3:
            self._set_status("A constraint needs two bodies and a type, e.g. b1 b2 cm.", ok=False)
            return
        body1, body2, ctype, *rest = tokens
        lines = [f"    first {body1}", f"    second {body2}", f"    type {ctype}"]
        if ctype in ("attract", "repel"):
            if len(rest) != 1:
                self._set_status(f"A {ctype} constraint needs a distance, e.g. b1 b2 {ctype} 30.", ok=False)
                return
            lines.append(f"    distance {rest[0]}")
        elif rest:
            self._set_status(f"A {ctype} constraint takes no arguments beyond the two bodies.", ok=False)
            return
        self._apply_element("constrain {\n" + "\n".join(lines) + "\n}")
        self._constraint_entry.clear()

    def _rebuild_applied_list(self):
        for w in self._applied.winfo_children():
            w.destroy()
        if not self._elements:
            ttk.Label(self._applied, text="No changes staged yet.", style="Muted.TLabel").pack(anchor="w")
            return
        # bounded and scrollable, so a long list of edits never grows past the section's own space
        scroll = ScrollableFrame(self._applied, max_height=150)
        scroll.pack(fill="x")
        for i, element in enumerate(self._elements):
            row = ttk.Frame(scroll.body)
            row.pack(fill="x", pady=1)
            # the delete button is packed first at side=right so it always keeps its full size; the summary then fills whatever width is left and 
            # is ellipsized to it, so a wide element (e.g. a constrain block collapsed to one line) can never push the button out of reach
            ttk.Button(row, text="✕", width=2, style="Icon.TButton", command=lambda i=i: self._remove_element(i)).pack(side="right", padx=(4, 0))
            summary = ttk.Label(row, font=FONTS["mono"], anchor="w")
            summary.pack(side="left", fill="x", expand=True)
            # collapse any multi-line element (e.g. a convert_to_symmetry block) to one tidy line
            ellipsize_label(summary, " ".join(element.split()))

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
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        new_base = _with_split(base, self._splits) if _LOAD_BLOCK_RE.search(base) else base
        new = _insert_elements(new_base, self._elements)
        if new == base:  # neither the splits nor the staged elements actually differ from the base
            self._set_status("No changes to send.", ok=False)
            return
        ScriptDiffDialog(self, base, new, on_confirm=lambda: self._confirm_send(new))

    def _confirm_send(self, new: str):
        """Write the composed script back to the editor, then drop the staged edits: they now live
        in the base, so keeping them staged would double-apply them on the next rebuild."""
        self._on_apply_script(new)
        self._elements = []
        self._rebuild_applied_list()
        self._rebuild(self._elements)  # rebuild from the new base, re-recording its fingerprint


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

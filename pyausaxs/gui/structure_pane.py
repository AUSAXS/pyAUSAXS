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

from . import plotting
from .plotting import draw_structure, nearest_ca_residue, residue_ca_mask, _BODY_COLORS
from .theme import FONTS, PALETTE, blend
from .widgets import (
    CollapsibleSection, PlaceholderEntry, ScrollableFrame, ellipsize_label, quote_script_value,
)

# the load block whose bodies we manage; setup elements are inserted just after it
_LOAD_BLOCK_RE = re.compile(r"load\s*\{.*?\}", re.DOTALL)

# display label -> draw_structure `color_by` mode; the first entry is the default
COLOUR_BY_MODES = {"Body": "body", "Symmetry copy": "copy", "Residue": "residue"}
COLOUR_BY_LABELS = list(COLOUR_BY_MODES)

_SUBOPTION_INDENT = 16  # px a folded-out display sub-option is inset from its parent toggle

# Every element the structure pane reads or writes, so that an external edit to any of them (in the main editor) marks the view stale: the
# load block plus all body-affecting setup elements. The tail captures either a whole brace block or the rest of the inline line, so an edit 
# inside a symmetry/constraint block counts too. Longer keywords precede the prefixes they contain.
_STALE_TAIL = r"(?:[ \t]*\{.*?\}|[^\n]*)"
_STALE_RE = re.compile(
    r"(?m)^(?:[ \t]*(?:load[ \t]*\{.*?\}|"
    r"(?:merge|delete|rename|convert_to_symmetry|constraint|constrain"
    r"|autoconstraints|autoconstrain|copy_body|copy|split)\b)" + _STALE_TAIL
    + r"|symmetry\b" + _STALE_TAIL + r")",
    re.DOTALL,
)

# Setup elements are emitted in declaration order by the backend. Newly staged structure-pane elements must therefore be appended after existing
# setup elements rather than blindly inserted immediately after `load`; otherwise a staged rename can precede an existing symmetry declaration.
_SETUP_RE = re.compile(
    r"(?m)^(?:[ \t]*(?:merge|delete|rename|convert_to_symmetry|constraint|constrain"
    r"|autoconstraints|autoconstrain|copy_body|copy|split)\b" + _STALE_TAIL
    + r"|symmetry\b" + _STALE_TAIL + r")",
    re.DOTALL,
)

# Constraint declarations and the staged elements that must remain after existing setup declarations. Every other structure element must
# be declared before the first constraint, because constraints are indexed by the structure that exists when they are declared. See
# _insert_elements.
_CONSTRAINT_RE = re.compile(
    r"(?m)^[ \t]*(?:autoconstraints|autoconstrain|constraint|constrain)\b" + _STALE_TAIL, re.DOTALL)
_CONSTRAINT_ELEMENTS = frozenset({"autoconstraints", "autoconstrain", "constraint", "constrain"})
# Status-line fade-out: hold the message legible, then fade it away over the rest of the budget (see _set_status).
_STATUS_HOLD_MS, _STATUS_FADE_MS, _STATUS_FADE_STEPS = 2500, 2500, 25


def _structure_signature(script: str) -> tuple:
    """A fingerprint of the parts of `script` the structure pane cares about, so staleness is flagged when — and only when — one of them 
    changes in the main editor. Broader than the rigid-body pane's preview signature: it also covers rename/merge/copy, which change the 
    body list this pane shows even when they leave the drawn geometry untouched."""
    return tuple(m.group(0) for m in _STALE_RE.finditer(script))


def _synth_load_block(pdb_path: str, splits: str) -> str:
    inner = [f"    pdb {quote_script_value(pdb_path)}"]
    if splits.strip():
        inner.append(f"    split {splits.strip()}")
    return "load {\n" + "\n".join(inner) + "\n}"


def _norm_splits(value: str) -> str:
    """Normalise a splits string to a canonical space-separated form, so equal splits written with different spacing/commas compare equal 
    (and don't churn the load block)."""
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


def _is_residue_id(token: str) -> bool:
    """Whether `token` is a residue sequence id. Ids may legitimately be negative, so a leading minus is allowed."""
    return token.lstrip("-").isdigit() and token not in ("-", "")


def _parse_split(element: str) -> tuple[str, list[int]] | None:
    """(target body, cut residues) of a staged `split` element, or None for any other element."""
    tokens = element.split()
    if len(tokens) < 3 or tokens[0] != "split" or not all(_is_residue_id(t) for t in tokens[2:]):
        return None
    return tokens[1], [int(t) for t in tokens[2:]]


def _insert_elements(base: str, elements: list[str]) -> str:
    """Return `base` with staged setup elements inserted among the existing setup declarations.

    The staged list itself remains in the order in which the user applied its elements, and existing declarations are left untouched. Two
    anchors are used, since the two ends of the setup block have opposite requirements:

            * structural elements go immediately *before* the first constraint declaration. Constraints are indexed by the structure that exists
                when they are declared, so later structural changes can be rejected by the backend.
            * staged constraint elements go *after* the last existing setup declaration, so they remain after any existing structural changes.

    With no constraints in the base script the two anchors coincide, and the staged block is appended whole.
    """
    if not elements:
        return base
    match = _LOAD_BLOCK_RE.search(base)
    if match is None:  # no load block to anchor to: prepend
        return "".join(f"{e}\n" for e in elements) + base

    setup_end = match.end()
    existing_setup = list(_SETUP_RE.finditer(base, match.end()))
    if existing_setup:
        setup_end = existing_setup[-1].end()
    constraint = _CONSTRAINT_RE.search(base, match.end())
    body_set_at = constraint.start() if constraint else setup_end

    def insert(text: str, at: int, staged: list[str]) -> str:
        if not staged:
            return text
        block = "".join(f"{e}\n" for e in staged)
        sep = "" if not text[:at] or text[:at].endswith("\n") else "\n"
        return text[:at] + sep + block + text[at:]

    if body_set_at == setup_end:  # the anchors coincide, so the staged block stays whole and in the order it was applied
        return insert(base, setup_end, elements)

    structural, constraints = [], []
    for e in elements:
        (constraints if e.split()[:1] and e.split()[0] in _CONSTRAINT_ELEMENTS else structural).append(e)
    # the later insertion first, so the earlier offset is still valid when it is applied
    return insert(insert(base, setup_end, constraints), body_set_at, structural)


class StructurePane(ttk.Frame):
    """Interactive structure-inspection and body-management pane for a single PDB."""

    def __init__(
        self, parent, pdb_path: str, *,
        splits: str = "",
        base_script: Optional[Callable[[], str]] = None,
        on_apply_script: Optional[Callable[[str], None]] = None,
        base_signature: Optional[Callable[[str], object]] = None,
        relaxed_loads=None,
    ):
        super().__init__(parent)
        self.pdb_path = pdb_path
        self._splits = splits
        # relaxed-load grants, shared with every other pane by default, so a structure the user already accepted with relaxed settings
        # elsewhere doesn't start failing again here.
        from .load_recovery import RELAXED_LOADS
        self._relaxed_loads = relaxed_loads if relaxed_loads is not None else RELAXED_LOADS
        self._base_script = base_script          # target script to diff/patch, or None
        self._on_apply_script = on_apply_script   # apply a confirmed new script, or None
        # reduce the base script to a structural fingerprint, so a later edit to the same body/split setup is detected as "stale". Defaults to
        # the structure-pane signature, which covers every element this pane reads or writes (rename/merge/split/symmetry/constraint/...).
        self._base_signature = base_signature or _structure_signature
        self._built_sig = None                    # base fingerprint the current view was built from
        self.title = "Structure: " + pdb_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        self._elements: list[str] = []            # setup elements the user has applied, in order
        # Cα positions each staged `split` element's target body covered when it was staged, keyed by that target's name. Fragments are
        # registered as brand-new bodies and inherit nothing from their parent, so the atoms themselves are the only link back to it; this
        # map is what lets a later split of a fragment be folded into the element that produced it (see _apply_body_split). Positions, not
        # residue ids: a chain-split structure repeats its ids across chains, so ids would match a sibling chain's split element too.
        self._split_coverage: dict[str, frozenset[tuple[float, ...]]] = {}
        self._data: dict | None = None            # last good preview-structure dict
        self._names: list[str] = []               # body names aligned to body indices
        self._bodies: list[dict] = []             # per-body summary rows (index/name/atoms/res/copies)
        self._replica_info: dict[tuple[int, int], dict] = {}  # (body, copy) -> {"type", "name"}
        # the view selection isolated in the plot, as a set of (body, copy) selectors: copy None selects the whole body (all its symmetry copies), 
        # an int selects just that one replica. Empty = nothing selected. Plain click replaces the whole set with just the clicked row. Ctrl-click 
        # toggles that row's membership without touching the rest (matching the OS convention for adding single items to a selection). Shift-click 
        # selects every row between the last clicked row (the "anchor", see _select_anchor) and the one just clicked, in list order — the OS 
        # convention for range-selecting. Together these let several bodies be selected at once (e.g. to merge them) without typing every name out.
        self._highlighted: set[tuple[int, int | None]] = set()
        # residues click-selected in the preview, highlighted with their own colour, as (body, residue id) pairs — an id alone does not
        # identify a residue, since a chain-split structure repeats the same numbering in every chain. Like the Bodies-list selection
        # above (but for residues, not bodies), it stands in for an empty residue field: hitting Apply / Add with no
        # residues typed splits at the selected residues instead. Consumed (and cleared) when a split is applied.
        self._selected_residues: set[tuple[int, int]] = set()
        self._select_anchor: tuple[int, int | None] | None = None  # last plain/ctrl-clicked row, the shift-click range's fixed end
        self._ready_checks: list[tuple[ttk.Button, PlaceholderEntry, Callable]] = []  # action buttons that light up green
        self._expanded_bodies: set[int] = set()   # body indices whose replica children are unfolded
        self._row_frames: list[tk.Widget] = []
        self._rows: list[tuple] = []              # ((body, copy) selector, recolourable row widgets)
        self._body_row_frames: dict[int, tk.Frame] = {}       # body index -> its row (anchor to pack replicas after)
        self._body_chevrons: dict[int, tk.Label] = {}         # body index -> its fold chevron, flipped in place
        self._replica_row_frames: dict[int, list[tk.Frame]] = {}  # body index -> its currently-built replica rows
        self._redraw_job: Optional[str] = None    # pending after_idle handle for a deferred _redraw()
        self._status_fade_job: Optional[str] = None  # pending after handle for the status line's fade-out

        self._show_atoms = tk.BooleanVar(value=False)
        self._show_copies = tk.BooleanVar(value=True)
        self._show_backbone = tk.BooleanVar(value=True)
        self._show_constraints = tk.BooleanVar(value=True)
        self._colour_by = tk.StringVar(value=COLOUR_BY_LABELS[0])

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
        if self._status_fade_job is not None:
            self.after_cancel(self._status_fade_job)
            self._status_fade_job = None
        super().destroy()

    # ----- layout -------------------------------------------------------------
    def _build_plot(self, paned):
        frame = tk.Frame(paned, background=PALETTE["surface"])
        paned.add(frame, weight=1)
        self._fig = Figure(facecolor=PALETTE["surface"])
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_axis_off()
        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._toolbar = NavigationToolbar2Tk(self._canvas, frame, pack_toolbar=False)
        self._toolbar.configure(background=PALETTE["surface"])
        for child in self._toolbar.winfo_children():
            try:
                child.configure(background=PALETTE["surface"])
            except tk.TclError:
                pass
        self._toolbar.update()
        self._toolbar.pack(side="bottom", fill="x")
        # a thin readout above the canvas: shows the residue under the cursor and that a click toggles a split there
        self._hover_label = tk.Label(frame, text="", background=PALETTE["surface"], foreground=PALETTE["muted"],
                                     anchor="w", font=FONTS["base"])
        self._hover_label.pack(side="top", fill="x", padx=6)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # A "clear selection" chip floating in the plot's top-right corner, placed over the canvas rather than parked in the control column:
        # it belongs next to what it clears (the lit bodies and the pink residue spheres). Shown only while something is selected, see
        # _refresh_clear_chip. As a real Tk widget it also swallows the clicks that land on it, so pressing it can't pick a residue underneath.
        self._clear_chip = tk.Label(
            self._canvas.get_tk_widget(), text="✕  Clear selection", background=PALETTE["surface_alt"],
            foreground=PALETTE["muted"], font=FONTS["small"], padx=8, pady=4, cursor="hand2",
            borderwidth=1, relief="solid")
        self._clear_chip.bind("<Button-1>", lambda _e: self._clear_all_selection())
        self._clear_chip.bind("<Enter>", lambda _e: self._clear_chip.configure(background=PALETTE["accent_soft"], foreground=PALETTE["text"]))
        self._clear_chip.bind("<Leave>", lambda _e: self._clear_chip.configure(background=PALETTE["surface_alt"], foreground=PALETTE["muted"]))

        # clickable preview: hover to read the nearest residue, click to toggle a split there. A left press is
        # recorded and only treated as a click if the cursor barely moved by release (otherwise it was a rotation).
        self._press_xy = None
        self._canvas.mpl_connect("motion_notify_event", self._on_preview_hover)
        self._canvas.mpl_connect("button_press_event", self._on_preview_press)
        self._canvas.mpl_connect("button_release_event", self._on_preview_release)

    def _build_controls(self, parent):
        # The applied-elements list, status line, and Send button stay pinned to the bottom so they're visible no matter which sections 
        # are open; packed first with side="bottom" so the collapsible sections above can grow and shrink freely without displacing them.
        if self._on_apply_script is not None:
            ttk.Button(parent, text="Send to script…", style="Accent.TButton",
                       command=self._send_to_script).pack(side="bottom", fill="x", pady=(8, 0))
        self._status = ttk.Label(parent, text="", style="Muted.TLabel", wraplength=290, justify="left")
        self._status.pack(side="bottom", fill="x", pady=(6, 0))

        # An amber action bar styled like a section header, but it triggers a refresh instead of expanding. It is packed above the sections 
        # only while the view is stale (see _set_stale).
        self._build_refresh_bar(parent)

        # --- collapsible control sections; each opens and closes independently, so the body list can stay open for reference (e.g. body names) 
        # while another section is in use
        self._sections: list[CollapsibleSection] = []

        # --- display toggles ---
        display = self._section(parent, "Display", expanded=False)
        # "Backbone trace" hangs off "Atomic detail" behind a fold chevron: hiding the trace only makes sense once the atom cloud is
        # there to replace it, and nesting it shows that dependency instead of leaving it to be discovered.
        self._atom_chevron = self._display_row(display.body, "Atomic detail", self._show_atoms, chevron=True)
        self._atom_suboptions = ttk.Frame(display.body)
        self._display_row(self._atom_suboptions, "Backbone trace", self._show_backbone)
        self._atom_chevron.bind("<Button-1>", lambda _e: self._toggle_atom_suboptions())
        for text, var in (("Symmetry copies", self._show_copies), ("Constraints", self._show_constraints)):
            self._display_row(display.body, text, var)

        colour_row = ttk.Frame(display.body)
        colour_row.pack(fill="x", pady=(4, 0))
        tk.Label(colour_row, text="", background=PALETTE["bg"], font=FONTS["base"], width=2).pack(side="left")
        ttk.Label(colour_row, text="Colour by", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        colour_box = ttk.Combobox(colour_row, textvariable=self._colour_by, values=COLOUR_BY_LABELS, state="readonly", width=14)
        colour_box.pack(side="left", fill="x", expand=True)
        colour_box.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        # --- body list (scrolls when long, so the section keeps a bounded height), with a splits editor above it so the structure can be 
        # re-split here without leaving the pane ---
        bodies = self._section(parent, "Bodies", expanded=True)
        splits_row = ttk.Frame(bodies.body)
        splits_row.pack(fill="x", pady=(0, 6))
        ttk.Label(splits_row, text="Split at residues", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w")
        self._splits_var = tk.StringVar(value=self._splits)
        self._splits_entry = ttk.Entry(splits_row, textvariable=self._splits_var)
        self._splits_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(splits_row, text="Apply", style="Icon.TButton", command=self._apply_splits).grid(row=1, column=1)
        splits_row.columnconfigure(0, weight=1)
        self._splits_entry.bind("<Return>", lambda _e: self._apply_splits())

        # A chevron reveals a second, distinct kind of split. The row above re-partitions the freshly-read PDB (a load-block directive, so it 
        # re-reads the file and defines the whole body set). The `split` element revealed here instead partitions a body already in the setup 
        # — e.g. one produced by convert_to_symmetry — tying its fragments together in a shared symmetry, and is staged like every other element 
        # (it appears in "Applied elements"). Collapsed by default, since re-splitting the whole structure is the common case.
        self._body_split_frame = ttk.Frame(splits_row)
        self._body_split_entry = self._action_row(
            self._body_split_frame, "Additional splits (an existing body)", "body residues…",
            self._apply_body_split, button="Add",
            ready_check=lambda entry: bool(self._selected_residues) or len(entry.get().split()) >= 2,
        )

        self._split_chevron = ttk.Label(splits_row, text="▸", style="Muted.TLabel", cursor="hand2")
        self._split_chevron.grid(row=1, column=2, padx=(4, 0))
        self._split_chevron.bind("<Button-1>", lambda _e: self._toggle_body_split())
        self._body_list = ScrollableFrame(bodies.body, max_height=220)
        self._body_list.pack(fill="both", expand=True)

        # --- merge / delete ---
        actions = self._section(parent, "Manage bodies", expanded=False)
        # Rename's "old" half can come from the current selection instead of being typed: with exactly one row (body or replica) selected, 
        # typing just the new name (one token) is enough. Two typed tokens are always taken literally (old new), regardless of any selection 
        # — typed content wins over the selection.
        self._rename_entry = self._action_row(
            actions.body, "Rename", "old new", self._apply_rename, button="Rename",
            ready_check=lambda entry: len(entry.get().split()) == 2
                or (len(entry.get().split()) == 1 and self._selected_single_name() is not None),
        )
        # Merge/Delete take only body identifiers, so a selection in the Bodies list can stand in for a typed one: with the field empty,
        # clicking Apply falls back to the current whole-body selection if it satisfies the arity (>=2 for merge, >=1 for delete). The
        # button lights up green whenever that would work, so the possibility is discoverable without ever silently prefilling the field.
        self._merge_entry = self._action_row(
            actions.body, "Merge", "first others...", self._apply_merge,
            ready_check=lambda entry: bool(entry.get()) or len(self._selected_body_names()) >= 2,
        )
        self._delete_entry = self._action_row(
            actions.body, "Delete", "body", self._apply_delete,
            ready_check=lambda entry: bool(entry.get()) or len(self._selected_body_names()) >= 1,
        )

        # --- symmetry: two distinct operations, the more common one (adding a symmetry to a single body) on top, decomposing one or more
        # bodies into a shared, fitted symmetry below (a single body is split into copies itself; several are treated as ready-made copies)
        sym = self._section(parent, "Symmetry", expanded=False).body
        self._sym_add_entry = self._action_row(
            sym, "Add symmetry to a body", "body type", self._apply_add_symmetry, button="Apply"
        )
        self._sym_convert_entry = self._action_row(
            sym, "Decompose bodies into a symmetry", "body(s)... type", self._apply_convert_symmetry, button="Convert",
            advanced="tolerance (default 2.0 Å)"
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

        # --- staged elements: kept as its own accordion section, always last, so it can be collapsed like any other section when the user 
        # needs more room for e.g. Bodies + Constraints at once
        self._applied = self._section(parent, "Applied elements", expanded=True).body
        self._rebuild_applied_list()  # seed the initial "no changes" placeholder

    def _toggle_body_split(self, *, expand: bool | None = None):
        """Fold the per-body `split` row in or out. `expand` forces a direction instead of toggling, so it can be opened on demand without
        closing it again on the next call."""
        showing = self._body_split_frame.winfo_ismapped()
        expand = not showing if expand is None else expand
        if expand == showing:
            return
        if expand:
            self._body_split_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        else:
            self._body_split_frame.grid_forget()
        self._split_chevron.configure(text="▾" if expand else "▸")

    def _split_would_override(self) -> bool:
        """Whether applying the load-block splits field would override work already done, rather than being the plain first split it looks
        like: a split is already in force (including `chain`, which typing residues silently replaces), or staged elements name bodies that
        a re-split renumbers. In both cases the per-body `split` element is what the user actually wants."""
        return bool(self._elements) or bool(self._splits.strip())

    def _refresh_split_field(self):
        """Dim the load-block splits field once applying it would override an existing split. It stays fully usable — this only stops it
        reading as the obvious next step, since the per-body split below is."""
        self._splits_entry.configure(foreground=PALETTE["muted"] if self._split_would_override() else PALETTE["text"])

    def _display_row(self, parent, text: str, var, *, chevron: bool = False) -> tk.Label:
        """A Display-section checkbutton, prefixed by a fold chevron or, without one, a blank of the same width so all labels line up."""
        row = ttk.Frame(parent)
        row.pack(fill="x")
        glyph = tk.Label(row, text="▸" if chevron else "", background=PALETTE["bg"], foreground=PALETTE["muted"],
                         font=FONTS["base"], width=2, cursor="hand2" if chevron else "")
        glyph.pack(side="left")
        ttk.Checkbutton(row, text=text, variable=var, command=self._redraw).pack(side="left")
        return glyph

    def _toggle_atom_suboptions(self):
        """Fold the sub-options hanging off "Atomic detail" in or out, leaving the toggles themselves untouched."""
        if self._atom_suboptions.winfo_manager():
            self._atom_suboptions.pack_forget()
            self._atom_chevron.configure(text="▸")
        else:
            self._atom_suboptions.pack(fill="x", padx=(_SUBOPTION_INDENT, 0), after=self._atom_chevron.master)
            self._atom_chevron.configure(text="▾")

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

    def _action_row(self, parent, label, hint, command, button="Apply", ready_check=None,
                     advanced: str | None = None) -> PlaceholderEntry:
        """An action row: a short label, a text entry whose greyed placeholder carries the format hint (so no separate hint line is needed),
        and a button. Returns the entry so the caller can read it with .get() and reset it with .clear().

        `ready_check(entry) -> bool`, if given, is re-evaluated on every keystroke and on every Bodies-list selection change
        (see _refresh_action_readiness); the button turns solid green while it returns True, so the user can tell at a glance that clicking
        it will actually do something.

        `advanced`, if given, is the placeholder text for an optional second field (e.g. a tolerance override) tucked behind a small
        chevron after the button, collapsed by default so it stays out of the way for the common case. It's reachable afterwards as
        the returned entry's `.advanced` attribute (a PlaceholderEntry, or None if `advanced` wasn't given)."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text=label, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = PlaceholderEntry(row, hint)
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        btn = ttk.Button(row, text=button, style="Icon.TButton", command=command)
        btn.grid(row=1, column=1)
        row.columnconfigure(0, weight=1)
        entry.bind("<Return>", lambda _e: command())
        if ready_check is not None:
            self._ready_checks.append((btn, entry, ready_check))
            entry.bind("<KeyRelease>", lambda _e: self._refresh_action_readiness(), add="+")

        entry.advanced = None
        if advanced is not None:
            adv_entry = PlaceholderEntry(row, advanced)
            adv_entry.bind("<Return>", lambda _e: command())
            entry.advanced = adv_entry

            def toggle_advanced(_e=None):
                if adv_entry.winfo_ismapped():
                    adv_entry.grid_forget()
                    chevron.configure(text="▸")
                else:
                    adv_entry.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(20, 0), pady=(4, 0))
                    chevron.configure(text="▾")

            chevron = ttk.Label(row, text="▸", style="Muted.TLabel", cursor="hand2")
            chevron.grid(row=1, column=2, padx=(4, 0))
            chevron.bind("<Button-1>", toggle_advanced)
        return entry

    def _refresh_action_readiness(self):
        """Recolour every action button registered via _action_row's ready_check: green while its check
        passes (there's either typed text or a selection it can use instead), grey otherwise."""
        for btn, entry, check in self._ready_checks:
            btn.configure(style="Ready.TButton" if check(entry) else "Icon.TButton")

    # ----- backend rebuild ----------------------------------------------------
    def _compose(self, elements: list[str]) -> str:
        base = self._base_script() if self._base_script else _synth_load_block(self.pdb_path, self._splits)
        if _LOAD_BLOCK_RE.search(base) is None:  # target has no load block: fall back to a synthetic one
            base = _synth_load_block(self.pdb_path, self._splits)
        else:  # honour the pane's (possibly edited) splits over whatever the base load block carries
            base = _with_split(base, self._splits)
        return _insert_elements(base, elements)

    def _rebuild(self, elements: list[str], *, rebuild_widgets: bool = True) -> tuple[bool, str]:
        """Load the composed setup script through the backend and refresh the view/body list from the bodies that remain.
        Returns (ok, message); on failure nothing is mutated.

        rebuild_widgets=False skips the Tk body-list teardown/rebuild (self._bodies etc. are still refreshed from the
        backend) — for a rename, which changes nothing else, the caller patches the one affected label in place instead."""
        script = self._compose(elements)
        try:
            from ..wrapper.Rigidbody import Rigidbody
            with self._relaxed_loads.applied(self.pdb_path):
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
        before = self._body_structure()
        self._compute_bodies()
        # Bodies are addressed by index, and a merge/delete/split/convert renumbers them: the indices a selection holds would silently
        # re-point at unrelated bodies (and residues) once the operation goes through. So both selections survive only a rebuild that
        # leaves the body/copy structure exactly as it was — e.g. a rename, which changes nothing but a name.
        if self._body_structure() != before:
            self._highlighted, self._select_anchor = set(), None
            self._selected_residues = set()
        if rebuild_widgets:
            self._rebuild_body_list()
        self._refresh_action_readiness()
        self._refresh_split_field()
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
        """Flag the view as stale when the base script has changed since it was built (e.g. the user edited the load block in the main editor).
        If the structure has never loaded successfully, retry instead of just flagging: there is no built view or staged edits to protect, so
        a load-time setting relaxed elsewhere (see load_recovery.RelaxedLoads) — which changes nothing _base_sig() can see — gets a chance to
        take effect. Called when the pane is switched back to."""
        if self._built_sig is None:
            ok, msg = self._rebuild(self._elements)
            if not ok:
                self._set_status(f"Could not load the structure: {msg}", ok=False)
                self._redraw()
            return
        self._set_stale(self._base_sig() != self._built_sig)

    def _set_stale(self, stale: bool):
        if not hasattr(self, "_refresh_bar") or not self._sections:
            return
        if stale and not self._refresh_bar.winfo_ismapped():
            self._refresh_bar.pack(fill="x", pady=(0, 6), before=self._sections[0])
        elif not stale and self._refresh_bar.winfo_ismapped():
            self._refresh_bar.pack_forget()

    def _do_refresh(self):
        """Reload the view from the current script, discarding all staged edits 
        (they are the only thing lost, so confirm only when there are any)."""
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

    def _body_structure(self) -> list[tuple[int, tuple[int, ...]]]:
        """The body/copy layout every selector is addressed against, as (body index, copy indices). Compared across a rebuild to tell a
        structural change (merge/delete/split/…) from one that only relabels — see _rebuild."""
        return [(b["index"], tuple(b["copies"])) for b in self._bodies]

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
        """Full teardown and rebuild of every row. Used when the underlying body data itself changes (a backend rebuild
        that adds, removes, or reorders bodies/copies) — a plain rename patches its one label in place instead, see
        _start_rename, since nothing else about the body list changes."""
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
        self._body_list.refresh()  # the row count just changed; re-fit the viewport and scroll position

    def _build_body_row(self, b: dict):
        replicas = b["copies"][1:]
        row = tk.Frame(self._body_list.body, background=PALETTE["surface"], cursor="hand2")
        row.pack(fill="x")

        # a fold chevron for bodies that have replicas, or a matching-width spacer for those that don't so the swatches line up. 
        # The chevron toggles the fold without changing the highlight.
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

        for w in (row, swatch, label, meta):  # click anywhere on the row highlights the whole body; shift-click adds/removes it from the selection
            w.bind("<Button-1>", lambda e, i=b["index"]: self._toggle_highlight(i, None, shift=self._is_shift(e), ctrl=self._is_ctrl(e)))
        for w in (row, label):  # double-click the name (or the row) to rename the body
            w.bind("<Double-Button-1>", lambda _e, i=b["index"], lbl=label: self._start_rename(lbl, self._name_of(i, None), i, None))
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
            w.bind("<Button-1>", lambda e, i=b["index"], c=copy: self._toggle_highlight(i, c, shift=self._is_shift(e), ctrl=self._is_ctrl(e)))
        for w in (row, label):  # double-click the name (or the row) to rename the replica, same as a base body
            w.bind("<Double-Button-1>", lambda _e, i=b["index"], c=copy, lbl=label: self._start_rename(lbl, self._name_of(i, c), i, c))
        self._row_frames.append(row)
        self._rows.append(((b["index"], copy), tuple(widgets)))
        return row

    def _refresh_row_highlight(self):
        """Recolour the rows to reflect the highlighted bodies/replicas, in place — so a click doesn't
        tear down the row widgets (which would make double-click-to-rename impossible)."""
        for selector, widgets in self._rows:
            bg = PALETTE["accent_soft"] if selector in self._highlighted else PALETTE["surface"]
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
        self._body_list.refresh()  # folding changes the row count, so the viewport has to follow

    @staticmethod
    def _is_shift(event) -> bool:
        return bool(event.state & 0x1)

    @staticmethod
    def _is_ctrl(event) -> bool:
        return bool(event.state & 0x4)

    def _row_order(self) -> list[tuple[int, int | None]]:
        """Selectors in the order their rows currently appear in the Bodies list (base bodies with
        any expanded replicas folded in beneath them), needed to resolve a shift-click range."""
        return [selector for selector, _widgets in self._rows]

    def _range_between(self, anchor: tuple[int, int | None], selector: tuple[int, int | None]) -> set:
        """Every selector between `anchor` and `selector` in current list order, inclusive. Falls back to just `selector` 
        if either endpoint isn't currently shown (e.g. its body was folded away since the anchor was set)."""
        order = self._row_order()
        if anchor not in order or selector not in order:
            return {selector}
        i, j = order.index(anchor), order.index(selector)
        lo, hi = min(i, j), max(i, j)
        return set(order[lo:hi + 1])

    def _toggle_highlight(self, body: int, copy: int | None, *, shift: bool = False, ctrl: bool = False):
        """Plain click: select just this row, or deselect it if it was the only one already selected — and become the anchor for a future 
        shift-click range. Ctrl-click: add/remove this row from the current selection without touching the rest (also becomes the new anchor), 
        so several bodies can be picked one at a time (e.g. to merge them — see _selected_body_names). Shift-click: select every row between 
        the anchor and this one, replacing the current selection, without moving the anchor."""
        selector = (body, copy)
        if shift and self._select_anchor is not None:
            self._highlighted = self._range_between(self._select_anchor, selector)
        elif ctrl:
            if selector in self._highlighted:
                self._highlighted.discard(selector)
            else:
                self._highlighted.add(selector)
            self._select_anchor = selector
        else:
            self._highlighted = set() if self._highlighted == {selector} else {selector}
            self._select_anchor = selector
        # isolating a specific replica is pointless while copies are hidden, so reveal them
        if copy not in (None, 0) and selector in self._highlighted and not self._show_copies.get():
            self._show_copies.set(True)
        self._refresh_row_highlight()
        self._refresh_action_readiness()
        self._schedule_redraw()

    def _selected_body_names(self) -> list[str]:
        """Names of the whole bodies (copy=None) currently selected in the Bodies list, ordered by body index. Replica-only selections 
        don't count — merge/delete operate on whole bodies."""
        names_by_index = {b["index"]: b["name"] for b in self._bodies}
        return [names_by_index[i] for i, c in sorted(self._highlighted) if c is None and i in names_by_index]

    def _known_names(self) -> set[str]:
        """Every body/replica display name currently known, used to tell whether a field's first token is already a body identifier the 
        user typed themselves."""
        names = {b["name"] for b in self._bodies}
        names |= {info["name"] for info in self._replica_info.values()}
        return names

    def _with_selected_bodies(self, entry: PlaceholderEntry, *, exact: int | None = None, minimum: int = 1) -> list[str]:
        """Tokens for an action whose field is a leading run of body identifiers followed by other arguments (a symmetry type, a constraint 
        type/distance, ...). If the field's first token is already a recognized body/replica name, it's taken literally — the user typed the 
        bodies themselves, so the selection is left out of it entirely. Otherwise, if the Bodies-list selection satisfies the requirement 
        (exactly `exact` whole bodies if given, else at least `minimum`), the selected bodies' names are prepended to whatever's already 
        typed — so e.g. selecting two bodies and typing just "cm" is enough for a constraint. Falls back to the typed tokens as-is when 
        neither applies, so the existing per-action validation reports the real problem (e.g. "needs a body and a type") rather than 
        silently doing nothing."""
        tokens = entry.get().split()
        if tokens and tokens[0] in self._known_names():
            return tokens
        selected = self._selected_body_names()
        satisfied = len(selected) == exact if exact is not None else len(selected) >= minimum
        if not satisfied:
            return tokens
        return selected + tokens

    def _name_of(self, body: int, copy: int | None) -> str | None:
        """Current display name of a body or replica, read fresh from state rather than a value captured at row-build time —
        matters after an in-place rename (see _start_rename), which patches this state without rebuilding the row widgets."""
        if copy is None:
            return next((b["name"] for b in self._bodies if b["index"] == body), None)
        info = self._replica_info.get((body, copy))
        return info["name"] if info else None

    def _selected_single_name(self) -> str | None:
        """Display name of the one currently selected row — a whole body or a single replica — or None unless exactly one is selected. Used
        to let Rename infer "old" from a click instead of typing it, which (unlike merge/delete) makes sense for a replica selection too."""
        if len(self._highlighted) != 1:
            return None
        body, copy = next(iter(self._highlighted))
        return self._name_of(body, copy)

    def _start_rename(self, label: tk.Label, old: str, body: int, copy: int | None):
        """Replace a body or replica name label with an inline entry so the user can rename it in place. Committing applies a 
        `rename <old> <new>` element; the backend keeps the default name too, so a rename can always be undone by renaming back. Works 
        the same for a base body's name and a replica's addressable name (e.g. "b1s1r1"), since both are just names the backend accepts."""
        # Tk only fires <Button-1> for a double-click's first (leading) press, not its second — the second press fires <Double-Button-1> instead. 
        # So the leading click already toggled this row's highlight via <Button-1>; re-invoke the same toggle here (called only once per double-click) 
        # to flip it right back, cancelling that highlight out exactly as if the row had never been clicked, before opening the rename editor.
        self._toggle_highlight(body, copy)
        # match the label's own font (replica labels use a smaller font than base bodies), and take the label's spot in the row's left-to-right 
        # pack order so a sibling packed after it (e.g. the replica's type badge) doesn't visually jump to its left while the entry is showing
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

        def restore_label():
            # put the label back in its old spot instead of rebuilding the whole body list, which
            # would otherwise visibly redraw every row just to close one inline entry
            if before is not None:
                label.pack(side="left", before=before)
            else:
                label.pack(side="left")

        def finish(apply: bool):
            if state["done"]:
                return
            state["done"] = True
            toplevel.unbind("<Button-1>", click_id)
            new = var.get().strip()
            entry.destroy()
            if not apply or not new or new == old:
                restore_label()
                return
            if any(c.isspace() for c in new):
                self._set_status("A name cannot contain spaces.", ok=False)
                restore_label()
                return
            # a rename changes nothing else about the structure (no body/copy is added, removed, or reshuffled), so the backend call still 
            # validates and persists it, but the Tk body list doesn't need a full teardown/rebuild — just this one label's text needs to change.
            if self._apply_element(f"rename {old} {new}", rebuild_widgets=False):
                label.config(text=self._name_of(body, copy))
            restore_label()

        def click_outside(event):
            # most row widgets (labels/frames) never take keyboard focus, so clicking them doesn't fire <FocusOut> on the entry
            #  — commit/close explicitly on any click that isn't on the entry itself, so clicking anywhere else in the GUI 
            # (not just the plot or another entry) closes the editor as expected
            if event.widget is not entry:
                finish(True)

        entry.bind("<Return>", lambda _e: finish(True))
        entry.bind("<FocusOut>", lambda _e: finish(True))
        entry.bind("<Escape>", lambda _e: finish(False))
        # bound on the toplevel (not just this row) so a click anywhere in the window is caught, added after the entry is focused so it 
        # doesn't fire for the double-click that opened it
        toplevel = self.winfo_toplevel()
        click_id = toplevel.bind("<Button-1>", click_outside, add="+")

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
                ax, self._data, self._split_residues(),
                show_atoms=self._show_atoms.get(),
                show_copies=self._show_copies.get(),
                show_backbone=self._show_backbone.get(),
                show_constraints=self._show_constraints.get(),
                highlight=self._highlighted,
                color_by=COLOUR_BY_MODES[self._colour_by.get()],
                body_names={b["index"]: b["name"] for b in self._bodies},
                selected_residues=self._selected_residues,
            )
            if lims is not None and self._preserve_view:
                ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        if self._preserve_view:  # keep the camera angle across redraws, not just the zoom
            self._set_orientation(view)
        self._preserve_view = True
        self._fig.set_layout_engine("tight")
        self._canvas.draw_idle()
        self._refresh_clear_chip()  # the chip tracks what the plot shows, so it follows the same path as the drawing itself

    _preserve_view = False

    def _refresh_clear_chip(self):
        """Show the floating "clear selection" chip while either selection the plot reflects is non-empty, and hide it again once neither is."""
        show = bool(self._selected_residues or self._highlighted)
        if show and not self._clear_chip.winfo_ismapped():
            self._clear_chip.place(relx=1.0, y=8, x=-8, anchor="ne")
        elif not show and self._clear_chip.winfo_ismapped():
            self._clear_chip.place_forget()

    def _clear_all_selection(self):
        """Drop both selections in one go: the click-selected residues and the Bodies-list highlight. They are separate selections, but the
        chip sits in the plot, where they read as one — everything lit up is dropped."""
        self._selected_residues = set()
        self._highlighted, self._select_anchor = set(), None
        self._refresh_row_highlight()
        self._refresh_action_readiness()
        self._schedule_redraw()
        self._set_status("Selection cleared.", ok=True)

    # ----- clickable preview ---------------------------------------------------
    def _on_preview_hover(self, event):
        """Show the residue under the cursor in the hover readout (or clear it over empty space)."""
        hit = nearest_ca_residue(self._ax, self._data, event)
        if hit is None:
            if self._hover_label["text"]:
                self._hover_label.configure(text="")
            return
        name = next((b["name"] for b in self._bodies if b["index"] == hit["body"]), f"b{hit['body'] + 1}")
        verb = "deselect" if (hit["body"], hit["residue"]) in self._selected_residues else "select"
        self._hover_label.configure(text=f"{name} · residue {hit['residue']} · click to {verb}")

    def _on_preview_press(self, event):
        """Record a left-button press so _on_preview_release can tell a click from a view rotation."""
        self._press_xy = (event.x, event.y) if event.button == 1 else None

    def _on_preview_release(self, event):
        """A left release that barely moved from the press is a click: toggle the nearest residue in the selection. A release after a real 
        drag (a view rotation) or while a toolbar pan/zoom mode is active is ignored."""
        press_xy, self._press_xy = self._press_xy, None
        if (event.button != 1 or press_xy is None or event.x is None or event.y is None
                or getattr(self._toolbar, "mode", "")):
            return
        if abs(event.x - press_xy[0]) + abs(event.y - press_xy[1]) > plotting.PICK_CLICK_DRAG_TOLERANCE:
            return
        hit = nearest_ca_residue(self._ax, self._data, event)
        if hit is not None:
            self._toggle_selected_residue(hit["body"], hit["residue"])

    def _toggle_selected_residue(self, body: int, resid: int):
        """Add the residue to the click-selection, or remove it if already selected. It is keyed by the body it was picked in as well as its
        id, so clicking residue 100 of one chain marks that one residue rather than every chain's residue 100. Nothing is applied — the
        selection just stands in for an empty residue field when a split is applied (see _apply_splits / _apply_body_split), mirroring how a
        Bodies-list selection stands in for an empty Merge/Delete field."""
        pick = (body, resid)
        self._selected_residues.discard(pick) if pick in self._selected_residues else self._selected_residues.add(pick)
        self._schedule_redraw()
        self._refresh_action_readiness()  # the click-selection can stand in for a typed field, so it changes what is clickable
        # once the load-block split would override rather than establish the partition, the per-body split is the only thing a click can
        # usefully feed, so reveal it rather than leaving it behind a chevron the user has no reason to suspect
        if self._selected_residues and self._split_would_override():
            self._toggle_body_split(expand=True)
        n = len(self._selected_residues)
        if not n:
            self._set_status("Selection cleared.", ok=True)
            return
        # Add targets one body, so it is only offered while the selection resolves to one
        target = self._split_target_name()
        hint = f", or Add to split {target} there" if target else ""
        self._set_status(f"{n} residue{'' if n == 1 else 's'} selected — Apply to split the whole structure there{hint}.", ok=True)

    def _selected_bodies(self) -> set[int]:
        """Body indices the click-selected residues lie in."""
        return {body for body, _resid in self._selected_residues}

    def _selection_body_name(self) -> str | None:
        """Name of the body every click-selected residue lies in, or None when there is no selection or it spans several bodies."""
        bodies = self._selected_bodies()
        return self._name_of(next(iter(bodies)), None) if len(bodies) == 1 else None

    def _split_target_name(self) -> str | None:
        """Body an Add would split the click-selection at: the original target of a staged split that already covers the selection (whose
        fragments count as one body's business), otherwise the single live body the residues lie in."""
        staged = self._staged_split_for(self._selected_residues)
        return staged[1] if staged is not None else self._selection_body_name()

    def _clear_selection(self):
        """Drop the click-selection and repaint (called once a split has consumed it)."""
        if self._selected_residues:
            self._selected_residues = set()
            self._schedule_redraw()
            self._refresh_action_readiness()

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

    def _split_residues(self) -> list[int]:
        """Residue ids to mark as split points in the view: the load-block split (self._splits) plus every `split` element — both the ones staged in 
        this pane and any already in the base script. Non-numeric tokens (a body name, the keyword itself) are ignored; draw_structure deduplicates."""
        ids = self._parse_splits()
        sources = list(self._elements)
        if self._base_script:
            sources.append(self._base_script())
        for text in sources:
            for m in re.finditer(r"(?m)^[ \t]*split\b[^\n]*", text):
                ids += [int(t) for t in re.split(r"[,\s]+", m.group(0).strip()) if t.isdigit()]
        return ids

    # ----- setup actions ------------------------------------------------------
    def _apply_element(self, element: str, *, rebuild_widgets: bool = True) -> bool:
        candidate = self._elements + [element]
        ok, msg = self._rebuild(candidate, rebuild_widgets=rebuild_widgets)
        if ok:
            self._elements = candidate
            self._rebuild_applied_list()
            self._set_status(f"Applied: {element}", ok=True)
        else:
            self._set_status(f"Rejected “{element}”: {msg}", ok=False)
        return ok

    def _replace_element(self, i: int, element: str) -> bool:
        """Swap the staged element at `i` for `element`, keeping its position in the order. Used to extend an existing `split` rather than
        stage a second one against a fragment the backend would refuse to split."""
        candidate = self._elements[:i] + [element] + self._elements[i + 1:]
        ok, msg = self._rebuild(candidate)
        if ok:
            self._elements = candidate
            self._rebuild_applied_list()
            self._set_status(f"Extended: {element}", ok=True)
        else:
            self._set_status(f"Rejected “{element}”: {msg}", ok=False)
        return ok

    def _apply_splits(self):
        """Re-split the structure at the residue numbers in the splits field and rebuild the view. The split lives in the load block, so it is
        applied by recomposing (see _with_split); on a bad value the field is reverted so it always mirrors the splits actually in force.

        With the field left empty, the click-selection (see _toggle_selected_residue) stands in for it — click residues in
        the preview and hit Apply, no typing — matching how a Bodies-list selection stands in for an empty Merge/Delete field."""
        typed = self._splits_var.get().strip()
        from_selection = not typed and bool(self._selected_residues)
        # the load-block split cuts the file as a whole, so only the ids matter here; two chains' residue 100 collapse to one cut
        new = " ".join(str(r) for r in sorted({r for _b, r in self._selected_residues})) if from_selection else typed
        if _norm_splits(new) == _norm_splits(self._splits):
            self._splits_var.set(new)
            if from_selection:
                self._clear_selection()  # already the splits in force, but consume the selection so it doesn't linger
            return
        prev = self._splits
        self._splits = new
        ok, msg = self._rebuild(self._elements)
        if ok:
            self._splits_var.set(new)
            if from_selection:
                self._clear_selection()
            self._set_status(f"Re-split at {new}." if new else "Removed all splits.", ok=True)
        else:
            self._splits = prev
            self._splits_var.set(prev)
            self._set_status(f"Could not re-split: {msg}", ok=False)

    def _apply_body_split(self):
        """Split an existing body into fragments at the given residue ids: `split <body> <residues…>`. Distinct from the load-block split
        above (which re-partitions the freshly-read PDB and defines the whole body set): this is a staged setup element that partitions a
        body already in the setup — e.g. one produced by convert_to_symmetry — so its fragments stay tied together in a shared symmetry.
        The body may come from a single Bodies-list selection or, failing that, from the click-selected residues themselves — each one
        belongs to a body, so clicking a few residues along one body and hitting Add, with nothing typed and nothing else selected, is
        enough.

        Residues already inside a staged split's target extend that element instead of staging a second one. The backend refuses to split
        a fragment that carries a symmetry shared with its siblings, and one element is enough regardless: BodySplitter cuts at a *set* of
        residue ids taken in atom order, so cutting the original body at the union of both sets yields exactly the same fragments. This
        also lets a selection spanning two fragments of one original body work, since they are one split element's business."""
        tokens = self._with_selected_bodies(self._body_split_entry, exact=1)
        if tokens and tokens[0] in self._known_names():
            named, typed_residues = tokens[0], tokens[1:]
        else:  # nothing names a body; whatever was typed is then all residues, and the body follows from them
            named, typed_residues = None, tokens
        from_selection = not typed_residues and bool(self._selected_residues)
        if from_selection:
            picks = set(self._selected_residues)
        else:
            if not typed_residues:
                self._set_status("Splitting a body needs residue ids — type them or click residues in the preview.", ok=False)
                return
            if not all(_is_residue_id(r) for r in typed_residues):
                self._set_status("Split residue ids must be integers, e.g. b1 100 200.", ok=False)
                return
            # typed ids are only residues of the body they are typed against, so pair them with it; with no body to pair them with there is
            # nothing to match or split, and the "needs a body" branch below reports it
            index = self._index_of(named or self._selection_body_name() or "")
            picks = {(index, int(r)) for r in typed_residues} if index is not None else set()
        cuts = {r for _b, r in picks}

        staged = self._staged_split_for(picks)
        if staged is not None:
            i, target, existing = staged
            merged = sorted(existing | cuts)
            if len(merged) == len(existing):
                self._set_status(f"Already splitting {target} at {' '.join(str(c) for c in sorted(cuts))}.", ok=False)
                return
            ok = self._replace_element(i, "split " + target + " " + " ".join(str(c) for c in merged))
        else:
            body = named or self._selection_body_name()
            if body is None:
                self._set_status(
                    "The selected residues span several bodies — split one body at a time." if len(self._selected_bodies()) > 1 else
                    "Splitting a body needs a body — type one, select it in the Bodies list, or click residues on it.", ok=False)
                return
            coverage = self._coverage_of_body(body)  # read before the rebuild replaces the body with its fragments
            ok = self._apply_element("split " + body + " " + " ".join(str(c) for c in sorted(cuts)))
            if ok:
                self._split_coverage[body] = coverage
        if ok:
            self._body_split_entry.clear()
            if from_selection:
                self._clear_selection()

    def _staged_split_for(self, picks: set[tuple[int, int]]) -> tuple[int, str, frozenset[int]] | None:
        """(index, target body, existing cuts) of the staged `split` element whose target body contained every one of the (body, residue)
        `picks`, or None when they fall outside every staged split. Targets are whole bodies, so at most one element can ever match."""
        positions = self._ca_positions(residue_ca_mask(self._data, picks))
        if not positions:
            return None
        for i, element in enumerate(self._elements):
            parsed = _parse_split(element)
            if parsed is not None and positions <= self._split_coverage.get(parsed[0], frozenset()):
                return i, parsed[0], frozenset(parsed[1])
        return None

    def _index_of(self, name: str) -> int | None:
        """Index of the body with this display name, or None if no body has it (a replica name included: replicas aren't bodies)."""
        return next((b["index"] for b in self._bodies if b["name"] == name), None)

    def _ca_positions(self, mask) -> frozenset[tuple[float, ...]]:
        """The masked atoms' positions, rounded to a stable key. Positions identify residues where ids and body indices cannot: ids repeat
        across chains, and indices are renumbered by every merge/delete/split — the atoms themselves are only ever re-partitioned."""
        if self._data is None:
            return frozenset()
        return frozenset(tuple(round(float(v), 3) for v in p) for p in self._data["coords"][mask])

    def _coverage_of_body(self, name: str) -> frozenset[tuple[float, ...]]:
        """Every Cα position in the named body, as it stands right now — the coverage a `split` element staged against it takes over."""
        index = self._index_of(name)
        if index is None or self._data is None:
            return frozenset()
        d = self._data
        return self._ca_positions(d["is_ca"] & (d["copy"] == 0) & (d["body"] == index))

    def _apply_rename(self):
        """Rename a body: `rename <old> <new>`, the same element the inline double-click-to-rename (see _start_rename) applies. Either typed 
        as two tokens (old new), or — with exactly one row selected in the Bodies list — as just the new name, taking "old" from that selection."""
        tokens = self._rename_entry.get().split()
        if len(tokens) == 2:
            old, new = tokens
        elif len(tokens) == 1 and self._selected_single_name() is not None:
            old, new = self._selected_single_name(), tokens[0]
        else:
            self._set_status("Rename needs the current name and the new name, e.g. b1 core.", ok=False)
            return
        if old == new:
            self._set_status("The new name is the same as the current one.", ok=False)
            return
        if self._apply_element(f"rename {old} {new}"):
            self._rename_entry.clear()
        self._refresh_action_readiness()

    def _apply_merge(self):
        # an empty field falls back to the Bodies-list selection, so a shift-clicked group of bodies can be merged without typing every name 
        # out (see _selected_body_names)
        tokens = self._merge_entry.get().split() or self._selected_body_names()
        if len(tokens) < 2:
            self._set_status("Merge needs a target body and at least one other.", ok=False)
            return
        if self._apply_element("merge " + " ".join(tokens)):
            self._merge_entry.clear()
        self._refresh_action_readiness()

    def _apply_delete(self):
        tokens = self._delete_entry.get().split() or self._selected_body_names()
        if not tokens:
            self._set_status("Delete needs at least one body.", ok=False)
            return
        if self._apply_element("delete " + " ".join(tokens)):
            self._delete_entry.clear()
        self._refresh_action_readiness()

    def _apply_add_symmetry(self):
        """Add a symmetry to a single body: `symmetry <body> <type>` (e.g. b1 c4). A lone type is allowed for a single-body system (the backend 
        infers the body), or — with exactly one whole body selected in the Bodies list — for any system, taking the body from that selection."""
        tokens = self._with_selected_bodies(self._sym_add_entry, exact=1)
        if not tokens:
            self._set_status("Add symmetry needs a body and a type, e.g. b1 c4.", ok=False)
            return
        if len(tokens) > 2:
            self._set_status("Add symmetry takes one body and one type, e.g. b1 c4.", ok=False)
            return
        if self._apply_element("symmetry " + " ".join(tokens)):
            self._sym_add_entry.clear()

    def _apply_convert_symmetry(self):
        """Decompose one or more bodies into one shared symmetry, collapsing the copies into the first body plus a fitted symmetry:
        `convert_to_symmetry { type <type> bodies <b…> }`. A single body is fitted directly — the backend splits it into the
        symmetry's own copy count itself, rather than needing several bodies handed to it as ready-made copies. Typing just the type
        is enough when one or more whole bodies are selected in the Bodies list, or — with only one body in the whole system —
        with nothing selected at all, since decomposing it is then the only sensible target. An optional tolerance (Å) typed into
        the field tucked behind the chevron overrides the backend's default when the assembly's residual RMSD to the fitted
        symmetry is just barely out of range."""
        tokens = self._with_selected_bodies(self._sym_convert_entry, minimum=1)
        if len(tokens) == 1 and len(self._bodies) == 1:
            tokens = [self._bodies[0]["name"]] + tokens
        if len(tokens) < 2:
            self._set_status("Decomposing needs at least one body and a type, e.g. b1 c2.", ok=False)
            return
        *bodies, sym = tokens
        tolerance = self._sym_convert_entry.advanced.get().strip()
        extra = ""
        if tolerance:
            try:
                float(tolerance)
            except ValueError:
                self._set_status("Tolerance must be a number, e.g. 3.5.", ok=False)
                return
            extra = "\n    tolerance " + tolerance
        element = "convert_to_symmetry {\n    type " + sym + "\n    bodies " + " ".join(bodies) + extra + "\n}"
        if self._apply_element(element):
            self._sym_convert_entry.clear()
            self._sym_convert_entry.advanced.clear()

    def _apply_autoconstrain(self):
        """Auto-generate a set of constraints: `autoconstrain <backbone|none>`. Defaults to backbone, the usual choice; `none` clears 
        any auto-generated set."""
        choice = self._autoconstrain_entry.get().strip() or "backbone"
        if self._apply_element(f"autoconstrain {choice}"):
            self._autoconstrain_entry.clear()

    def _apply_add_constraint(self):
        """Add a distance constraint between two bodies: `<body1> <body2> <type> [distance]`. `bond` and `cm` need only the two bodies;
        `attract` and `repel` also take a target distance (e.g. b1 b2 attract 30). Built as a `constrain { … }` block for the backend.
        Typing just the type (and distance, if needed) is enough when exactly two whole bodies are selected in the Bodies list."""
        tokens = self._with_selected_bodies(self._constraint_entry, exact=2)
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
        if self._apply_element("constrain {\n" + "\n".join(lines) + "\n}"):
            self._constraint_entry.clear()

    def _prune_split_coverage(self):
        """Drop residue coverage for bodies no staged `split` targets any more, e.g. after one is removed from the applied list."""
        targets = {parsed[0] for e in self._elements if (parsed := _parse_split(e)) is not None}
        self._split_coverage = {name: r for name, r in self._split_coverage.items() if name in targets}

    def _rebuild_applied_list(self):
        self._prune_split_coverage()  # called after every change to _elements, so coverage is pruned in step with it
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
            # the delete button is packed first at side=right so it always keeps its full size; the summary then fills whatever width is left 
            # and is ellipsized to it, so a wide element (e.g. a constrain block collapsed to one line) can never push the button out of reach
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
        """Show a transient status message. It holds at full strength, fades into the background, and then clears itself: the status line is
        pinned to the bottom of the control column, so a long message wraps to several lines and squeezes the sections above it out of reach
        until the window is resized."""
        if self._status_fade_job is not None:
            self.after_cancel(self._status_fade_job)
            self._status_fade_job = None
        colour = PALETTE["ok_border"] if ok else PALETTE["danger"]
        self._status.configure(text=text, foreground=colour)
        if text:
            self._status_fade_job = self.after(_STATUS_HOLD_MS, lambda: self._fade_status(colour, 1))

    def _fade_status(self, colour: str, step: int):
        if step > _STATUS_FADE_STEPS:  # fully faded; drop the text so the line collapses back to its empty height
            self._status_fade_job = None
            self._status.configure(text="")
            return
        self._status.configure(foreground=blend(colour, PALETTE["bg"], step / _STATUS_FADE_STEPS))
        self._status_fade_job = self.after(_STATUS_FADE_MS // _STATUS_FADE_STEPS, lambda: self._fade_status(colour, step + 1))

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
        """Write the composed script back to the editor, then drop the staged edits: they now live in the base, so keeping them staged would 
        double-apply them on the next rebuild."""
        self._on_apply_script(new)
        self._elements = []
        self._rebuild_applied_list()
        self._rebuild(self._elements)  # rebuild from the new base, re-recording its fingerprint


class ScriptDiffDialog(tk.Toplevel):
    """A modal side-by-side diff: the original script on the left with removed lines tinted red, the new script on the right with added lines 
    tinted green. Confirm applies the change."""

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
        ttk.Button(buttons, text="Apply changes", style="Accent.TButton", command=self._confirm).pack(side="right", padx=(0, 8))

        self.geometry("820x520")
        self.grab_set()

    def _make_text(self, parent, heading) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=heading, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        text = tk.Text(
            frame, wrap="none", font=FONTS["mono"], relief="flat", borderwidth=0,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            padx=8, pady=6, height=24, width=48
        )
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

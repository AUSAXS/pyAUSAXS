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
disappear from the list (and the plot) exactly as it will at run time. That model lives in
core.structure; this pane draws it and forwards the user's input to it.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .core import plotting
from .core.plotting import draw_structure, nearest_ca_residue
from .core.structure import StructureModel, side_by_side_diff
from .theme import FONTS, PALETTE, blend
from .widgets import (
    CollapsibleSection, PlaceholderEntry, ScrollableFrame, ellipsize_label,
)

# display label -> draw_structure `color_by` mode; the first entry is the default
COLOUR_BY_MODES = {"Body": "body", "Symmetry copy": "copy", "Residue": "residue"}
COLOUR_BY_LABELS = list(COLOUR_BY_MODES)

_SUBOPTION_INDENT = 16  # px a folded-out display sub-option is inset from its parent toggle

# Status-line fade-out: hold the message legible, then fade it away over the rest of the budget (see _set_status).
_STATUS_HOLD_MS, _STATUS_FADE_MS, _STATUS_FADE_STEPS = 2500, 2500, 25


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
        # relaxed-load grants, shared with every other pane by default, so a structure the user already accepted with relaxed settings
        # elsewhere doesn't start failing again here.
        from .load_recovery import RELAXED_LOADS
        # the bodies, staged elements and selections; this pane only shows them and forwards the user's actions
        self.model = StructureModel(
            pdb_path, splits=splits, base_script=base_script, base_signature=base_signature,
            relaxed_loads=relaxed_loads if relaxed_loads is not None else RELAXED_LOADS,
        )
        self._on_apply_script = on_apply_script   # apply a confirmed new script, or None
        self.title = "Structure: " + pdb_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        self._ready_checks: list[tuple[ttk.Button, PlaceholderEntry, Callable]] = []  # action buttons that light up green
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

        ok, msg = self._rebuild()  # initial draw of the untouched structure
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
        self._splits_var = tk.StringVar(value=self.model.splits)
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
            ready_check=lambda entry: self.model.can_split_body(entry.get()),
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
            ready_check=lambda entry: self.model.can_rename(entry.get()),
        )
        # Merge/Delete take only body identifiers, so a selection in the Bodies list can stand in for a typed one: with the field empty,
        # clicking Apply falls back to the current whole-body selection if it satisfies the arity (>=2 for merge, >=1 for delete). The
        # button lights up green whenever that would work, so the possibility is discoverable without ever silently prefilling the field.
        self._merge_entry = self._action_row(
            actions.body, "Merge", "first others...", self._apply_merge,
            ready_check=lambda entry: self.model.can_merge(entry.get()),
        )
        self._delete_entry = self._action_row(
            actions.body, "Delete", "body", self._apply_delete,
            ready_check=lambda entry: self.model.can_delete(entry.get()),
        )

        # --- symmetry: two distinct operations, the more common one (adding a symmetry to a single body) on top, decomposing one or more
        # bodies into a shared, fitted symmetry below (a single body is split into copies itself; several are treated as ready-made copies)
        sym = self._section(parent, "Symmetry", expanded=False).body
        self._sym_add_entry = self._action_row(
            sym, "Add symmetry to a body", "body type", self._apply_add_symmetry, button="Apply"
        )
        self._sym_convert_entry = self._action_row(
            sym, "Decompose bodies into a symmetry", "body(s)... type", self._apply_convert_symmetry, button="Convert",
            advanced="tolerance (default 5.0 Å)"
        )

        # --- constraints: auto-generate a set (backbone) on top, then add an individual constraint between two bodies below. Existing constraints
        # are edited by removing/re-adding via the applied-elements list, and shown in the view via the "Constraints" display toggle.
        # `backbone` is prefilled since it is the only generator currently available: the row is then a one-click affair.
        con = self._section(parent, "Constraints", expanded=False).body
        self._autoconstrain_entry = self._action_row(
            con, "Auto-generate constraints", "backbone", self._apply_autoconstrain, button="Generate", prefill="backbone"
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

    def _refresh_split_field(self):
        """Dim the load-block splits field once applying it would override an existing split. It stays fully usable — this only stops it
        reading as the obvious next step, since the per-body split below is."""
        self._splits_entry.configure(foreground=PALETTE["muted"] if self.model.split_would_override() else PALETTE["text"])

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
                     advanced: str | None = None, prefill: str = "") -> PlaceholderEntry:
        """An action row: a short label, a text entry whose greyed placeholder carries the format hint (so no separate hint line is needed),
        and a button. Returns the entry so the caller can read it with .get() and reset it with .clear().

        `ready_check(entry) -> bool`, if given, is re-evaluated on every keystroke and on every Bodies-list selection change
        (see _refresh_action_readiness); the button turns solid green while it returns True, so the user can tell at a glance that clicking
        it will actually do something.

        `advanced`, if given, is the placeholder text for an optional second field (e.g. a tolerance override) tucked behind a small
        chevron after the button, collapsed by default so it stays out of the way for the common case. It's reachable afterwards as
        the returned entry's `.advanced` attribute (a PlaceholderEntry, or None if `advanced` wasn't given).

        `prefill`, if given, seeds the entry with a real value rather than a hint — for a row whose one sensible answer can just be
        submitted by clicking the button. It is restored whenever the entry is cleared."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text=label, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = PlaceholderEntry(row, hint, initial=prefill)
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
    def _rebuild(self) -> tuple[bool, str]:
        """Rebuild the model from its current base and staged elements, and refresh the view on success."""
        ok, msg = self.model.rebuild()
        if ok:
            self._on_rebuilt()
        return ok, msg

    def _on_rebuilt(self, *, rebuild_widgets: bool = True):
        """Refresh the view from a freshly rebuilt model. rebuild_widgets=False skips the Tk body-list teardown/rebuild — for a rename, which
        changes nothing else, the caller patches the one affected label in place instead."""
        if rebuild_widgets:
            self._rebuild_body_list()
        self._refresh_action_readiness()
        self._refresh_split_field()
        self._schedule_redraw()
        # the view now matches this base, so it is no longer stale
        self._set_stale(False)

    def _show_result(self, result, *, rebuild_widgets: bool = True):
        """Reflect an action's outcome: refresh from the model if it was rebuilt, and show its status message."""
        if result.rebuilt:
            self._on_rebuilt(rebuild_widgets=rebuild_widgets)
            self._rebuild_applied_list()
        if result.message:
            self._set_status(result.message, ok=result.ok)

    # ----- staleness / refresh ------------------------------------------------
    def check_stale(self):
        """Flag the view as stale when the base script has changed since it was built (e.g. the user edited the load block in the main editor).
        If the structure has never loaded successfully, retry instead of just flagging: there is no built view or staged edits to protect, so
        a load-time setting relaxed elsewhere (see core.loads.RelaxedLoads) — which changes nothing the base signature can see — gets a chance
        to take effect. Called when the pane is switched back to."""
        if not self.model.built:
            ok, msg = self._rebuild()
            if not ok:
                self._set_status(f"Could not load the structure: {msg}", ok=False)
                self._redraw()
            return
        self._set_stale(self.model.is_stale())

    def _set_stale(self, stale: bool):
        if not hasattr(self, "_refresh_bar") or not self._sections:
            return
        if stale and not self._refresh_bar.winfo_manager():
            self._refresh_bar.pack(fill="x", pady=(0, 6), before=self._sections[0])
        elif not stale and self._refresh_bar.winfo_manager():
            self._refresh_bar.pack_forget()

    def _do_refresh(self):
        """Reload the view from the current script, discarding all staged edits
        (they are the only thing lost, so confirm only when there are any)."""
        if self.model.elements:
            n = len(self.model.elements)
            if not messagebox.askyesno(
                "Refresh from script",
                f"Discard {n} staged change{'' if n == 1 else 's'} and reload from the current script?",
                parent=self):
                return
        ok, msg = self.model.refresh_from_base()
        self._rebuild_applied_list()
        self._splits_var.set(self.model.splits)  # mirror the (possibly re-edited) script's splits again
        if ok:
            self._on_rebuilt()
        self._set_status("Reloaded from the current script." if ok
                         else f"Could not reload from the script: {msg}", ok=ok)

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
        for b in self.model.bodies:
            self._build_body_row(b)
            # symmetry replicas (copy > 0), foldable beneath the base body they belong to
            if b["index"] in self.model.expanded_bodies:
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
                row, text="▾" if b["index"] in self.model.expanded_bodies else "▸",
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
            w.bind("<Double-Button-1>", lambda _e, i=b["index"], lbl=label: self._start_rename(lbl, self.model.name_of(i, None), i, None))
        self._row_frames.append(row)
        self._rows.append(((b["index"], None), (row, label, meta)))
        self._body_row_frames[b["index"]] = row

    def _build_replica_row(self, b: dict, copy: int, *, after: tk.Widget | None = None) -> tk.Frame:
        """A single symmetry-replica child row, indented under its base body. Clicking it isolates just that replica in the view. Name and
        type badge come straight from the backend's symmetry layout, keyed to this (body, copy) pair. Packed right after `after` (an
        existing row) when given, so a single body's replicas can be spliced into their exact spot in the list without disturbing any
        other row — see _toggle_body_expand."""
        info = self.model.replica_info[(b["index"], copy)]
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
            w.bind("<Double-Button-1>", lambda _e, i=b["index"], c=copy, lbl=label: self._start_rename(lbl, self.model.name_of(i, c), i, c))
        self._row_frames.append(row)
        self._rows.append(((b["index"], copy), tuple(widgets)))
        return row

    def _refresh_row_highlight(self):
        """Recolour the rows to reflect the highlighted bodies/replicas, in place — so a click doesn't
        tear down the row widgets (which would make double-click-to-rename impossible)."""
        for selector, widgets in self._rows:
            bg = PALETTE["accent_soft"] if selector in self.model.highlighted else PALETTE["surface"]
            for w in widgets:
                w.configure(background=bg)

    def _toggle_body_expand(self, body: int):
        """Fold/unfold a body's symmetry-replica rows in place: splice its replica rows in or out
        right after its own row, and flip its chevron — without touching any other row. """
        expanding = body not in self.model.expanded_bodies
        if expanding:
            self.model.expanded_bodies.add(body)
        else:
            self.model.expanded_bodies.discard(body)
        chevron = self._body_chevrons.get(body)
        if chevron is not None:
            chevron.configure(text="▾" if expanding else "▸")

        if expanding:
            b = next(bb for bb in self.model.bodies if bb["index"] == body)
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

    def _toggle_highlight(self, body: int, copy: int | None, *, shift: bool = False, ctrl: bool = False):
        """Select/deselect a row; see StructureModel.toggle_highlight for what plain, ctrl- and shift-clicks do."""
        # isolating a specific replica is pointless while copies are hidden, so reveal them
        if self.model.toggle_highlight(body, copy, shift=shift, ctrl=ctrl) and not self._show_copies.get():
            self._show_copies.set(True)
        self._refresh_row_highlight()
        self._refresh_action_readiness()
        self._schedule_redraw()

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
            new = var.get()
            entry.destroy()
            if not apply:
                restore_label()
                return
            # a rename changes nothing else about the structure (no body/copy is added, removed, or reshuffled), so the backend call still
            # validates and persists it, but the Tk body list doesn't need a full teardown/rebuild — just this one label's text needs to change.
            result = self.model.rename(old, new)
            self._show_result(result, rebuild_widgets=False)
            if result.rebuilt:
                label.config(text=self.model.name_of(body, copy))
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
        lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) if self.model.data is not None else None
        ax.clear()
        ax.set_axis_off()
        if self.model.data is None:
            ax.text2D(0.5, 0.5, "Could not read the structure", transform=ax.transAxes,
                      ha="center", va="center", color=PALETTE["muted"])
        else:
            draw_structure(
                ax, self.model.data, self.model.split_residues(),
                show_atoms=self._show_atoms.get(),
                show_copies=self._show_copies.get(),
                show_backbone=self._show_backbone.get(),
                show_constraints=self._show_constraints.get(),
                highlight=self.model.highlighted,
                color_by=COLOUR_BY_MODES[self._colour_by.get()],
                body_names={b["index"]: b["name"] for b in self.model.bodies},
                selected_residues=self.model.selected_residues,
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
        show = bool(self.model.selected_residues or self.model.highlighted)
        if show and not self._clear_chip.winfo_ismapped():
            self._clear_chip.place(relx=1.0, y=8, x=-8, anchor="ne")
        elif not show and self._clear_chip.winfo_ismapped():
            self._clear_chip.place_forget()

    def _clear_all_selection(self):
        """Drop both selections in one go: the click-selected residues and the Bodies-list highlight. They are separate selections, but the
        chip sits in the plot, where they read as one — everything lit up is dropped."""
        self.model.clear_all_selection()
        self._refresh_row_highlight()
        self._refresh_action_readiness()
        self._schedule_redraw()
        self._set_status("Selection cleared.", ok=True)

    # ----- clickable preview ---------------------------------------------------
    def _on_preview_hover(self, event):
        """Show the residue under the cursor in the hover readout (or clear it over empty space)."""
        hit = nearest_ca_residue(self._ax, self.model.data, event)
        if hit is None:
            if self._hover_label["text"]:
                self._hover_label.configure(text="")
            return
        name = next((b["name"] for b in self.model.bodies if b["index"] == hit["body"]), f"b{hit['body'] + 1}")
        verb = "deselect" if (hit["body"], hit["residue"]) in self.model.selected_residues else "select"
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
        hit = nearest_ca_residue(self._ax, self.model.data, event)
        if hit is not None:
            self._toggle_selected_residue(hit["body"], hit["residue"])

    def _toggle_selected_residue(self, body: int, resid: int):
        """Add the residue to the click-selection, or remove it if already selected (see StructureModel.toggle_selected_residue)."""
        message = self.model.toggle_selected_residue(body, resid)
        self._schedule_redraw()
        self._refresh_action_readiness()  # the click-selection can stand in for a typed field, so it changes what is clickable
        # once the load-block split would override rather than establish the partition, the per-body split is the only thing a click can
        # usefully feed, so reveal it rather than leaving it behind a chevron the user has no reason to suspect
        if self.model.selected_residues and self.model.split_would_override():
            self._toggle_body_split(expand=True)
        self._set_status(message, ok=True)

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

    # ----- setup actions ------------------------------------------------------
    def _apply_splits(self):
        """Re-split the structure at the residue numbers in the splits field (see StructureModel.apply_splits)."""
        result, shown = self.model.apply_splits(self._splits_var.get())
        self._splits_var.set(shown)
        if result.rebuilt:
            self._on_rebuilt()
        elif result.ok:  # nothing to re-split, but a consumed click-selection still has to disappear from the plot
            self._schedule_redraw()
            self._refresh_action_readiness()
        if result.message:
            self._set_status(result.message, ok=result.ok)

    def _apply_body_split(self):
        """Split an existing body into fragments (see StructureModel.apply_body_split)."""
        result = self.model.apply_body_split(self._body_split_entry.get())
        self._show_result(result)
        if result.ok:
            self._body_split_entry.clear()

    def _apply_rename(self):
        result = self.model.apply_rename(self._rename_entry.get())
        self._show_result(result)
        if result.ok:
            self._rename_entry.clear()
        self._refresh_action_readiness()

    def _apply_merge(self):
        result = self.model.apply_merge(self._merge_entry.get())
        self._show_result(result)
        if result.ok:
            self._merge_entry.clear()
        self._refresh_action_readiness()

    def _apply_delete(self):
        result = self.model.apply_delete(self._delete_entry.get())
        self._show_result(result)
        if result.ok:
            self._delete_entry.clear()
        self._refresh_action_readiness()

    def _apply_add_symmetry(self):
        result = self.model.apply_add_symmetry(self._sym_add_entry.get())
        self._show_result(result)
        if result.ok:
            self._sym_add_entry.clear()

    def _apply_convert_symmetry(self):
        result = self.model.apply_convert_symmetry(self._sym_convert_entry.get(), self._sym_convert_entry.advanced.get())
        self._show_result(result)
        if result.ok:
            self._sym_convert_entry.clear()
            self._sym_convert_entry.advanced.clear()

    def _apply_autoconstrain(self):
        result = self.model.apply_autoconstrain(self._autoconstrain_entry.get())
        self._show_result(result)
        if result.ok:
            self._autoconstrain_entry.clear()

    def _apply_add_constraint(self):
        result = self.model.apply_add_constraint(self._constraint_entry.get())
        self._show_result(result)
        if result.ok:
            self._constraint_entry.clear()

    def _rebuild_applied_list(self):
        for w in self._applied.winfo_children():
            w.destroy()
        if not self.model.elements:
            ttk.Label(self._applied, text="No changes staged yet.", style="Muted.TLabel").pack(anchor="w")
            return
        # bounded and scrollable, so a long list of edits never grows past the section's own space
        scroll = ScrollableFrame(self._applied, max_height=150)
        scroll.pack(fill="x")
        for i, element in enumerate(self.model.elements):
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
        self._show_result(self.model.remove_element(i))

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
        change = self.model.send_script()
        if change is None:  # neither the splits nor the staged elements actually differ from the base
            self._set_status("No changes to send.", ok=False)
            return
        old, new = change
        ScriptDiffDialog(self, old, new, on_confirm=lambda: self._confirm_send(new))

    def _confirm_send(self, new: str):
        """Write the composed script back to the editor, then drop the staged edits: they now live in the base, so keeping them staged would 
        double-apply them on the next rebuild."""
        self._on_apply_script(new)
        ok, _msg = self.model.sent()  # rebuilds from the new base, re-recording its fingerprint
        self._rebuild_applied_list()
        if ok:
            self._on_rebuilt()


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
        for row in side_by_side_diff(old, new):
            lt.insert("end", row.old + "\n", "del" if row.old_changed else ())
            rt.insert("end", row.new + "\n", "add" if row.new_changed else ())
        lt.configure(state="disabled")
        rt.configure(state="disabled")

    def _confirm(self):
        self._on_confirm()
        self.destroy()

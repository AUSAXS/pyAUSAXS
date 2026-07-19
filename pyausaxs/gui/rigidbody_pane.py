# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .data_pane import SaxsDataPane
from .panes import (
    SAXS_EXTENSIONS, STRUCTURE_EXTENSIONS, _make_validator, add_figure_tab,
    make_on_load_structure, make_on_load_saxs,
)
from .plotting import draw_structure, fit_figure_from_curves
from .runner import RigidbodyRunner
from .structure_pane import StructurePane
from .theme import FONTS, PALETTE
from .widgets import ConsolePane, FileField, RigidbodyHighlighter, Tooltip, enable_file_drop


DEFAULT_RIGIDBODY_SCRIPT = """\
output output/rigidbody/
load {
    pdb
    saxs
    split
}
autoconstrain backbone
save initial_state.pdb
save trajectory.xyz
parameter_generator {
    iterations 100
    translate 1
    rotate 1
}

print "Initial chi2: {chi2_no_penalty}"
loop
    optimize_once
        on_improvement
            print {
                msg "{iteration}/{iterations_total}: Accepted with new chi2 {chi2_no_penalty}"
                colour green
            }
            save trajectory.xyz
        end
    end
end
save final_state.pdb
"""

_LOAD_BLOCK_RE = re.compile(r"load\s*\{.*?\}", re.DOTALL)
# a 'symmetry' element: either a brace block (symmetry { ... }) or a single inline line
# (symmetry c6 / symmetry b1 c6), anchored to the first token on a line
_SYMMETRY_RE = re.compile(r"(?m)^[ \t]*symmetry\b\s*(?:\{.*?\}|[^\n]*)", re.DOTALL)
# a constraint element — autoconstrain/autoconstraints (inline) or constrain/constraint
# (inline or a brace block) — anchored to the first token on a line. The whole { ... } block
# is captured so edits to its arguments (type, bodies, …) refresh the preview too.
_CONSTRAINT_RE = re.compile(r"(?m)^[ \t]*(?:auto)?constrain(?:ts?)?\b\s*(?:\{.*?\}|[^\n]*)", re.DOTALL)
# an 'update' element (e.g. `update structure`) as the first token on a line; its presence makes
# the GUI poll the backend for the live structure during a run
_UPDATE_RE = re.compile(r"(?m)^[ \t]*update\b")
# the top-level 'output' directive (output <dir>), captured as (prefix, path) so the path can be
# rewritten to an absolute one at boot
_OUTPUT_RE = re.compile(r"(?m)^([ \t]*output[ \t]+)(\S+)")


class RigidbodyPane(ttk.Frame):
    """[Experimental] Rigid-body refinement driven by an AUSAXS sequencer script.

    Mirrors the SasView rigid-body refinement workflow: a script editor with helpers to fill in the load block from
    chosen files, plus Validate and Run actions that stream the backend's output into a log and plot the resulting fit."""

    title = "Rigidbody"

    def __init__(self, parent):
        super().__init__(parent)
        self.runner = RigidbodyRunner(self)
        # tell the backend a live-structure consumer exists, so `update` elements actually publish
        from ..wrapper.Rigidbody import Rigidbody
        Rigidbody.set_live_consumer(True)
        self._mode = "run"
        self._expanded = False
        self._fit_tabs: list = []        # result tabs added by a run (the structure tab persists)
        self._preview_job = None         # pending debounced preview redraw
        self._preview_key = None         # signature of what the preview currently shows
        self._preview_cache_key = None   # signature the preview structure was last built from
        self._preview_cache = None       # cached backend preview-structure dict, or None
        self._last_valid_lims = None     # axis limits from the last successful preview draw
        self._live_job = None            # pending live-structure poll during a run
        self._live_meta = None           # backbone mask (preview dict) for the active run, or None
        self._live_version = 0           # last live-structure version drawn
        self._script_cache_path = None   # where the script is autosaved/restored
        self._last_saved_script = None   # last text written, to skip unchanged autosaves
        self._script_file_path = None    # last file the user manually saved to / loaded from
        self._autosave_job = None        # pending periodic autosave
        self._data_pane = None           # SaxsDataPane tab for inspecting the SAXS data, or None
        self._structure_pane = None      # StructurePane tab for inspecting/managing bodies, or None

        # three panes: controls | script editor | results. The editor can expand over the results pane (and collapses again
        # when a refinement is launched).
        self.outer = ttk.Panedwindow(self, orient="horizontal")
        self.outer.pack(fill="both", expand=True, padx=6, pady=6)

        # --- controls pane (left) --------------------------------------------
        controls = ttk.Frame(self.outer, padding=(4, 4, 10, 4), width=340)
        controls.pack_propagate(False)
        self.outer.add(controls, weight=0)

        # the Input fields are a shortcut for editing the script's load block: each one writes only its own directive. The
        # script itself is always the authority.
        input_frame = ttk.Labelframe(controls, text="Input", padding=12)
        input_frame.pack(fill="x")

        self.structure_field = FileField(
            input_frame, "Structure",
            validator=_make_validator(STRUCTURE_EXTENSIONS, "_is_pdb_file"),
            on_valid=lambda _p: self._refresh_view_struct_btn(),
            on_commit=lambda p: self._on_load_structure(p),
            filetypes=[("Structure", "*.pdb *.ent *.cif *.xyz")],
        )
        self.saxs_field = FileField(
            input_frame, "SAXS data",
            validator=_make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file"),
            on_valid=lambda _p: self._refresh_view_btn(),
            on_commit=lambda p: self._on_load_saxs(p),
            filetypes=[("SAXS data", "*.dat *.rsr *.xvg")],
        )
        self.structure_field.pack(fill="x")
        self.saxs_field.pack(fill="x", pady=(6, 0))
        # actions that act on the inputs as a whole rather than a single field, grouped to
        # the right: hand the inputs to the SAXS fitter, or open a data-inspection tab
        button_row = ttk.Frame(input_frame)
        button_row.pack(fill="x", pady=(8, 0))
        self._view_btn = ttk.Button(button_row, text="View data", command=self._open_data_pane,
                                    state="disabled")
        self._view_btn.pack(side="right")
        ttk.Button(button_row, text="Send to SAXS fitter", command=self._send_to_saxs_fitter).pack(
            side="right", padx=(0, 8))
        self._on_load_structure = make_on_load_structure(self._set_load_directive, self.saxs_field)
        self._on_load_saxs = make_on_load_saxs(self._set_load_directive, self.structure_field)

        struct_row = ttk.Frame(input_frame)
        struct_row.pack(fill="x", pady=(6, 0))
        self._view_struct_btn = ttk.Button(
            struct_row, text="View / manage structure", command=self._open_structure_pane, state="disabled")
        self._view_struct_btn.pack(side="right")

        splits_row = ttk.Frame(input_frame)
        splits_row.pack(fill="x", pady=(6, 0))
        ttk.Label(splits_row, text="Splits", style="Muted.TLabel").pack(anchor="w")
        self.splits_var = tk.StringVar()
        ttk.Entry(splits_row, textvariable=self.splits_var).pack(fill="x")
        # the trace is attached at the end of __init__, once the preview exists

        run_frame = ttk.Frame(controls)
        run_frame.pack(fill="x", pady=12)
        self.validate_button = ttk.Button(run_frame, text="Validate", command=self._validate_clicked)
        self.validate_button.pack(side="left")
        self.run_button = ttk.Button(run_frame, text="Run refinement", style="Accent.TButton", command=self._run_clicked)
        self.run_button.pack(side="left", padx=(8, 0))
        self.progress = ttk.Progressbar(run_frame, mode="indeterminate")  # packed only while running

        console_frame = ttk.Labelframe(controls, text="Output", padding=(2, 4))
        console_frame.pack(fill="both", expand=True)
        self.console = ConsolePane(console_frame, height=7)
        self.console.pack(fill="both", expand=True, padx=2, pady=2)

        # --- script editor pane (middle), with the expand toggle on its right -
        editor_pane = ttk.Frame(self.outer)
        self.outer.add(editor_pane, weight=2)
        self.expand_button = ttk.Button(editor_pane, text=">", width=2, command=self._toggle_expand)
        self.expand_button.pack(side="right", fill="y", padx=(4, 0))

        editor_frame = ttk.Labelframe(editor_pane, padding=(2, 4))
        editor_frame.pack(side="left", fill="both", expand=True)
        # Replace the labelframe's plain text title with a custom row so a reset cross can sit at its right end, on the same line
        # as the "Refinement script" text.
        title_row = ttk.Frame(editor_frame)
        ttk.Label(title_row, text="Refinement script", style="Heading.TLabel").pack(side="left")
        # right-aligned icon actions. Packed right-to-left, so the visual order is load, save, clear,
        # with the destructive reset cross furthest right.
        self.reset_button = self._make_icon_button(
            title_row, "✕", self._reset_clicked, "Reset to the default script",
            color=PALETTE["danger"], hover=PALETTE["danger_hover"], bold=True)
        self.reset_button.pack(side="right", padx=(0, 2))
        self._make_icon_button(
            title_row, "↧", self._save_to_file_clicked, "Save script to a file…"
        ).pack(side="right", padx=(0, 8))
        self._make_icon_button(
            title_row, "↥", self._load_from_file_clicked, "Load script from a file…"
        ).pack(side="right", padx=(0, 8))
        editor_frame.configure(labelwidget=title_row)
        # Stretch the title row to the frame width (the labelframe won't do it) so the reset cross sits flush right.
        def _stretch_title_row():
            row_w, row_h = title_row.winfo_reqwidth(), title_row.winfo_reqheight()
            title_row.pack_propagate(False)
            title_row.configure(width=row_w, height=row_h)
            editor_frame.bind("<Configure>", lambda e: title_row.configure(width=max(e.width - 16, row_w)))
        self.after_idle(_stretch_title_row)
        self.editor = tk.Text(
            editor_frame, wrap="none", undo=True, font=FONTS["mono"], height=12,
            relief="flat", borderwidth=0, padx=8, pady=6,
            background=PALETTE["surface"], foreground=PALETTE["text"],
            insertbackground=PALETTE["text"], selectbackground=PALETTE["accent"],
        )
        # Show tabs as 4 spaces: compute pixel width of a space in the editor font
        try:
            _mono_font = tkfont.Font(font=FONTS["mono"])
            _space_px = _mono_font.measure(" ") or 8
            # configure tab stops to 4 * space width
            self.editor.configure(tabs=(_space_px * 4,))
        except Exception:
            # fall back silently if font metrics aren't available
            pass

        editor_scroll = ttk.Scrollbar(editor_frame, command=self.editor.yview)
        self.editor.configure(yscrollcommand=editor_scroll.set)
        editor_scroll.pack(side="right", fill="y")
        self.editor.pack(fill="both", expand=True, padx=2, pady=2)
        operations, keywords = self._fetch_vocabulary()
        self.highlighter = RigidbodyHighlighter(self.editor, operations, keywords)
        # restore the last session's script from the cache, falling back to the default
        self._script_cache_path = self._resolve_script_cache_path()
        cached_script = self._load_cached_script()
        self.editor.insert("1.0", cached_script or DEFAULT_RIGIDBODY_SCRIPT)
        if cached_script is not None:
            # the cached script may have been written from a different working directory; resolve
            # its relative file paths against that directory so they still point at the right files,
            # then mirror the load block into the Input fields once (later edits don't propagate back)
            self._absolutize_script_paths(self._last_launch_directory())
            self._populate_inputs_from_script()
        self._last_saved_script = self.editor.get("1.0", "end-1c")
        self.highlighter.highlight()
        # manual edits to the script (e.g. the pdb/split lines) refresh both the syntax highlighting and the structure preview; clicking
        # moves the cursor, which can change the highlighted scope pair
        self.editor.bind("<KeyRelease>", self._on_editor_changed)
        self.editor.bind("<ButtonRelease-1>", lambda _e: self.highlighter.highlight_brackets())
        # Ctrl-A selects all (Tk's default binds it to "start of line")
        self.editor.bind("<Control-a>", self._select_all)
        self.editor.bind("<Control-A>", self._select_all)
        # Tk's <<Paste>> on X11 does not delete the selection before inserting
        self.editor.bind("<<Paste>>", self._on_paste)

        # --- results pane (right), the larger pane by default ----------------
        self.results_pane = ttk.Frame(self.outer, padding=(10, 4, 4, 4))
        self.outer.add(self.results_pane, weight=3)
        self.results = ttk.Notebook(self.results_pane)
        self.results.pack(fill="both", expand=True)

        # a persistent "structure" tab: a 3D Cα backbone with the split residues in red,
        # kept across runs (the fit tabs are added alongside it)
        self.structure_tab = tk.Frame(self.results, background=PALETTE["surface"])
        struct_toolbar = tk.Frame(self.structure_tab, background=PALETTE["surface"])
        struct_toolbar.pack(side="top", fill="x", padx=4, pady=(2, 0))
        home_btn = self._make_icon_button(
            struct_toolbar, "⌂", self._home_preview, "Reset to default view"
        )
        home_btn.configure(font=(FONTS["base"][0], 18), padding=(10, 0))
        home_btn.pack(side="left", padx=2, pady=2)
        self._struct_fig = Figure(facecolor=PALETTE["surface"])
        self._struct_ax = self._struct_fig.add_subplot(111, projection="3d")
        self._struct_canvas = FigureCanvasTkAgg(self._struct_fig, master=self.structure_tab)
        self._struct_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.results.add(self.structure_tab, text="structure")
        self.results.bind("<<NotebookTabChanged>>", lambda _e: self._schedule_preview_update())

        # now that the preview exists, drive it live from the Splits field and draw it once
        self.splits_var.trace_add("write", lambda *_: self._set_load_directive("split", self.splits_var.get()))
        self.after(60, self._restore_split)
        self.after(80, self._update_structure_preview)
        self._autosave_job = self.after(self._AUTOSAVE_INTERVAL_MS, self._autosave_script)
        enable_file_drop(
            self, [self.structure_field, self.saxs_field],
            on_unmatched=self._on_drop_unmatched,
            on_leave_without_drop=self._on_drop_leave_without_drop,
        )

    # ----- data pane management -----------------------------------------------
    def _on_drop_unmatched(self, path: str):
        """Called when a dropped file doesn't validate against either input field."""
        self.console.append(f'Ignored dropped file (unrecognised type): "{path}"\n')

    def _on_drop_leave_without_drop(self):
        """Called when a drag leaves before any drop has ever succeeded in this process."""
        self.console.append("Drag-and-drop didn't register — please try dropping the file again.\n")

    def _refresh_view_btn(self):
        """Enable "View data" whenever the SAXS field is valid (driven by on_valid, so it
        also fires when the field is filled from a restored/loaded script)."""
        if hasattr(self, "_view_btn"):
            self._view_btn.configure(state="normal" if self.saxs_field.valid else "disabled")

    def _open_data_pane(self):
        """Open (or focus) a data-inspection tab for the current SAXS file. Rebuilt if the
        file has changed since it was opened; the rigid-body run is script-driven, so this
        is purely for inspecting the data."""
        path = self.saxs_field.get()
        if not path:
            return
        if self._data_pane is not None and self._data_pane.file_path != path:
            self._close_data_pane()
        if self._data_pane is None:
            notebook = self.master
            self._data_pane = SaxsDataPane(notebook, path)
            notebook.add(self._data_pane, text=self._data_pane.title)
        self.master.select(self._data_pane)

    def _close_data_pane(self):
        if self._data_pane is None:
            return
        try:
            self.master.forget(self._data_pane)
        except Exception:
            pass
        self._data_pane.destroy()
        self._data_pane = None

    # ----- structure pane management ------------------------------------------
    def _refresh_view_struct_btn(self):
        """Enable "View / manage structure" whenever the structure field is valid."""
        if hasattr(self, "_view_struct_btn"):
            self._view_struct_btn.configure(
                state="normal" if self.structure_field.valid else "disabled")

    def _open_structure_pane(self):
        """Open (or focus) the structure-management tab for the current PDB. It reads the live
        script as its base and writes confirmed body changes back into the editor."""
        path = self.structure_field.get() or (self._load_value("pdb") or "")
        if not path:
            return
        if self._structure_pane is not None and self._structure_pane.pdb_path != path:
            self._close_structure_pane()
        if self._structure_pane is None:
            notebook = self.master
            self._structure_pane = StructurePane(
                notebook, path,
                splits=self.splits_var.get(),
                base_script=lambda: self.editor.get("1.0", "end-1c"),
                on_apply_script=self._apply_structure_script,
            )
            notebook.add(self._structure_pane, text=self._structure_pane.title)
        self.master.select(self._structure_pane)

    def _close_structure_pane(self):
        if self._structure_pane is None:
            return
        try:
            self.master.forget(self._structure_pane)
        except Exception:
            pass
        self._structure_pane.destroy()
        self._structure_pane = None

    def _apply_structure_script(self, new_script: str):
        """Replace the editor's script with one carrying the structure pane's body changes, then
        refresh highlighting and the preview and switch focus back to this pane."""
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", new_script)
        self.highlighter.highlight()
        self._schedule_preview_update()
        self.master.select(self)

    def _send_to_saxs_fitter(self):
        """Populate the SAXS fitter pane with the current structure and SAXS fields and switch to it."""
        from .saxs_pane import SaxsFitterPane
        notebook = self.master
        for tab_id in notebook.tabs():
            pane = notebook.nametowidget(tab_id)
            if isinstance(pane, SaxsFitterPane):
                pane.structure_field.set(self.structure_field.get(), touched=True)
                pane.saxs_field.set(self.saxs_field.get(), touched=True)
                notebook.select(tab_id)
                break

    # ----- layout -------------------------------------------------------------
    # fraction of the space (right of the controls) given to the editor when the results pane is visible; the results pane keeps the rest.
    _EDITOR_FRACTION = 0.42

    def _restore_split(self):
        """Position the editor|results sash so the results pane is the larger one."""
        if self._expanded:
            return
        self.update_idletasks()
        total = self.outer.winfo_width()
        if total <= 1:
            self.after(50, self._restore_split)
            return
        left = self.outer.sashpos(0)  # controls|editor boundary
        self.outer.sashpos(1, left + int((total - left) * self._EDITOR_FRACTION))

    def _toggle_expand(self):
        self._collapse_editor() if self._expanded else self._expand_editor()

    def _expand_editor(self):
        """Hand the results pane's space over to the script editor."""
        if self._expanded:
            return
        self.outer.forget(self.results_pane)
        self.expand_button.configure(text="<")
        self._expanded = True

    def _collapse_editor(self):
        """Restore the results pane, shrinking the editor back to its normal size."""
        if not self._expanded:
            return
        self.outer.add(self.results_pane, weight=3)
        self.expand_button.configure(text=">")
        self._expanded = False
        self.after(10, self._restore_split)

    # ----- syntax highlighting ------------------------------------------------
    @staticmethod
    def _fetch_vocabulary() -> tuple[set, set]:
        """Ask the backend for the valid script elements, split into line operations (the dict keys) and argument
        keywords (values not themselves keys), mirroring the Qt setValidElements logic. Returns empty sets if the
        backend is unavailable, in which case the highlighter still colours scopes/comments but flags nothing."""
        try:
            from ..wrapper.Rigidbody import Rigidbody
            mapping = Rigidbody.get_valid_elements_and_arguments()
        except Exception:
            return set(), set()
        operations = set(mapping)
        keywords = {arg for args in mapping.values() for arg in args} - operations
        return operations, keywords

    def _on_editor_changed(self, _event=None):
        self.highlighter.highlight()
        self.highlighter.highlight_brackets()
        self._schedule_preview_update()

    def _select_all(self, _event=None):
        """Ctrl-A: select the whole script (Tk otherwise jumps to line start)."""
        self.editor.tag_add("sel", "1.0", "end-1c")
        self.editor.mark_set("insert", "1.0")
        self.editor.see("insert")
        return "break"

    def _on_paste(self, _event=None):
        """Ctrl-V: replace selection before inserting — Tk's <<Paste>> on X11 skips this."""
        try:
            text = self.editor.clipboard_get()
        except tk.TclError:
            return "break"
        try:
            self.editor.delete("sel.first", "sel.last")
        except tk.TclError:
            pass  # no selection active
        self.editor.insert("insert", text)
        self.editor.see("insert")
        self._on_editor_changed()
        return "break"

    def _reset_clicked(self):
        """Restore the default script after a confirmation, so an accidental click
        can't silently wipe a hand-written script."""
        if not messagebox.askyesno(
                "Reset script",
                "Discard the current script and restore the default?",
                parent=self):
            return
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", DEFAULT_RIGIDBODY_SCRIPT)
        self.highlighter.highlight()
        self._schedule_preview_update()
        self._save_script()  # persist immediately so the default survives a restart

    def _make_icon_button(self, parent, glyph, command, tooltip, color=None, hover=None, bold=False):
        """A clickable glyph styled like the reset cross: muted by default, brightening on hover,
        with a tooltip (icon-only buttons need a label somewhere)."""
        color = color or PALETTE["muted"]
        hover = hover or PALETTE["text"]
        font = (FONTS["base"][0], 12, "bold") if bold else (FONTS["base"][0], 12)
        button = ttk.Label(parent, text=glyph, cursor="hand2", foreground=color, font=font)
        button.bind("<Button-1>", lambda _e: command())
        button.bind("<Enter>", lambda _e: button.configure(foreground=hover))
        button.bind("<Leave>", lambda _e: button.configure(foreground=color))
        Tooltip(button, tooltip)
        return button

    # ----- load / save the script to a file (independent of the cache) --------
    def _save_to_file_clicked(self):
        """Save the current script to a user-chosen file. This is separate from the cache: the
        periodic cache autosave continues untouched, and the file we write here is never overwritten
        by it."""
        path = filedialog.asksaveasfilename(
            parent=self, title="Save refinement script", defaultextension=".conf",
            initialdir=os.path.dirname(self._script_file_path) if self._script_file_path else None,
            initialfile=os.path.basename(self._script_file_path) if self._script_file_path else "refinement.conf",
            filetypes=[("Script", "*.conf"), ("All files", "*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(self.editor.get("1.0", "end-1c"))
        except OSError as e:
            messagebox.showerror("Save failed", f"Could not save the script:\n{e}", parent=self)
            return
        self._script_file_path = path

    def _load_from_file_clicked(self):
        """Load a script from a user-chosen file into the editor (not from the cache), then mirror
        its load block into the Input fields so they aren't left stale-empty beside a working script.
        As elsewhere, this is a one-shot fill: later edits don't propagate back, and the script
        remains the authority."""
        path = filedialog.askopenfilename(
            parent=self, title="Load refinement script",
            initialdir=os.path.dirname(self._script_file_path) if self._script_file_path else None,
            filetypes=[("Script", "*.conf"), ("All files", "*")],
        )
        if not path:
            return
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError as e:
            messagebox.showerror("Load failed", f"Could not load the script:\n{e}", parent=self)
            return
        self._script_file_path = path
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self.highlighter.highlight()
        self._populate_inputs_from_script()
        self._schedule_preview_update()

    # ----- script persistence -------------------------------------------------
    _AUTOSAVE_INTERVAL_MS = 10_000

    @staticmethod
    def _resolve_script_cache_path() -> str:
        """Path the script is autosaved to: <AUSAXS cache>/gui_rigidbody_script.txt."""
        from ..architecture import get_cache_dir
        return str(get_cache_dir() / "gui_rigidbody_script.txt")

    def _load_cached_script(self):
        """Return the autosaved script if one exists and is non-empty, else None."""
        path = self._script_cache_path
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            return None
        return text if text.strip() else None

    def _save_script(self):
        """Write the current script to the cache, skipping unchanged content."""
        path = self._script_cache_path
        if not path:
            return
        text = self.editor.get("1.0", "end-1c")
        if text == self._last_saved_script:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
            self._last_saved_script = text
        except OSError:
            pass

    def _autosave_script(self):
        """Periodically persist the script (every ~10 s) and reschedule."""
        self._save_script()
        self._autosave_job = self.after(self._AUTOSAVE_INTERVAL_MS, self._autosave_script)

    # ----- boot: restore paths and Input fields -------------------------------
    @staticmethod
    def _last_launch_directory() -> str:
        """The directory the GUI was last launched from, used to resolve relative paths in the
        cached script. Falls back to the current directory when no config has been written yet."""
        from .session import load_config
        return load_config().get("last_launch_directory") or os.getcwd()

    def _absolutize_script_paths(self, base_dir: str):
        """Rewrite the script's relative file paths (the load block's pdb/saxs and the top-level
        output directive) into absolute paths anchored at `base_dir`. Absolute paths are left as-is."""
        def absolutize(path: str) -> str:
            path = path.strip()
            if not path or os.path.isabs(path):
                return path
            # preserve a trailing separator: the backend appends file stems to the output dir
            absolute = os.path.normpath(os.path.join(base_dir, path))
            return absolute + os.sep if path.endswith(("/", os.sep)) else absolute

        text = self.editor.get("1.0", "end-1c")
        new_text = _OUTPUT_RE.sub(lambda m: m.group(1) + absolutize(m.group(2)), text, count=1)

        match = _LOAD_BLOCK_RE.search(new_text)
        if match:
            for directive in ("pdb", "saxs"):
                value = self._load_value(directive)  # reads from the editor, still the original text
                if value:
                    block = _LOAD_BLOCK_RE.search(new_text)
                    new_block = self._rewrite_directive(block.group(0), directive, absolutize(value))
                    new_text = new_text[:block.start()] + new_block + new_text[block.end():]

        if new_text != text:
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", new_text)
            self.highlighter.highlight()

    def _populate_inputs_from_script(self):
        """Mirror the script's load block into the Input fields as a one-shot fill — run when a script
        is restored from the cache at boot or loaded from a file. The fields are passive afterwards:
        the file fields set here fire no commit callback, and later script edits don't propagate back,
        so the script stays the single source of truth."""
        pdb = self._load_value("pdb")
        if pdb:
            self.structure_field.set(pdb)
        saxs = self._load_value("saxs")
        if saxs:
            self.saxs_field.set(saxs)
        split = self._load_value("split")
        if split:
            # at boot the splits trace isn't attached yet; on a later load it is, but it only writes
            # the identical value straight back, so the script's substance is unchanged
            self.splits_var.set(split)

    # ----- script helpers -----------------------------------------------------
    def _set_load_directive(self, directive: str, value: str):
        """Write a single load directive (pdb/saxs/split) into the script's load block, replacing any existing line for
        that directive and leaving the rest of the script untouched. An empty value removes the directive. If no load
        block exists, one is created. This only ever fires when the user directly commits an Input field, so a
        hand-edited script is never silently overwritten."""
        value = value.strip()
        text = self.editor.get("1.0", "end-1c")
        match = _LOAD_BLOCK_RE.search(text)
        if match:
            new_block = self._rewrite_directive(match.group(0), directive, value)
            new_text = text[:match.start()] + new_block + text[match.end():]
        elif value:
            new_text = f"load {{\n    {directive} {value}\n}}\n" + text
        else:
            return  # nothing to add and no block to edit

        yview = self.editor.yview()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", new_text)
        self.editor.yview_moveto(yview[0])
        self.highlighter.highlight()
        self._schedule_preview_update()

    @staticmethod
    def _rewrite_directive(block: str, directive: str, value: str) -> str:
        """Return the load block with `directive` set to `value` (or removed if empty),
        preserving every other directive line and their order."""
        inner = re.match(r"load\s*\{(.*)\}", block, re.DOTALL).group(1)
        keyword = re.compile(rf"^\s*{re.escape(directive)}\b")
        new_line = f"{directive} {value}" if value else None

        kept: list[str] = []
        replaced = False
        for line in inner.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if keyword.match(line):
                if new_line is not None and not replaced:  # replace in place; drop duplicates
                    kept.append(new_line)
                    replaced = True
            else:
                kept.append(stripped)
        if new_line is not None and not replaced:  # directive was absent: append it
            kept.append(new_line)

        body = "".join(f"    {line}\n" for line in kept)
        return "load {\n" + body + "}"

    # ----- structure preview --------------------------------------------------
    def _load_value(self, directive: str):
        """Return the argument of a directive in the script's load block, or None."""
        match = _LOAD_BLOCK_RE.search(self.editor.get("1.0", "end-1c"))
        if not match:
            return None
        inner = re.match(r"load\s*\{(.*)\}", match.group(0), re.DOTALL).group(1)
        for line in inner.splitlines():
            tokens = line.split(None, 1)
            if tokens and tokens[0] == directive:
                return tokens[1].strip() if len(tokens) == 2 else ""
        return None

    @staticmethod
    def _parse_splits(value) -> list[int]:
        if not value:
            return []
        return [int(t) for t in re.split(r"[,\s]+", value.strip()) if t.isdigit()]

    def _schedule_preview_update(self):
        """Debounce preview redraws so rapid edits (e.g. typing splits) stay smooth."""
        if not hasattr(self, "_struct_ax"):
            return
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(150, self._update_structure_preview)

    @staticmethod
    def _structural_signature(script: str) -> tuple:
        """Distil the parts of the script that affect the preview — the load block, any symmetry elements, and any
        constraint lines — so edits to unrelated lines (iterations, print, save, ...) don't trigger a redraw or a
        backend rebuild."""
        load = _LOAD_BLOCK_RE.search(script)
        return (load.group(0) if load else "",
                tuple(m.group(0) for m in _SYMMETRY_RE.finditer(script)),
                tuple(m.group(0) for m in _CONSTRAINT_RE.finditer(script)))

    def _preview_data(self, script: str, sig: tuple):
        """Build the rigid body from the current script and return its preview structure
        (coords + per-atom body/copy/residue/Cα metadata), or None if it can't be built. Cached on
        the structural signature; skipped while a refinement runs to avoid a concurrent backend call."""
        if self.runner.running():
            return None
        if sig != self._preview_cache_key:
            self._preview_cache_key = sig
            if not self._load_value("pdb"):
                self._preview_cache = None
            else:
                try:
                    from ..wrapper.Rigidbody import Rigidbody
                    data = Rigidbody(script).preview_structure()
                    self._preview_cache = data if len(data["coords"]) else None
                except Exception:
                    self._preview_cache = None  # script mid-edit / invalid: show the placeholder
        return self._preview_cache

    _update_structure_preview_first_draw = True
    def _update_structure_preview(self):
        self._preview_job = None
        if self._live_meta is not None:
            return  # a live run owns the preview axis; don't draw the static preview over it
        script = self.editor.get("1.0", "end-1c")
        splits = self._parse_splits(self._load_value("split"))

        # redraw only when the load or symmetry elements change; everything else is ignored
        sig = self._structural_signature(script)
        if sig == self._preview_key:
            return
        self._preview_key = sig

        data = self._preview_data(script, sig)

        ax = self._struct_ax
        ax.clear()
        ax.set_axis_off()
        if data is None:
            ax.text2D(
                0.5, 0.5, "Set a structure to preview the splits", transform=ax.transAxes,
                ha="center", va="center", color=PALETTE["muted"], fontsize=10
            )
        else:
            draw_structure(ax, data, splits)
            if self._update_structure_preview_first_draw:
                self._update_structure_preview_first_draw = False
            elif self._last_valid_lims is not None:
                ax.set_xlim(self._last_valid_lims[0])
                ax.set_ylim(self._last_valid_lims[1])
                ax.set_zlim(self._last_valid_lims[2])
            self._last_valid_lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        self._struct_fig.set_layout_engine("tight")
        self._struct_canvas.draw_idle()

    def _home_preview(self):
        """Reset the structure preview to the auto-fit default view."""
        self._last_valid_lims = None
        self._preview_key = None  # force a redraw so the reset takes effect immediately
        self._update_structure_preview()

    # ----- actions ------------------------------------------------------------
    def _set_busy(self, busy: bool, label: str = "Run refinement"):
        state = "disabled" if busy else "normal"
        self.run_button.configure(state=state, text="Running…" if busy else "Run refinement")
        self.validate_button.configure(state=state)
        if busy:
            self.progress.pack(side="left", fill="x", expand=True, padx=(12, 0))
            self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _validate_clicked(self):
        if self.runner.running():
            return
        self._mode = "validate"
        self.console.clear()
        self.console.append("Validating script…\n\n")
        self._set_busy(True)
        self.runner.start(self.editor.get("1.0", "end-1c"), validate_only=True,
                          on_line=self.console.append, on_done=self._on_done)

    def _run_clicked(self):
        if self.runner.running():
            return
        self._mode = "run"
        self._collapse_editor()  # minimize the editor so the results have room
        self.console.clear()
        self.console.append("Running rigid-body refinement…\n\n")
        script = self.editor.get("1.0", "end-1c")
        # if the script publishes its structure (`update structure`), prepare to watch it live.
        # Done before starting the run so the run's parse is the last to reset the live buffer.
        self._begin_live_preview(script)
        self._set_busy(True)
        self.runner.start(script, validate_only=False, on_line=self.console.append, on_done=self._on_done)
        if self._live_meta is not None:
            self._live_job = self.after(self._LIVE_POLL_MS, self._poll_live)

    # ----- live structure preview during a run --------------------------------
    _LIVE_POLL_MS = 200

    def _begin_live_preview(self, script: str):
        """If the script contains an `update` element, build the backbone mask once (atom order is
        fixed after setup) so live frames can reuse it, and surface the structure tab."""
        self._live_meta = None
        self._live_version = 0
        if not _UPDATE_RE.search(script):
            return
        from ..wrapper.Rigidbody import Rigidbody
        meta = Rigidbody(script).preview_structure()
        if len(meta["coords"]):
            self._live_meta = meta
            self.results.select(self.structure_tab)

    def _poll_live(self):
        self._live_job = None
        if self._live_meta is None:
            return
        from ..wrapper.Rigidbody import Rigidbody
        coords, version = Rigidbody.live_structure()
        if (coords is not None and version != self._live_version
                and len(coords) == len(self._live_meta["coords"])):
            self._live_version = version
            self._draw_live_frame(coords)
        if self.runner.running():
            self._live_job = self.after(self._LIVE_POLL_MS, self._poll_live)

    def _draw_live_frame(self, coords):
        data = dict(self._live_meta, coords=coords)  # reuse the mask, swap in the live coordinates
        ax = self._struct_ax
        lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        ax.clear()
        ax.set_axis_off()
        draw_structure(ax, data, self._parse_splits(self._load_value("split")))
        self._struct_fig.set_layout_engine("tight")
        self._struct_canvas.draw_idle()
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])

    def _stop_live_preview(self):
        if self._live_job is not None:
            self.after_cancel(self._live_job)
            self._live_job = None
        if self._live_meta is not None:
            try:  # draw the final published frame, then release the preview axis
                from ..wrapper.Rigidbody import Rigidbody
                coords, _ = Rigidbody.live_structure()
                if coords is not None and len(coords) == len(self._live_meta["coords"]):
                    self._draw_live_frame(coords)
            except Exception:
                pass
        self._live_meta = None

    @staticmethod
    def _backend_message(err) -> str:
        """Strip the wrapper that _check_error_code adds (`AUSAXS: "fn" failed with error code N: "..."`), leaving
        just the backend's own message. Non-matching exceptions (e.g. library-unavailable) are returned unchanged."""
        match = re.match(r'^AUSAXS: ".*?" failed with error code \d+:\s*"(.*)"\s*$', str(err), re.DOTALL)
        return match.group(1) if match else str(err)

    def _on_done(self, done):
        self._set_busy(False)
        self._stop_live_preview()
        if done.error is not None:
            if done.error_streamed:
                # the backend already streamed the error; just note the failure
                self.console.append("\nRefinement failed.\n", tag="error")
            else:
                self.console.append(f"\n{self._backend_message(done.error)}\n", tag="error")
            return
        if self._mode == "validate":
            self.console.append("\nValidation successful.\n", tag="success")
            return

        self.console.append("\nRefinement completed.\n", tag="success")
        if done.result is None or done.result.size == 0:
            self.console.append("No fit curves were returned.\n")
            return
        # replace previous fit tabs but keep the persistent structure tab
        for tab in self._fit_tabs:
            self.results.forget(tab)
        self._fit_tabs.clear()
        for logx, title in ((False, "fit (log)"), (True, "fit (log-log)")):
            try:
                self._fit_tabs.append(add_figure_tab(self.results, fit_figure_from_curves(done.result, logx=logx), title))
            except (Exception, SystemExit) as e:
                self.console.append(f"Failed to plot results: {e}\n")
        if self._fit_tabs:
            self.results.select(self._fit_tabs[0])

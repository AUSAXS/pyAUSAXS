# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import glob
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from matplotlib.figure import Figure

from .plotting import (
    draw_structure, fit_figure, fit_figure_from_curves,
    plot_file_figure, pretty_plot_name,
)
from .runner import CliRunner, RigidbodyRunner
from .theme import FONTS, PALETTE
from .widgets import ConsolePane, FileField, RangeSlider, RigidbodyHighlighter

QMIN, QMAX = 1e-4, 1.0

STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".xyz"}
SAXS_EXTENSIONS      = {".dat", ".rsr", ".xvg"}
EM_MAP_EXTENSIONS    = {".map", ".ccp4", ".mrc", ".rec"}


def _make_validator(extensions: set[str], lib_check: str):
    """File validator using the AUSAXS library when available, file extensions otherwise."""
    def validate(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            from ..wrapper import Filetypes
            return getattr(Filetypes, lib_check)(path)
        except Exception:
            return os.path.splitext(path)[1].lower() in extensions
    return validate


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _output_arg(path: str) -> str:
    # the backend appends file stems to this with plain string concatenation, so a trailing separator is required
    return path if path.endswith("/") else path + "/"


# ----- shared results-notebook helpers ---------------------------------------
def results_placeholder(notebook: ttk.Notebook, text: str = "Results will appear here after a fit."):
    frame = ttk.Frame(notebook)
    ttk.Label(frame, text=text, style="Muted.TLabel").place(relx=0.5, rely=0.5, anchor="center")
    notebook.add(frame, text="Results")


def clear_results(notebook: ttk.Notebook):
    for tab in notebook.tabs():
        notebook.forget(tab)


def add_figure_tab(notebook: ttk.Notebook, fig, title: str):
    fig.patch.set_facecolor(PALETTE["surface"])
    frame = tk.Frame(notebook, background=PALETTE["surface"])
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.configure(background=PALETTE["surface"])
    for child in toolbar.winfo_children():
        try:
            child.configure(background=PALETTE["surface"])
        except tk.TclError:
            pass
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    notebook.add(frame, text=title)
    return frame


def add_text_tab(notebook: ttk.Notebook, content: str, title: str):
    from .theme import FONTS
    frame = ttk.Frame(notebook)
    text = tk.Text(
        frame, wrap="none", font=FONTS["mono"], relief="flat", 
        background=PALETTE["surface"], foreground=PALETTE["text"], padx=12, pady=10, borderwidth=0
    )
    text.insert("1.0", content)
    text.configure(state="disabled")
    scroll = ttk.Scrollbar(frame, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(fill="both", expand=True)
    notebook.add(frame, text=title)


def _saxs_data_span(path: str) -> tuple[float, float] | None:
    """Extract the q-range spanned by a SAXS data file, ignoring non-numeric lines."""
    qs = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                words = line.split()
                if len(words) < 2:
                    continue
                try:
                    qs.append(float(words[0]))
                    float(words[1])
                except ValueError:
                    continue
    except OSError:
        return None
    return (min(qs), max(qs)) if qs else None


class FitterPane(ttk.Frame):
    """Common scaffolding for the SAXS and EM fitter panes: an input/settings column on
    the left, and a results notebook with embedded plots on the right."""

    title = ""

    def __init__(self, parent):
        super().__init__(parent)
        self.runner = CliRunner(self)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # left column: inputs, settings, run controls, console
        left = ttk.Frame(paned, padding=(4, 4, 10, 4), width=360)
        left.pack_propagate(False)
        paned.add(left, weight=0)

        self.input_frame = ttk.Labelframe(left, text="Input", padding=12)
        self.input_frame.pack(fill="x")
        self.settings_frame = ttk.Labelframe(left, text="Settings", padding=12)
        self.settings_frame.pack(fill="x", pady=(12, 0))

        run_frame = ttk.Frame(left)
        run_frame.pack(fill="x", pady=12)
        self.run_button = ttk.Button(run_frame, text="Run fit", style="Accent.TButton", command=self._run_clicked)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(run_frame, mode="indeterminate")  # packed only while running

        console_frame = ttk.Labelframe(left, text="Log", padding=(2, 4))
        console_frame.pack(fill="both", expand=True)
        self.console = ConsolePane(console_frame)
        self.console.pack(fill="both", expand=True, padx=2, pady=2)

        # right column: results
        right = ttk.Frame(paned, padding=(10, 4, 4, 4))
        paned.add(right, weight=1)
        self.results = ttk.Notebook(right)
        self.results.pack(fill="both", expand=True)
        self._results_placeholder()

        self._build_inputs(self.input_frame)
        self._build_settings(self.settings_frame)

    # ----- subclass interface -------------------------------------------------
    def _build_inputs(self, parent):
        raise NotImplementedError

    def _build_settings(self, parent):
        raise NotImplementedError

    def _input_fields(self) -> list[FileField]:
        raise NotImplementedError

    def _build_command(self) -> tuple[str, list[str]]:
        """Return the library CLI function name and its argv (including the program name)."""
        raise NotImplementedError

    def _result_dir(self) -> str:
        raise NotImplementedError

    # ----- shared behavior ----------------------------------------------------
    def _make_q_slider(self, parent) -> RangeSlider:
        ttk.Label(parent, text="q-range [1/Å]").pack(anchor="w", pady=(8, 0))
        slider = RangeSlider(parent, QMIN, QMAX, log=True, fmt="{:.4g}")
        slider.pack(fill="x")
        return slider

    def _update_q_range(self, saxs_path: str):
        span = _saxs_data_span(saxs_path)
        if span:
            self.q_slider.set_values(max(span[0] - 1e-3, QMIN), min(span[1] + 1e-3, QMAX))

    def _autodetect_saxs(self, near_file: str, saxs_field: FileField):
        """Look for a SAXS data file next to the given input file, like the old GUI did."""
        if saxs_field.valid:
            return
        directory = os.path.dirname(os.path.abspath(near_file))
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return
        if 20 < len(entries):
            return
        for entry in entries:
            if os.path.splitext(entry)[1].lower() in SAXS_EXTENSIONS:
                saxs_field.set(os.path.join(directory, entry))
                return

    def _run_clicked(self):
        if self.runner.running():
            return

        fields = self._input_fields()
        for field in fields:
            field.validate()
        if not all(field.valid for field in fields):
            self.console.append("Missing or invalid input files.\n")
            return

        func_name, argv = self._build_command()
        self.console.clear()
        self.console.append(" ".join(argv) + "\n\n")
        self.run_button.configure(text="Running…", state="disabled")
        self.progress.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.progress.start(15)
        self.runner.start(func_name, argv, on_line=self.console.append, on_done=self._run_finished)

    def _run_finished(self, returncode: int):
        self.progress.stop()
        self.progress.pack_forget()
        self.run_button.configure(text="Run fit", state="normal")
        if returncode != 0:
            self.console.append(f"\nFit failed with exit code {returncode}.\n")
            return
        result_dir = self._result_dir()
        self.console.append(f"\nFit completed. Results written to \"{result_dir}\".\n")
        self._load_results(result_dir)

    # ----- results ------------------------------------------------------------
    def _results_placeholder(self):
        results_placeholder(self.results)

    def _load_results(self, result_dir: str):
        clear_results(self.results)

        fit_file = os.path.join(result_dir, "ausaxs.fit")
        if os.path.isfile(fit_file):
            for logx, title in ((False, "fit (log)"), (True, "fit (log-log)")):
                try:
                    add_figure_tab(self.results, fit_figure(fit_file, logx=logx), title)
                except (Exception, SystemExit) as e:
                    self.console.append(f"Failed to plot \"{fit_file}\": {e}\n")

        for plot_file in sorted(glob.glob(os.path.join(result_dir, "*.plot"))):
            try:
                fig = plot_file_figure(plot_file)
            except (Exception, SystemExit) as e:
                self.console.append(f"Failed to plot \"{plot_file}\": {e}\n")
                continue
            add_figure_tab(self.results, fig, pretty_plot_name(_file_stem(plot_file)))

        report_file = os.path.join(result_dir, "report.txt")
        if os.path.isfile(report_file):
            with open(report_file, errors="replace") as f:
                add_text_tab(self.results, f.read(), "report")

        if not self.results.tabs():
            results_placeholder(self.results)
            self.console.append(f"No results found in \"{result_dir}\".\n")


class SaxsFitterPane(FitterPane):
    title = "SAXS fitter"

    def _build_inputs(self, parent):
        self.saxs_field = FileField(
            parent, "SAXS data",
            validator=_make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file"),
            on_valid=self._update_q_range,
            filetypes=[("SAXS data", "*.dat *.rsr *.xvg")],
        )
        self.structure_field = FileField(
            parent, "Structure",
            validator=_make_validator(STRUCTURE_EXTENSIONS, "_is_pdb_file"),
            on_valid=lambda path: self._autodetect_saxs(path, self.saxs_field),
            filetypes=[("Structure", "*.pdb *.ent *.cif *.xyz")],
        )
        self.output_field = FileField(parent, "Output folder", validator=lambda _p: True, directory=True)
        self.output_field.set("output/saxs_fitter")

        self.structure_field.pack(fill="x")
        self.saxs_field.pack(fill="x", pady=(6, 0))
        self.output_field.pack(fill="x", pady=(6, 0))

    def _build_settings(self, parent):
        self.q_slider = self._make_q_slider(parent)

        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(8, 0))
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="q unit").grid(row=0, column=0, sticky="w")
        self.unit_var = tk.StringVar(value="1/Å")
        ttk.Combobox(
            grid, textvariable=self.unit_var, values=["1/Å", "1/nm"],
            state="readonly", width=12).grid(row=0, column=1, sticky="ew", pady=2
        )

        ttk.Label(grid, text="Hydration model").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.hydration_var = tk.StringVar(value="radial")
        ttk.Combobox(grid, textvariable=self.hydration_var, values=["radial", "none"], state="readonly", width=12).grid(
            row=1, column=1, sticky="ew", pady=2
        )

        ttk.Label(grid, text="Excluded volume model").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self.exv_var = tk.StringVar(value="simple")
        exv_box = ttk.Combobox(grid, textvariable=self.exv_var, values=["simple", "fraser", "grid"], state="readonly", width=12)
        exv_box.grid(row=2, column=1, sticky="ew", pady=2)
        exv_box.bind("<<ComboboxSelected>>", lambda _e: self._exv_changed())

        self.fit_exv_var = tk.BooleanVar(value=False)
        self.fit_density_var = tk.BooleanVar(value=False)
        self.fit_exv_check = ttk.Checkbutton(parent, text="Fit excluded volume", variable=self.fit_exv_var)
        self.fit_density_check = ttk.Checkbutton(parent, text="Fit solvent density", variable=self.fit_density_var)
        self.fit_exv_check.pack(anchor="w", pady=(6, 0))
        self.fit_density_check.pack(anchor="w")
        self._exv_changed()

    def _exv_changed(self):
        # the simple excluded volume model does not support fitting its parameters
        state = "disabled" if self.exv_var.get() == "simple" else "normal"
        if state == "disabled":
            self.fit_exv_var.set(False)
            self.fit_density_var.set(False)
        self.fit_exv_check.configure(state=state)
        self.fit_density_check.configure(state=state)

    def _input_fields(self):
        return [self.structure_field, self.saxs_field]

    def _build_command(self):
        qmin, qmax = self.q_slider.values()
        argv = [
            "saxs_fitter",
            self.structure_field.get(), self.saxs_field.get(),
            "--output", _output_arg(self.output_field.get()),
            "data", "--qmin", f"{qmin:.6g}", "--qmax", f"{qmax:.6g}",
            "--unit", "nm" if self.unit_var.get() == "1/nm" else "A",
        ]
        argv += ["exv", "--model", self.exv_var.get()]
        if self.fit_exv_var.get():
            argv += ["--fit"]
        if self.fit_density_var.get():
            argv += ["--fit-density"]
        argv += ["solv", "--model", self.hydration_var.get()]
        return "cli_saxs_fitter", argv

    def _result_dir(self):
        # the CLI appends the measurement stem to the output folder
        return os.path.join(self.output_field.get(), _file_stem(self.saxs_field.get()))


class EmFitterPane(FitterPane):
    title = "EM fitter"

    def _build_inputs(self, parent):
        self.saxs_field = FileField(
            parent, "SAXS data",
            validator=_make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file"),
            on_valid=self._update_q_range,
            filetypes=[("SAXS data", "*.dat *.rsr *.xvg")],
        )
        self.map_field = FileField(
            parent, "EM map",
            validator=_make_validator(EM_MAP_EXTENSIONS, "_is_em_map_file"),
            on_valid=lambda path: self._autodetect_saxs(path, self.saxs_field),
            filetypes=[("EM map", "*.map *.ccp4 *.mrc *.rec")],
        )
        self.output_field = FileField(parent, "Output folder", validator=lambda _p: True, directory=True)
        self.output_field.set("output/em_fitter")

        self.map_field.pack(fill="x")
        self.saxs_field.pack(fill="x", pady=(6, 0))
        self.output_field.pack(fill="x", pady=(6, 0))

    def _build_settings(self, parent):
        self.q_slider = self._make_q_slider(parent)

        ttk.Label(parent, text="alpha levels").pack(anchor="w", pady=(8, 0))
        self.alpha_slider = RangeSlider(parent, 0, 15, start=(1, 10), fmt="{:.3g}")
        self.alpha_slider.pack(fill="x")

        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(8, 0))
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="q unit").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.unit_var = tk.StringVar(value="1/Å")
        ttk.Combobox(grid, textvariable=self.unit_var, values=["1/Å", "1/nm"],
                     state="readonly", width=12).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Fit evaluations").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.iterations_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.iterations_var, width=12).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Sample frequency").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self.frequency_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.frequency_var, width=12).grid(row=2, column=1, sticky="ew", pady=2)

        self.hydrate_var = tk.BooleanVar(value=True)
        self.fixed_weights_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Hydrate", variable=self.hydrate_var).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(parent, text="Fixed weights", variable=self.fixed_weights_var).pack(anchor="w")

    def _input_fields(self):
        return [self.map_field, self.saxs_field]

    def _build_command(self):
        qmin, qmax = self.q_slider.values()
        amin, amax = self.alpha_slider.values()
        argv = [
            "em_fitter",
            self.map_field.get(), self.saxs_field.get(),
            "--output", _output_arg(self.output_field.get()),
            "saxs", "--qmin", f"{qmin:.6g}", "--qmax", f"{qmax:.6g}",
            "--unit", "nm" if self.unit_var.get() == "1/nm" else "A",
        ]
        argv += [
            "em", "--levelmin", f"{amin:.6g}", "--levelmax", f"{amax:.6g}",
            "--hydrate" if self.hydrate_var.get() else "--no-hydrate",
            "--fixed-weight" if self.fixed_weights_var.get() else "--dynamic-weight",
        ]
        if self.frequency_var.get().strip():
            argv += ["--frequency", self.frequency_var.get().strip()]
        if self.iterations_var.get().strip():
            argv += ["fit", "--max-iterations", self.iterations_var.get().strip()]
        return "cli_em_fitter", argv

    def _result_dir(self):
        # the CLI appends both the measurement and map stems to the output folder
        return os.path.join(
            self.output_field.get(),
            _file_stem(self.saxs_field.get()),
            _file_stem(self.map_field.get()),
        )


DEFAULT_RIGIDBODY_SCRIPT = """\
output output/rigidbody/
load {
    pdb
    saxs
    split
}
autoconstrain linear
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


class RigidbodyPane(ttk.Frame):
    """[Experimental] Rigid-body refinement driven by an AUSAXS sequencer script.

    Mirrors the SasView rigid-body refinement workflow: a script editor with helpers to fill in the load block from 
    chosen files, plus Validate and Run actions that stream the backend's output into a log and plot the resulting fit."""

    title = "Rigidbody"

    def __init__(self, parent):
        super().__init__(parent)
        self.runner = RigidbodyRunner(self)
        self._mode = "run"
        self._expanded = False
        self._fit_tabs: list = []        # result tabs added by a run (the structure tab persists)
        self._preview_job = None         # pending debounced preview redraw
        self._preview_key = None         # signature of what the preview currently shows
        self._preview_cache_key = None   # signature the preview structure was last built from
        self._preview_cache = None       # cached backend preview-structure dict, or None
        self._script_cache_path = None   # where the script is autosaved/restored
        self._last_saved_script = None   # last text written, to skip unchanged autosaves
        self._autosave_job = None        # pending periodic autosave

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

        def _on_load_structure(p):
            self._set_load_directive("pdb", p)
            if p and not self.saxs_field.valid:
                # first try direct filename match
                for ext in SAXS_EXTENSIONS:
                    candidate = str(Path(p).with_suffix(ext))
                    if os.path.isfile(candidate):
                        self.saxs_field.set(candidate)
                        self._set_load_directive("saxs", candidate)
                        break
                
                # then check if only a single data file is present, and if so, use it
                else:
                    directory = os.path.dirname(os.path.abspath(p))
                    try:
                        entries = sorted(os.listdir(directory))
                    except OSError:
                        return
                    if 20 < len(entries):
                        return
                    saxs_candidates = [e for e in entries if os.path.splitext(e)[1].lower() in SAXS_EXTENSIONS]
                    if len(saxs_candidates) == 1:
                        self.saxs_field.set(os.path.join(directory, saxs_candidates[0]))
                        self._set_load_directive("saxs", saxs_candidates[0])

        def _on_load_saxs(p):
            self._set_load_directive("saxs", p)
            if p and not self.structure_field.valid:
                # first try direct filename match
                for ext in STRUCTURE_EXTENSIONS:
                    candidate = str(Path(p).with_suffix(ext))
                    if os.path.isfile(candidate):
                        self.structure_field.set(candidate)
                        self._set_load_directive("pdb", candidate)
                        break
                
                # then check if only a single structure file is present, and if so, use it
                else:
                    directory = os.path.dirname(os.path.abspath(p))
                    try:
                        entries = sorted(os.listdir(directory))
                    except OSError:
                        return
                    if 20 < len(entries):
                        return
                    struct_candidates = [e for e in entries if os.path.splitext(e)[1].lower() in STRUCTURE_EXTENSIONS]
                    if len(struct_candidates) == 1:
                        self.structure_field.set(os.path.join(directory, struct_candidates[0]))
                        self._set_load_directive("pdb", struct_candidates[0])

        self.structure_field = FileField(
            input_frame, "Structure",
            validator=_make_validator(STRUCTURE_EXTENSIONS, "_is_pdb_file"),
            on_commit=lambda p: _on_load_structure(p),
            filetypes=[("Structure", "*.pdb *.ent *.cif *.xyz")],
        )
        self.saxs_field = FileField(
            input_frame, "SAXS data",
            validator=_make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file"),
            on_commit=lambda p: _on_load_saxs(p),
            filetypes=[("SAXS data", "*.dat *.rsr *.xvg")],
        )
        self.structure_field.pack(fill="x")
        self.saxs_field.pack(fill="x", pady=(6, 0))

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
        self.run_button = ttk.Button(run_frame, text="Run refinement", style="Accent.TButton",
                                     command=self._run_clicked)
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
        self.reset_button = ttk.Label(
            title_row, text="✕", cursor="hand2",
            foreground=PALETTE["danger"], font=(FONTS["base"][0], 11, "bold"))
        self.reset_button.pack(side="right", padx=(0, 2))
        self.reset_button.bind("<Button-1>", lambda _e: self._reset_clicked())
        self.reset_button.bind("<Enter>", lambda _e: self.reset_button.configure(foreground=PALETTE["danger_hover"]))
        self.reset_button.bind("<Leave>", lambda _e: self.reset_button.configure(foreground=PALETTE["danger"]))
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
        editor_scroll = ttk.Scrollbar(editor_frame, command=self.editor.yview)
        self.editor.configure(yscrollcommand=editor_scroll.set)
        editor_scroll.pack(side="right", fill="y")
        self.editor.pack(fill="both", expand=True, padx=2, pady=2)
        operations, keywords = self._fetch_vocabulary()
        self.highlighter = RigidbodyHighlighter(self.editor, operations, keywords)
        # restore the last session's script from the cache, falling back to the default
        self._script_cache_path = self._resolve_script_cache_path()
        initial_script = self._load_cached_script() or DEFAULT_RIGIDBODY_SCRIPT
        self.editor.insert("1.0", initial_script)
        self._last_saved_script = initial_script
        self.highlighter.highlight()
        # manual edits to the script (e.g. the pdb/split lines) refresh both the syntax highlighting and the structure preview; clicking 
        # moves the cursor, which can change the highlighted scope pair
        self.editor.bind("<KeyRelease>", self._on_editor_changed)
        self.editor.bind("<ButtonRelease-1>", lambda _e: self.highlighter.highlight_brackets())
        # Ctrl-A selects all (Tk's default binds it to "start of line")
        self.editor.bind("<Control-a>", self._select_all)
        self.editor.bind("<Control-A>", self._select_all)

        # --- results pane (right), the larger pane by default ----------------
        self.results_pane = ttk.Frame(self.outer, padding=(10, 4, 4, 4))
        self.outer.add(self.results_pane, weight=3)
        self.results = ttk.Notebook(self.results_pane)
        self.results.pack(fill="both", expand=True)

        # a persistent "structure" tab: a 3D Cα backbone with the split residues in red,
        # kept across runs (the fit tabs are added alongside it)
        self.structure_tab = tk.Frame(self.results, background=PALETTE["surface"])
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

    # ----- script persistence -------------------------------------------------
    _AUTOSAVE_INTERVAL_MS = 10_000

    @staticmethod
    def _resolve_script_cache_path() -> str:
        """Path the script is autosaved to: <AUSAXS cache>/gui_rigidbody_script.txt."""
        from ..wrapper.settings import settings
        return os.path.join(settings.get("cache"), "gui_rigidbody_script.txt")

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
        """Distil the parts of the script that affect the preview — the load block and any symmetry elements — so 
        edits to unrelated lines (iterations, print, save, ...) don't trigger a redraw or a backend rebuild."""
        load = _LOAD_BLOCK_RE.search(script)
        return (load.group(0) if load else "",
                tuple(m.group(0) for m in _SYMMETRY_RE.finditer(script)))

    def _preview_data(self, script: str, sig: tuple):
        """Build the rigid body from the current script and return its preview structure 
        (coords + per-atom body/copy/residue/Cα metadata), or None if it can't be built. Cached on 
        the structural signature; skipped while a refinement runs to avoid a concurrent backend call."""
        if self.runner.running():
            return None
        if sig != self._preview_cache_key:
            self._preview_cache_key = sig
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
        script = self.editor.get("1.0", "end-1c")
        splits = self._parse_splits(self._load_value("split"))

        # redraw only when the load or symmetry elements change; everything else is ignored
        sig = self._structural_signature(script)
        if sig == self._preview_key:
            return
        self._preview_key = sig

        data = self._preview_data(script, sig)

        ax = self._struct_ax
        lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
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
            else:
                ax.set_xlim(lims[0])
                ax.set_ylim(lims[1])
                ax.set_zlim(lims[2])
        self._struct_fig.set_layout_engine("tight")
        self._struct_canvas.draw_idle()

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
        self._set_busy(True)
        self.runner.start(self.editor.get("1.0", "end-1c"), validate_only=False, on_line=self.console.append, on_done=self._on_done)

    @staticmethod
    def _backend_message(err) -> str:
        """Strip the wrapper that _check_error_code adds (`AUSAXS: "fn" failed with error code N: "..."`), leaving 
        just the backend's own message. Non-matching exceptions (e.g. library-unavailable) are returned unchanged."""
        match = re.match(r'^AUSAXS: ".*?" failed with error code \d+:\s*"(.*)"\s*$', str(err), re.DOTALL)
        return match.group(1) if match else str(err)

    def _on_done(self, done):
        self._set_busy(False)
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

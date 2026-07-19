# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import glob
import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .plotting import fit_figure, plot_file_figure, pretty_plot_name
from .runner import CliRunner
from .theme import FONTS, PALETTE
from .widgets import ConsolePane, FileField

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


def _read_saxs_data(path: str, unit: str = "A") -> tuple[list, list, list] | None:
    """Read a SAXS file through the bundled backend (Datafile), so the preview matches
    exactly what a fit will see. `unit` ("A" or "nm") forces the backend's interpretation
    of the file's q column, same as the --unit CLI flag; qmin/qmax are pushed to the full
    axis range first so the reader's own q-range clamping doesn't drop any rows here."""
    from ..wrapper.Datafile import Datafile
    from ..wrapper.settings import settings as backend_settings
    try:
        backend_settings.histogram(qmin=QMIN, qmax=QMAX, unit=unit)
        q, I, Ierr = Datafile(path).data()
    except Exception:
        return None
    return (q.tolist(), I.tolist(), Ierr.tolist()) if len(q) else None



def make_on_load_structure(set_load_directive=None, saxs_field=None):
    """Return a handler that mirrors a chosen structure file into the SAXS field.

    If `set_load_directive` is provided it will be called as `set_load_directive("pdb", path)` and 
    `set_load_directive("saxs", candidate)` when a candidate is found (used by the script editor). 
    `saxs_field` is a FileField instance to populate.
    """
    def _on_load_structure(p: str):
        if set_load_directive:
            set_load_directive("pdb", p)
        if not p or saxs_field is None or saxs_field.valid:
            return
        # try direct filename match (same stem, SAXS extensions)
        for ext in SAXS_EXTENSIONS:
            candidate = str(Path(p).with_suffix(ext))
            if os.path.isfile(candidate):
                saxs_field.set(candidate)
                if set_load_directive:
                    set_load_directive("saxs", candidate)
                return
        # otherwise, if the directory is small, look for a single SAXS file
        directory = os.path.dirname(os.path.abspath(p))
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return
        if 20 < len(entries):
            return
        saxs_candidates = [e for e in entries if os.path.splitext(e)[1].lower() in SAXS_EXTENSIONS]
        if len(saxs_candidates) == 1:
            saxs_field.set(os.path.join(directory, saxs_candidates[0]))
            if set_load_directive:
                set_load_directive("saxs", saxs_candidates[0])

    return _on_load_structure


def make_on_load_saxs(set_load_directive=None, structure_field=None):
    """Return a handler that mirrors a chosen SAXS file into the structure field.

    If `set_load_directive` is provided it will be called as `set_load_directive("saxs", path)` and `
    set_load_directive("pdb", candidate)` when a candidate is found. `structure_field` is a FileField to populate.
    """
    def _on_load_saxs(p: str):
        if set_load_directive:
            set_load_directive("saxs", p)
        if not p or structure_field is None or structure_field.valid:
            return
        for ext in STRUCTURE_EXTENSIONS:
            candidate = str(Path(p).with_suffix(ext))
            if os.path.isfile(candidate):
                structure_field.set(candidate)
                if set_load_directive:
                    set_load_directive("pdb", candidate)
                return
        directory = os.path.dirname(os.path.abspath(p))
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return
        if 20 < len(entries):
            return
        struct_candidates = [e for e in entries if os.path.splitext(e)[1].lower() in STRUCTURE_EXTENSIONS]
        if len(struct_candidates) == 1:
            structure_field.set(os.path.join(directory, struct_candidates[0]))
            if set_load_directive:
                set_load_directive("pdb", struct_candidates[0])

    return _on_load_saxs


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

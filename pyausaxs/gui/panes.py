# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import glob
import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .plotting import fit_figure, plot_file_figure, pretty_plot_name
from .runner import CliRunner
from .theme import PALETTE
from .widgets import ConsolePane, FileField, RangeSlider

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
    # the backend appends file stems to this with plain string concatenation,
    # so a trailing separator is required
    return path if path.endswith("/") else path + "/"


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
        self.run_button = ttk.Button(run_frame, text="Run fit", style="Accent.TButton",
                                     command=self._run_clicked)
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
        frame = ttk.Frame(self.results)
        ttk.Label(frame, text="Results will appear here after a fit.",
                  style="Muted.TLabel").place(relx=0.5, rely=0.5, anchor="center")
        self.results.add(frame, text="Results")

    def _clear_results(self):
        for tab in self.results.tabs():
            self.results.forget(tab)

    def _add_figure_tab(self, fig, title: str):
        fig.patch.set_facecolor(PALETTE["surface"])
        frame = tk.Frame(self.results, background=PALETTE["surface"])
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
        self.results.add(frame, text=title)

    def _add_text_tab(self, content: str, title: str):
        from .theme import FONTS
        frame = ttk.Frame(self.results)
        text = tk.Text(frame, wrap="none", font=FONTS["mono"], relief="flat",
                       background=PALETTE["surface"], foreground=PALETTE["text"],
                       padx=12, pady=10, borderwidth=0)
        text.insert("1.0", content)
        text.configure(state="disabled")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        self.results.add(frame, text=title)

    def _load_results(self, result_dir: str):
        self._clear_results()

        fit_file = os.path.join(result_dir, "ausaxs.fit")
        if os.path.isfile(fit_file):
            for logx, title in ((False, "fit (log)"), (True, "fit (log-log)")):
                try:
                    self._add_figure_tab(fit_figure(fit_file, logx=logx), title)
                except (Exception, SystemExit) as e:
                    self.console.append(f"Failed to plot \"{fit_file}\": {e}\n")

        for plot_file in sorted(glob.glob(os.path.join(result_dir, "*.plot"))):
            try:
                fig = plot_file_figure(plot_file)
            except (Exception, SystemExit) as e:
                self.console.append(f"Failed to plot \"{plot_file}\": {e}\n")
                continue
            self._add_figure_tab(fig, pretty_plot_name(_file_stem(plot_file)))

        report_file = os.path.join(result_dir, "report.txt")
        if os.path.isfile(report_file):
            with open(report_file, errors="replace") as f:
                self._add_text_tab(f.read(), "report")

        if not self.results.tabs():
            self._results_placeholder()
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
        ttk.Combobox(grid, textvariable=self.unit_var, values=["1/Å", "1/nm"],
                     state="readonly", width=12).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Hydration model").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.hydration_var = tk.StringVar(value="radial")
        ttk.Combobox(grid, textvariable=self.hydration_var, values=["radial", "none"],
                     state="readonly", width=12).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Excluded volume model").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self.exv_var = tk.StringVar(value="simple")
        exv_box = ttk.Combobox(grid, textvariable=self.exv_var, values=["simple", "fraser", "grid"],
                               state="readonly", width=12)
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

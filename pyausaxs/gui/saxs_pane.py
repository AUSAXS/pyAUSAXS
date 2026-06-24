# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import tkinter as tk
from tkinter import ttk

from .panes import (
    FitterPane, _make_validator, _file_stem, _output_arg,
    SAXS_EXTENSIONS, STRUCTURE_EXTENSIONS,
)
from .widgets import FileField


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

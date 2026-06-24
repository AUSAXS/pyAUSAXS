# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import tkinter as tk
from tkinter import ttk

from .panes import (
    FitterPane, _make_validator, _file_stem, _output_arg,
    SAXS_EXTENSIONS, EM_MAP_EXTENSIONS,
)
from .widgets import FileField, RangeSlider


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

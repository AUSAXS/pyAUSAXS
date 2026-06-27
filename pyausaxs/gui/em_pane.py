# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import tkinter as tk
from tkinter import messagebox, ttk

from .data_pane import SaxsDataPane
from .panes import (
    FitterPane, _make_validator, _file_stem, _output_arg,
    QMIN, QMAX, SAXS_EXTENSIONS, EM_MAP_EXTENSIONS,
)
from .widgets import FileField


class EmFitterPane(FitterPane):
    title = "EM fitter"

    def __init__(self, parent):
        self._data_pane: SaxsDataPane | None = None
        super().__init__(parent)

    def _build_inputs(self, parent):
        self.saxs_field = FileField(
            parent, "SAXS data",
            validator=_make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file"),
            on_valid=lambda _p: self._refresh_view_btn(),
            on_commit=self._on_saxs_commit,
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

        self._view_btn = ttk.Button(parent, text="View data", command=self._open_data_pane,
                                    state="disabled")
        self._view_btn.pack(anchor="e", pady=(8, 0))

    def _build_settings(self, parent):
        ttk.Label(parent, text="alpha levels").pack(anchor="w", pady=(4, 0))
        from .widgets import RangeSlider
        self.alpha_slider = RangeSlider(parent, 0, 15, start=(1, 10), fmt="{:.3g}")
        self.alpha_slider.pack(fill="x")

        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(4, 0))
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="q unit").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.unit_var = tk.StringVar()
        unit_box = ttk.Combobox(grid, textvariable=self.unit_var, values=["1/Å", "1/nm"],
                                 state="readonly", width=12)
        unit_box.set("1/Å")
        unit_box.grid(row=0, column=1, sticky="ew", pady=2)

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

    # ----- data pane management -----------------------------------------------

    def _on_saxs_commit(self, path: str):
        if self._data_pane is not None and self._data_pane.file_path != path:
            stem = self._data_pane.title
            if not messagebox.askyesno(
                    "Discard data pane",
                    f'Loading a new dataset will close the "{stem}" data pane and '
                    f'discard its customisation. Continue?',
                    parent=self):
                self.saxs_field.set(self._data_pane.file_path)
                return
            self._close_data_pane()
        self._refresh_view_btn()

    def _refresh_view_btn(self):
        """Enable "View data" whenever the SAXS field is valid. Driven by on_valid so it
        also fires for paths set programmatically (autodetection), not just on commit."""
        if hasattr(self, "_view_btn"):
            self._view_btn.configure(state="normal" if self.saxs_field.valid else "disabled")

    def _open_data_pane(self):
        path = self.saxs_field.get()
        if not path:
            return
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

    # ----- FitterPane interface ------------------------------------------------

    def _input_fields(self):
        return [self.map_field, self.saxs_field]

    def _build_command(self):
        if self._data_pane is not None:
            qmin, qmax = self._data_pane.qrange()
        else:
            qmin, qmax = QMIN, QMAX
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
        return os.path.join(
            self.output_field.get(),
            _file_stem(self.saxs_field.get()),
            _file_stem(self.map_field.get()),
        )

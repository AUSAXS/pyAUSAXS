# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

import os
import tkinter as tk
from tkinter import messagebox, ttk

from .data_pane import SaxsDataPane
from .panes import (
    FitterPane, _make_validator, _file_stem, _output_arg,
    QMIN, QMAX, SAXS_EXTENSIONS, STRUCTURE_EXTENSIONS,
    make_on_load_structure, make_on_load_saxs,
)
from .widgets import FileField


class SaxsFitterPane(FitterPane):
    title = "SAXS fitter"

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
        self.structure_field = FileField(
            parent, "Structure",
            validator=_make_validator(STRUCTURE_EXTENSIONS, "_is_pdb_file"),
            on_valid=lambda path: self._autodetect_saxs(path, self.saxs_field),
            on_commit=lambda p: self._on_load_structure(p),
            filetypes=[("Structure", "*.pdb *.ent *.cif *.xyz")],
        )
        self.output_field = FileField(parent, "Output folder", validator=lambda _p: True, directory=True)
        self.output_field.set("output/saxs_fitter")

        self.structure_field.pack(fill="x")
        self.saxs_field.pack(fill="x", pady=(6, 0))
        self.output_field.pack(fill="x", pady=(6, 0))

        self._view_btn = ttk.Button(parent, text="View data", command=self._open_data_pane,
                                    state="disabled")
        self._view_btn.pack(anchor="e", pady=(8, 0))

        self._on_load_structure = make_on_load_structure(None, self.saxs_field)
        self._on_load_saxs = make_on_load_saxs(None, self.structure_field)

    def _build_settings(self, parent):
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="q unit").grid(row=0, column=0, sticky="w")
        self.unit_var = tk.StringVar()
        unit_box = ttk.Combobox(grid, textvariable=self.unit_var, values=["1/Å", "1/nm"],
                                 state="readonly", width=12)
        unit_box.set("1/Å")
        unit_box.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Hydration model").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.hydration_var = tk.StringVar()
        hydration_box = ttk.Combobox(grid, textvariable=self.hydration_var,
                                      values=["radial", "none"], state="readonly", width=12)
        hydration_box.set("radial")
        hydration_box.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(grid, text="Excluded volume model").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self.exv_var = tk.StringVar()
        exv_box = ttk.Combobox(grid, textvariable=self.exv_var,
                                values=["simple", "fraser", "grid"], state="readonly", width=12)
        exv_box.set("simple")
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
        state = "disabled" if self.exv_var.get() == "simple" else "normal"
        if state == "disabled":
            self.fit_exv_var.set(False)
            self.fit_density_var.set(False)
        self.fit_exv_check.configure(state=state)
        self.fit_density_check.configure(state=state)

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
        self._on_load_saxs(path)
        self._refresh_view_btn()

    def _refresh_view_btn(self):
        """Enable "View data" whenever the SAXS field is valid. Driven by on_valid so it
        also fires for paths set programmatically (autodetection, Send to SAXS fitter),
        not just on an explicit commit."""
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
        return [self.structure_field, self.saxs_field]

    def _build_command(self):
        if self._data_pane is not None:
            qmin, qmax = self._data_pane.qrange()
        else:
            qmin, qmax = QMIN, QMAX
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
        return os.path.join(self.output_field.get(), _file_stem(self.saxs_field.get()))

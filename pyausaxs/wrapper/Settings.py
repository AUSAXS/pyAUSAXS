from .AUSAXS import AUSAXS, _check_error_code
from .Models import ExvModel, ExvTable, WaterModel
import ctypes as ct

class Settings:
    def set_exv_settings(self, exv_model: ExvModel = ExvModel.simple):
        exv_model = ExvModel.validate(exv_model)
        ausaxs = AUSAXS()
        status = ct.c_int()
        model_ptr = ct.c_char_p(exv_model.value.encode('utf-8'))
        ausaxs.lib().functions.settings_set_exv_model(
            model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_exv_model")

    def set_fit_settings(
        self, 
        max_iterations: int = 100, 
        fit_hydration: bool = True,
        fit_excluded_volume: bool = False,
        fit_solvent_density: bool = False,
        fit_atomic_debye_waller: bool = False, 
        fit_exv_debye_waller: bool = False
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.settings_set_fit_settings(
            ct.c_uint(max_iterations),
            ct.c_bool(fit_excluded_volume),
            ct.c_bool(fit_solvent_density),
            ct.c_bool(fit_hydration),
            ct.c_bool(fit_atomic_debye_waller),
            ct.c_bool(fit_exv_debye_waller),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_fit_settings")

    def set_grid_settings(
        self, 
        water_scaling: float = 0.01,
        cell_width: float = 1,
        scaling: float = 0.25,
        min_exv_radius: float = 2.15,
        min_bins: int = 0
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.settings_set_grid_settings(
            ct.c_double(water_scaling),
            ct.c_double(cell_width),
            ct.c_double(scaling),
            ct.c_double(min_exv_radius),
            ct.c_uint(min_bins),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_grid_settings")

    def set_hist_settings(
        self, 
        skip_entries: int = 0,
        qmin: float = 1e-4, 
        qmax: float = 0.5, 
        weighted_bins: bool = True
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.settings_set_hist_settings(
            ct.c_uint(skip_entries),
            ct.c_double(qmin),
            ct.c_double(qmax),
            ct.c_bool(weighted_bins),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_hist_settings")

    def set_molecule_settings(
        self, 
        center: bool = True,
        throw_on_unknown_atom: bool = True,
        implicit_hydrogens: bool = True,
        use_occupancy: bool = True,
        exv_table: ExvTable = ExvTable.MinimumFluctutation_implicit_H,
        water_model: WaterModel = WaterModel.radial
    ):
        exv_table = ExvTable.validate(exv_table)
        water_model = WaterModel.validate(water_model)
        ausaxs = AUSAXS()
        status = ct.c_int()
        exv_model_ptr = ct.c_char_p(exv_table.value.encode('utf-8'))
        water_model_ptr = ct.c_char_p(water_model.value.encode('utf-8'))
        ausaxs.lib().functions.settings_set_molecule_settings(
            ct.c_bool(center),
            ct.c_bool(throw_on_unknown_atom),
            ct.c_bool(implicit_hydrogens),
            ct.c_bool(use_occupancy),
            exv_model_ptr,
            water_model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_molecule_settings")
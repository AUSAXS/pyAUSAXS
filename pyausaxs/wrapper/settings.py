from .AUSAXS import AUSAXS, _check_error_code
from .Models import ExvModel, ExvTable, WaterModel
import multiprocessing
import ctypes as ct

# lowercase 'settings' since it's meant to be used with dot-notation
class settings:
    @staticmethod
    def set_exv_settings(exv_model: ExvModel = ExvModel.simple):
        exv_model = ExvModel.validate(exv_model)
        ausaxs = AUSAXS()
        status = ct.c_int()
        model_ptr = ct.c_char_p(exv_model.value.encode('utf-8'))
        ausaxs.lib().functions.set_exv_settings(
            model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_exv_model")

    @staticmethod
    def set_fit_settings(
        fit_hydration: bool = True,
        fit_excluded_volume: bool = False,
        fit_solvent_density: bool = False,
        fit_atomic_debye_waller: bool = False, 
        fit_exv_debye_waller: bool = False,
        max_iterations: int = 100,
        sampled_points: int = 100
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_fit_settings(
            ct.c_uint(sampled_points),
            ct.c_uint(max_iterations),
            ct.c_bool(fit_excluded_volume),
            ct.c_bool(fit_solvent_density),
            ct.c_bool(fit_hydration),
            ct.c_bool(fit_atomic_debye_waller),
            ct.c_bool(fit_exv_debye_waller),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_fit_settings")

    @staticmethod
    def set_grid_settings(
        water_scaling: float = 0.01,
        cell_width: float = 1,
        scaling: float = 0.25,
        min_exv_radius: float = 2.15,
        min_bins: int = 0
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_grid_settings(
            ct.c_double(water_scaling),
            ct.c_double(cell_width),
            ct.c_double(scaling),
            ct.c_double(min_exv_radius),
            ct.c_uint(min_bins),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_grid_settings")

    @staticmethod
    def set_hist_settings(
        skip_entries: int = 0,
        qmin: float = 1e-4, 
        qmax: float = 0.5, 
        weighted_bins: bool = True
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_hist_settings(
            ct.c_uint(skip_entries),
            ct.c_double(qmin),
            ct.c_double(qmax),
            ct.c_bool(weighted_bins),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_hist_settings")

    @staticmethod
    def set_molecule_settings(
        center: bool = True,
        throw_on_unknown_atom: bool = True,
        implicit_hydrogens: bool = True,
        use_occupancy: bool = True,
        exv_table: ExvTable = ExvTable.minimum_fluctutation_implicit_H,
        water_model: WaterModel = WaterModel.radial
    ):
        exv_table = ExvTable.validate(exv_table)
        water_model = WaterModel.validate(water_model)
        ausaxs = AUSAXS()
        status = ct.c_int()
        exv_model_ptr = ct.c_char_p(exv_table.value.encode('utf-8'))
        water_model_ptr = ct.c_char_p(water_model.value.encode('utf-8'))
        ausaxs.lib().functions.set_molecule_settings(
            ct.c_bool(center),
            ct.c_bool(throw_on_unknown_atom),
            ct.c_bool(implicit_hydrogens),
            ct.c_bool(use_occupancy),
            exv_model_ptr,
            water_model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_molecule_settings")

    @staticmethod
    def set_general_settings(
        offline: bool = False,
        verbose: bool = False,
        warnings: bool = True,
        threads: int = multiprocessing.cpu_count()-1
    ):
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_general_settings(
            ct.c_bool(offline),
            ct.c_bool(verbose),
            ct.c_bool(warnings),
            ct.c_uint(threads),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_general_settings")
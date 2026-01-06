from .AUSAXS import AUSAXS, _check_error_code
from .Models import ExvModel, ExvTable, WaterModel
import multiprocessing
import ctypes as ct
from typing import Any

def _type_cast(value: str, type: str):
    """Helper function to cast a string value to the specified type."""
    match type.lower():
        case "int": return int(value)
        case "double": return float(value)
        case "bool": return value.lower() in ("true", "1", "yes")
        case "string": return value
        case _: raise ValueError(f"Unknown setting type: {type}")

# lowercase 'settings' since it's meant to be used with dot-notation
class settings:
    @staticmethod
    def _get(name: str) -> Any:
        """Get a setting by name."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        name_ptr = ct.c_char_p(name.encode('utf-8'))
        value_ptr = ct.POINTER(ct.c_char)()
        type_ptr = ct.POINTER(ct.c_char)()
        tmp_id = ausaxs.lib().functions.get_setting(
            name_ptr,
            ct.byref(value_ptr),
            ct.byref(type_ptr),
            ct.byref(status)
        )
        _check_error_code(status, "settings_get_setting")

        type_str = ct.cast(type_ptr, ct.c_char_p).value.decode('utf-8')
        value_str = ct.cast(value_ptr, ct.c_char_p).value.decode('utf-8')
        ausaxs.deallocate(tmp_id)
        return _type_cast(value_str, type_str)

    @staticmethod
    def _set(name: str, val: str):
        """Set a setting by name and string value."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        name_ptr = ct.c_char_p(name.encode('utf-8'))
        value_ptr = ct.c_char_p(val.encode('utf-8'))
        ausaxs.lib().functions.set_setting(
            name_ptr,
            value_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_setting")

    @staticmethod
    def exv(exv_model: ExvModel = ExvModel.simple):
        """Set the excluded volume model to use in calculations."""
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
    def fit(
        fit_hydration: bool = True,
        fit_excluded_volume: bool = False,
        fit_solvent_density: bool = False,
        # fit_atomic_debye_waller: bool = False, 
        # fit_exv_debye_waller: bool = False,
        # max_iterations: int = 100,
        # sampled_points: int = 100
    ):
        """
        Settings related to model fitting.
        param fit_hydration: Whether to fit the hydration shell parameters.
        param fit_excluded_volume: Whether to fit the excluded volume parameters.
        param fit_solvent_density: Whether to fit the solvent density contrast.
        param max_iterations: Maximum number of fitting iterations.
        param sampled_points: Number of q-points to sample during fitting.
        """
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_fit_settings(
            ct.c_uint(100),     # sampled_points: meaningless for most users
            ct.c_uint(100),     # max_iterations: same
            ct.c_bool(fit_excluded_volume),
            ct.c_bool(fit_solvent_density),
            ct.c_bool(fit_hydration),
            ct.c_bool(False),   # atomic_debye_waller: removed to avoid overfitting
            ct.c_bool(False),   # exv_debye_waller: same
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_fit_settings")

    @staticmethod
    def grid(
        # water_scaling: float = 0.01,
        cell_width: float = 1,
        expansion_factor: float = 0.25,
        min_exv_radius: float = 2.15,
        # min_bins: int = 0
    ):
        """
        Grid settings mostly related to excluded volume calculations.
        param cell_width: The width of each grid cell in Angstroms.
        param scaling: Additional expansion factor relative to the maximal molecular dimensions.
        param min_exv_radius: Minimum radius for expanding every atom in the grid. This directly affects the size of the excluded volume. 
        """
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_grid_settings(
            ct.c_double(0.01),          # water_scaling: meaningless for most users
            ct.c_double(cell_width),
            ct.c_double(expansion_factor),
            ct.c_double(min_exv_radius),
            ct.c_uint(0),               # min_bins: meaningless for most users
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_grid_settings")

    @staticmethod
    def histogram(
        # skip_entries: int = 0,
        qmin: float = 1e-4, 
        qmax: float = 0.5, 
        weighted_bins: bool = True,
        bin_width: float = 0.25,
        bin_count: int = 8000
    ):
        """
        Settings related to histogramming of Debye scattering calculations.
        param qmin: Minimum calculated intensity. 
        param qmax: Maximum calculated intensity. 
        param weighted_bins: Whether to use weighted bins.
        param bin_width: Width of each histogram bin.
        param bin_count: Number of histogram bins.
        """
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.set_hist_settings(
            ct.c_uint(0),           # skip_entries: users can do this themselves
            ct.c_double(qmin),
            ct.c_double(qmax),
            ct.c_bool(weighted_bins),
            ct.c_double(bin_width),
            ct.c_uint(bin_count),
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_hist_settings")

    @staticmethod
    def molecule(
        # center: bool = True,
        throw_on_unknown_atom: bool = True,
        implicit_hydrogens: bool = True,
        use_occupancy: bool = True,
        exv_table: ExvTable = ExvTable.minimum_fluctutation_implicit_H,
        # water_model: WaterModel = WaterModel.radial
    ):
        """
        Settings related to molecule handling. 
        param throw_on_unknown_atom: Whether to throw an error when an unknown atom type is encountered.
        param implicit_hydrogens: Whether to add implicit hydrogens to the molecule. 
        param use_occupancy: Whether to consider atomic occupancy in calculations.
        param exv_table: The excluded volume table to use.
        """
        exv_table = ExvTable.validate(exv_table)
        # water_model = WaterModel.validate(water_model)
        ausaxs = AUSAXS()
        status = ct.c_int()
        exv_model_ptr = ct.c_char_p(exv_table.value.encode('utf-8'))
        # water_model_ptr = ct.c_char_p(water_model.value.encode('utf-8'))
        water_model_ptr = ct.c_char_p(WaterModel.radial.value.encode('utf-8'))
        ausaxs.lib().functions.set_molecule_settings(
            ct.c_bool(True),                    # center: meaningless for most users
            ct.c_bool(throw_on_unknown_atom),
            ct.c_bool(implicit_hydrogens),
            ct.c_bool(use_occupancy),
            exv_model_ptr,
            water_model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "settings_set_molecule_settings")

    @staticmethod
    def general(
        offline: bool = False,
        verbose: bool = False,
        warnings: bool = True,
        threads: int = multiprocessing.cpu_count()-1
    ):
        """
        General settings.
        param offline: Whether to run in offline mode (no internet access). This will disable implicit hydrogen determination for exotic residues. 
        param verbose: Whether to enable verbose output.
        param warnings: Whether to show warnings.
        param threads: Number of threads to use for calculations.
        """
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
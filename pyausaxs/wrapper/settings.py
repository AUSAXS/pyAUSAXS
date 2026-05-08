from .AUSAXS import AUSAXS, _check_error_code
from .Models import ExvModel, ExvTable, WaterModel
import ctypes as ct
from typing import Any

def _type_cast(value: str, type: str):
    """Helper function to cast a string value to the specified type."""
    match type.lower():
        case "int" | "uint": return int(value)
        case "double": return float(value)
        case "bool": return value.lower() in ("true", "1", "yes")
        case "string": return value
        case _: raise ValueError(f"Unknown setting type: {type}")

def _b(val: bool) -> str:
    return "1" if val else "0"

# lowercase 'settings' since it's meant to be used with dot-notation
class settings:
    @staticmethod
    def get(name: str) -> Any:
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
    def set(name: str, val: str):
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
        settings.set("exv_model", ExvModel.validate(exv_model).value)

    @staticmethod
    def fit(
        fit_hydration: bool = None,
        fit_excluded_volume: bool = None,
        fit_solvent_density: bool = None,
    ):
        """
        Settings related to model fitting.
        param fit_hydration: Whether to fit the hydration shell parameters.
        param fit_excluded_volume: Whether to fit the excluded volume parameters.
        param fit_solvent_density: Whether to fit the solvent density contrast.
        """
        if fit_hydration is not None:
            settings.set("hydration", _b(fit_hydration))
        if fit_excluded_volume is not None:
            settings.set("excluded_volume", _b(fit_excluded_volume))
        if fit_solvent_density is not None:
            settings.set("solvent_density", _b(fit_solvent_density))

    @staticmethod
    def grid(
        cell_width: float = None,
        expansion_factor: float = None,
        min_exv_radius: float = None,
    ):
        """
        Grid settings mostly related to excluded volume calculations.
        param cell_width: The width of each grid cell in Angstroms.
        param expansion_factor: Additional expansion factor relative to the maximal molecular dimensions.
        param min_exv_radius: Minimum radius for expanding every atom in the grid. This directly affects the size of the excluded volume. 
        """
        if cell_width is not None:
            settings.set("width", str(cell_width))
        if expansion_factor is not None:
            settings.set("scaling", str(expansion_factor))
        if min_exv_radius is not None:
            settings.set("rvol", str(min_exv_radius))

    @staticmethod
    def histogram(
        qmin: float = None,
        qmax: float = None,
        weighted_bins: bool = None,
        bin_width: float = None,
        bin_count: int = None,
    ):
        """
        Settings related to histogramming of Debye scattering calculations.
        param qmin: Minimum calculated intensity. 
        param qmax: Maximum calculated intensity. 
        param weighted_bins: Whether to use weighted bins.
        param bin_width: Width of each histogram bin.
        param bin_count: Number of histogram bins.
        """
        if qmin is not None:
            settings.set("qmin", str(qmin))
        if qmax is not None:
            settings.set("qmax", str(qmax))
        if weighted_bins is not None:
            settings.set("weighted_bins", _b(weighted_bins))
        if bin_width is not None:
            settings.set("bin_width", str(bin_width))
        if bin_count is not None:
            settings.set("bin_count", str(bin_count))

    @staticmethod
    def molecule(
        throw_on_unknown_atom: bool = None,
        implicit_hydrogens: bool = None,
        use_occupancy: bool = None,
        exv_table: ExvTable = None,
    ):
        """
        Settings related to molecule handling. 
        param throw_on_unknown_atom: Whether to throw an error when an unknown atom type is encountered.
        param implicit_hydrogens: Whether to add implicit hydrogens to the molecule. 
        param use_occupancy: Whether to consider atomic occupancy in calculations.
        param exv_table: The excluded volume table to use.
        """
        if throw_on_unknown_atom is not None:
            settings.set("allow_unknown_atoms", _b(not throw_on_unknown_atom))
        if implicit_hydrogens is not None:
            settings.set("implicit_hydrogens", _b(implicit_hydrogens))
        if use_occupancy is not None:
            settings.set("use_occupancy", _b(use_occupancy))
        if exv_table is not None:
            settings.set("exv_volume", ExvTable.validate(exv_table).value)

    @staticmethod
    def general(
        offline: bool = None,
        verbose: bool = None,
        warnings: bool = None,
        threads: int = None,
    ):
        """
        General settings.
        param offline: Whether to run in offline mode (no internet access). This will disable implicit hydrogen determination for exotic residues. 
        param verbose: Whether to enable verbose output.
        param warnings: Whether to show warnings.
        param threads: Number of threads to use for calculations.
        """
        if offline is not None:
            settings.set("offline", _b(offline))
        if verbose is not None:
            settings.set("verbose", _b(verbose))
        if warnings is not None:
            settings.set("warnings", _b(warnings))
        if threads is not None:
            settings.set("threads", str(threads))
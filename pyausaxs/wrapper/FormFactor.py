from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays
from pyausaxs.signatures import register
import ctypes as ct
import numpy as np

register({
    "ff_valid_form_factor_types": (
        [
            ct.POINTER(ct.POINTER(ct.c_char_p)), # types (output)
            ct.POINTER(ct.c_int),                # n_types (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
    "ff_get_five_gaussian_coefficients": (
        [
            ct.c_char_p,             # form factor type
            ct.POINTER(ct.c_double), # a coefficients (output)
            ct.POINTER(ct.c_double), # b coefficients (output)
            ct.POINTER(ct.c_double), # c coefficient (output)
            ct.POINTER(ct.c_int)     # status (0 = success)
        ],
        None
    ),
    "ff_get_current_exv_volume": (
        [
            ct.c_char_p,             # form factor type
            ct.POINTER(ct.c_double), # volume (output)
            ct.POINTER(ct.c_int)     # status (0 = success)
        ],
        None
    ),
})

class form_factor():
    @staticmethod
    def valid_types() -> np.ndarray:
        """
        Get a list of the valid form factor types.
        """
        ausaxs = AUSAXS()
        types_ptr = ct.POINTER(ct.c_char_p)()
        n_types = ct.c_int()
        status = ct.c_int()
        obj_id = ausaxs.lib().functions.ff_valid_form_factor_types(
            ct.byref(types_ptr),
            ct.byref(n_types),
            ct.byref(status)
        )
        _check_error_code(status, "ff_valid_form_factor_types")
        types = np.array([types_ptr[i].decode('utf-8') for i in range(n_types.value)], dtype=np.str_)
        ausaxs.deallocate(obj_id)
        return types

    @staticmethod
    def get_five_gaussian_coefficients(element: str) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Get the five Gaussian coefficients for a given form factor.
        Returns: a tuple of (a, b, c) where a and b are 5-element numpy arrays and c is a float.
        """
        ausaxs = AUSAXS()
        a = np.empty(5, dtype=np.float64)
        b = np.empty(5, dtype=np.float64)
        c = ct.c_double()
        status = ct.c_int()
        ausaxs.lib().functions.ff_get_five_gaussian_coefficients(
            ct.c_char_p(element.encode('utf-8')),
            a.ctypes.data_as(ct.POINTER(ct.c_double)),
            b.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.byref(c),
            ct.byref(status)
        )
        _check_error_code(status, "ff_get_five_gaussian_coefficients")
        return a, b, c.value

    @staticmethod
    def get_current_exv_volume(element: str) -> float:
        """
        Get the current excluded volume for a given form factor.
        """
        ausaxs = AUSAXS()
        volume = ct.c_double()
        status = ct.c_int()
        ausaxs.lib().functions.ff_get_current_exv_volume(
            ct.c_char_p(element.encode('utf-8')),
            ct.byref(volume),
            ct.byref(status)
        )
        _check_error_code(status, "ff_get_current_exv_volume")
        return volume.value
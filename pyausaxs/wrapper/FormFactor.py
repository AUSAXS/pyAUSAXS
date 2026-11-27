from .AUSAXS import call
from pyausaxs.signatures import register, out, out_array, buffer, STATUS, HOLDER
import ctypes as ct
import numpy as np

register({
    "ff_valid_form_factor_types":        ([out_array("types", ct.c_char_p, n="n"), out("n", ct.c_int), STATUS], HOLDER),
    "ff_get_five_gaussian_coefficients": ([ct.c_char_p, buffer("a", n=5), buffer("b", n=5), out("c", ct.c_double), STATUS], None),
    "ff_get_current_exv_volume":         ([ct.c_char_p, out("volume", ct.c_double), STATUS], None),
})

class form_factor():
    @staticmethod
    def valid_types() -> np.ndarray:
        """
        Get a list of the valid form factor types.
        """
        return call("ff_valid_form_factor_types")

    @staticmethod
    def get_five_gaussian_coefficients(element: str) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Get the five Gaussian coefficients for a given form factor.
        Returns: a tuple of (a, b, c) where a and b are 5-element numpy arrays and c is a float.
        """
        return tuple(call("ff_get_five_gaussian_coefficients", element))

    @staticmethod
    def get_current_exv_volume(element: str) -> float:
        """
        Get the current excluded volume for a given form factor.
        """
        return call("ff_get_current_exv_volume", element)

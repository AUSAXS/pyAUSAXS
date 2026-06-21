from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays, _ptr_to_array
from .Molecule import Molecule
from pyausaxs.signatures import register
import ctypes as ct
import numpy as np

register({
    "molecule_debye_exact": (
        [
            ct.c_int,                            # molecule id
            ct.POINTER(ct.POINTER(ct.c_double)), # q
            ct.POINTER(ct.POINTER(ct.c_double)), # I (output)
            ct.POINTER(ct.c_int),                # n_points
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return obj id
    ),
    "molecule_debye_exact_userq": (
        [
            ct.c_int,                # molecule id
            ct.POINTER(ct.c_double), # q
            ct.POINTER(ct.c_double), # I (output)
            ct.c_int,                # n_points
            ct.POINTER(ct.c_int)     # status (0 = success)
        ],
        None
    ),
})

class unoptimized():
    @staticmethod
    def debye_exact(molecule: Molecule, q_vals: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the exact Debye scattering intensity of the molecule. No form factors will be applied. 
        Warning: This method is _not_ optimized, and may be very slow for large molecules. It is only meant for testing and validation purposes.
        Returns: (q, I)
        """
        ausaxs = AUSAXS()
        if q_vals is not None:
            q = _as_numpy_f64_arrays(q_vals)[0]
            i = np.zeros_like(q, dtype=np.float64)
            n_q = ct.c_int(len(q_vals))
            status = ct.c_int()
            ausaxs.lib().functions.molecule_debye_exact_userq(
                molecule._get_id(),
                q.ctypes.data_as(ct.POINTER(ct.c_double)),
                i.ctypes.data_as(ct.POINTER(ct.c_double)),
                n_q,
                ct.byref(status)
            )
            _check_error_code(status, "unoptimized_debye_exact_q")
            return q, i
        else:
            q_ptr = ct.POINTER(ct.c_double)()
            i_ptr = ct.POINTER(ct.c_double)()
            n_q = ct.c_int()
            status = ct.c_int()
            tmp_id = ausaxs.lib().functions.molecule_debye_exact(
                molecule._get_id(),
                ct.byref(q_ptr),
                ct.byref(i_ptr),
                ct.byref(n_q),
                ct.byref(status)
            )
            _check_error_code(status, "unoptimized_debye_exact")

            n = n_q.value
            q = _ptr_to_array(q_ptr, n)
            i = _ptr_to_array(i_ptr, n)
            ausaxs.deallocate(tmp_id)
            return q, i
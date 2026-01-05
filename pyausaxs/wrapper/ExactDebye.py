from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays
from .Molecule import Molecule
import ctypes as ct
import numpy as np

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
            q = np.array([q_ptr[i] for i in range(n)], dtype=np.float64)
            i = np.array([i_ptr[i] for i in range(n)], dtype=np.float64)
            ausaxs.deallocate(tmp_id)
            return q, i
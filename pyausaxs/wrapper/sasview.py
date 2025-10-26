from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _as_numpy_f64_arrays, _check_similar_length
import ctypes as ct
import numpy as np

class sasview:
    def debye_no_ff(
        q_vector: list[float] | np.ndarray, 
        atom_x: list[float] | np.ndarray, atom_y: list[float] | np.ndarray, atom_z: list[float] | np.ndarray, 
        weights: list[float] | np.ndarray
    ):
        """
        Compute the Debye scattering intensity I(q) for given q values and atomic coordinates.
        No form factors or excluded volume effects are considered; only the pure Debye formula is evaluated. 
        """
        ausaxs = AUSAXS()
        _check_array_inputs(q_vector, atom_x, atom_y, atom_z, weights)
        _check_similar_length(atom_x, atom_y, atom_z, weights, msg="Atomic coordinates and weights must have the same length")
        q_vector, atom_x, atom_y, atom_z, weights = _as_numpy_f64_arrays(q_vector, atom_x, atom_y, atom_z, weights)

        Iq = (ct.c_double * len(q_vector))()
        nq = ct.c_int(len(q_vector))
        nc = ct.c_int(len(weights))
        q = q_vector.ctypes.data_as(ct.POINTER(ct.c_double))
        x = atom_x.ctypes.data_as(ct.POINTER(ct.c_double))
        y = atom_y.ctypes.data_as(ct.POINTER(ct.c_double))
        z = atom_z.ctypes.data_as(ct.POINTER(ct.c_double))
        w = weights.ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        ausaxs.lib().functions.debye_no_ff(q, x, y, z, w, nq, nc, Iq, ct.byref(status))
        _check_error_code(status, "debye")

        return np.ctypeslib.as_array(Iq)
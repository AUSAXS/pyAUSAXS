from integration import AUSAXSLIB
from typing import Union
import ctypes as ct
import numpy as np

class ausaxs:
    """
    AUSAXS Python wrapper for the C++ library.
    """
    def __init__(self):
        self.lib = AUSAXSLIB()
        if not self.lib.is_ready():
            raise RuntimeError("AUSAXS library initialization failed.")

    def debye(
            self, q_vector: Union[list[float], np.ndarray], 
            atom_x: Union[list[float], np.ndarray], atom_y: Union[list[float], np.ndarray], atom_z: Union[list[float], np.ndarray], 
            weights: Union[list[float], np.ndarray]
        ):
        """
        Compute I(q) for a set of points using Debye sums.
        *q_vector* is the q values for the calculation.
        *atom_x* is the x coordinates of the atoms.
        *atom_y* is the y coordinates of the atoms.
        *atom_z* is the z coordinates of the atoms.
        *weights* is the scattering weight of each atom.
        """

        # ensure inputs are numpy arrays and contiguous in memory
        if isinstance(q_vector, list):     q_vector = np.array(q_vector, dtype=np.float64)
        elif q_vector.dtype != np.float64: q_vector = q_vector.astype(np.float64)

        if isinstance(atom_x, list):       atom_x = np.array(atom_x, dtype=np.float64)
        elif atom_x.dtype != np.float64:   atom_x = atom_x.astype(np.float64)

        if isinstance(atom_y, list):       atom_y = np.array(atom_y, dtype=np.float64)
        elif atom_y.dtype != np.float64:   atom_y = atom_y.astype(np.float64)

        if isinstance(atom_z, list):       atom_z = np.array(atom_z, dtype=np.float64)
        elif atom_z.dtype != np.float64:   atom_z = atom_z.astype(np.float64)

        if isinstance(weights, list):      weights = np.array(weights, dtype=np.float64)
        elif weights.dtype != np.float64:  weights = weights.astype(np.float64)

        # prepare ctypes args
        Iq = (ct.c_double * len(q))()
        nq = ct.c_int(len(q))
        nc = ct.c_int(len(w))
        q = q_vector.ctypes.data_as(ct.POINTER(ct.c_double))
        x = atom_x.ctypes.data_as(ct.POINTER(ct.c_double))
        y = atom_y.ctypes.data_as(ct.POINTER(ct.c_double))
        z = atom_z.ctypes.data_as(ct.POINTER(ct.c_double))
        w = weights.ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        self.lib.functions.evaluate_sans_debye(q, x, y, z, w, nq, nc, Iq, ct.byref(status))

        if (status.value == 0):
            return np.array(Iq)
        raise RuntimeError(f"AUSAXS: \"debye\" terminated unexpectedly (error code \"{status.value}\").")
from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _as_numpy_f64_arrays
from .BackendObject import BackendObject
from .Molecule import Molecule
from .Datafile import Datafile
import ctypes as ct
import numpy as np

class IterativeFit(BackendObject):
    """Manual fitting class for step-by-step SAXS fitting control."""

    def __init__(self, mol: Molecule, data: Datafile):
        super().__init__()
        self.ausaxs = AUSAXS()
        status = ct.c_int()
        self._object_id = self.ausaxs._lib.functions.iterative_fit_start(
            mol._object_id,
            data._object_id,
            ct.byref(status)
        )
        _check_error_code(status, "iterative_fit_start")
        self._qsize = len(data.q())

    def step(self, params: np.ndarray | list[float]) -> np.ndarray:
        """Perform one fitting iteration and return the current I(q)."""
        _check_array_inputs(params, names=['params'])
        params_ptr = _as_numpy_f64_arrays(params)[0].ctypes.data_as(ct.POINTER(ct.c_double))
        Iq = (ct.c_double * self._qsize)()
        n_params = ct.c_int(len(params))
        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_step(
            self._object_id,
            params_ptr, 
            n_params,
            Iq,
            ct.byref(status)
        )
        _check_error_code(status, "iterative_fit_step")
        return np.ctypeslib.as_array(Iq)

    def finish(self, params) -> np.ndarray:
        """Finalize the fitting process and return the optimal I(q)."""
        _check_array_inputs(params, names=['params'])
        params_ptr = _as_numpy_f64_arrays(params)[0].ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_finish(params_ptr, ct.byref(status))
        _check_error_code(status, "iterative_fit_finish")

def manual_fit(mol: Molecule, data: Datafile) -> IterativeFit:
    """Create an IterativeFit object for manual fitting control."""
    return IterativeFit(mol, data)
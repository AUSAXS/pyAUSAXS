from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _as_numpy_f64_arrays
from .BackendObject import BackendObject
from .Molecule import Molecule
from .Datafile import Datafile
import ctypes as ct
import numpy as np
from typing import overload

            # # iterative_fit_start
            # self.functions.iterative_fit_start.argtypes = [
            #     ct.c_int,               # molecule id
            #     ct.POINTER(ct.c_double),# q vector to use for fitting
            #     ct.c_int,               # n_points q
            #     ct.POINTER(ct.c_int)    # return status (0 = success)
            # ]
            # self.functions.iterative_fit_start.restype = ct.c_int # return iterative fit id

            # # iterative_fit_step
            # self.functions.iterative_fit_step.argtypes = [
            #     ct.c_int,                           # iterative fit id
            #     ct.POINTER(ct.c_double),            # parameters vector
            #     ct.c_int,                           # number of parameters
            #     ct.POINTER(ct.POINTER(ct.c_double)),# resulting I vector
            #     ct.POINTER(ct.c_int),               # return status (0 = success)
            # ]
            # self.functions.iterative_fit_step.restype = None

class IterativeFit(BackendObject):
    """Manual fitting class for step-by-step SAXS fitting control."""

    @overload
    def __init__(self, mol: Molecule, q_vals: list[float] | np.ndarray): ...
    @overload
    def __init__(self, mol: Molecule, data: Datafile): ...

    _qsize: int = 0
    def __init__(self, mol: Molecule, arg: list[float] | np.ndarray | Datafile = None):
        super().__init__()
        self.ausaxs = AUSAXS()
        if isinstance(arg, Datafile):
            arg = arg.q()
        if arg is not None:
            q_vals = _as_numpy_f64_arrays(arg)[0]
            status = ct.c_int()
            self._object_id = self.ausaxs._lib.functions.iterative_fit_init_userq(
                mol._object_id,
                q_vals.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(len(q_vals)),
                ct.byref(status)
            )
            _check_error_code(status, "iterative_fit_init_userq")
            self._qsize = len(q_vals)
        else:
            nq = ct.c_int()
            status = ct.c_int()
            self._object_id = self.ausaxs._lib.functions.iterative_fit_init(
                mol._object_id,
                ct.byref(nq),
                ct.byref(status)
            )
            _check_error_code(status, "iterative_fit_init")
            self._qsize = nq.value

    def evaluate(self, params: np.ndarray | list[float]) -> np.ndarray:
        """Perform one fitting iteration and return the current I(q)."""
        _check_array_inputs(params)
        params_array = _as_numpy_f64_arrays(params)[0]
        out_ptr = ct.POINTER(ct.c_double)()
        n_params = ct.c_int(len(params_array))
        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_evaluate(
            self._object_id,
            params_array.ctypes.data_as(ct.POINTER(ct.c_double)),
            n_params,
            ct.byref(out_ptr),
            ct.byref(status)
        )
        _check_error_code(status, "iterative_fit_evaluate")
        return np.ctypeslib.as_array(out_ptr, shape=(self._qsize,))

@overload
def manual_fit(mol: Molecule, q_vals: list[float] | np.ndarray) -> IterativeFit: ...
@overload
def manual_fit(mol: Molecule, data: Datafile) -> IterativeFit: ...

def manual_fit(mol: Molecule, arg) -> IterativeFit:
    """Start a fitting session with manual control over the fitting session."""
    return IterativeFit(mol, arg)
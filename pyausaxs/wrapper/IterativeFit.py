from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _as_numpy_f64_arrays, _ptr_to_array
from .BackendObject import BackendObject
from .Molecule import Molecule
from .Datafile import Datafile
from pyausaxs.signatures import register
import ctypes as ct
import numpy as np
from typing import overload

register({
    "iterative_fit_init": (
        [
            ct.c_int,            # molecule id
            ct.POINTER(ct.c_int) # return status (0 = success)
        ],
        ct.c_int                 # return iterative fit id
    ),
    "iterative_fit_init_userq": (
        [
            ct.c_int,               # molecule id
            ct.POINTER(ct.c_double),# q vector to use for fitting
            ct.c_int,               # n_points q
            ct.POINTER(ct.c_int)    # return status (0 = success)
        ],
        ct.c_int                    # return iterative fit id
    ),
    "iterative_fit_evaluate": (
        [
            ct.c_int,                            # iterative fit id
            ct.POINTER(ct.c_double),             # parameters vector
            ct.c_int,                            # number of parameters
            ct.POINTER(ct.POINTER(ct.c_double)), # resulting I vector
            ct.POINTER(ct.c_int),                # number of points in resulting I vector
            ct.POINTER(ct.c_int),                # return status (0 = success)
        ],
        None
    ),
    "iterative_fit_evaluate_userq": (
        [
            ct.c_int,                # iterative fit id
            ct.POINTER(ct.c_double), # parameters vector
            ct.c_int,                # number of parameters
            ct.POINTER(ct.c_double), # q vector to evaluate
            ct.POINTER(ct.c_double), # resulting I vector
            ct.c_int,                # number of points in q vector
            ct.POINTER(ct.c_int),    # return status (0 = success)
        ],
        None
    ),
})

class IterativeFit(BackendObject):
    """Manual fitting class for step-by-step SAXS fitting control."""

    @overload
    def __init__(self, mol: Molecule, q_vals: list[float] | np.ndarray): ...
    @overload
    def __init__(self, mol: Molecule, data: Datafile): ...
    def __init__(self, mol: Molecule, arg: list[float] | np.ndarray | Datafile = None): # type: ignore[reportInconsistentOverload]
        super().__init__()
        self.ausaxs = AUSAXS()
        if isinstance(arg, Datafile):
            arg = arg.q()
        if arg is not None:
            q_vals = _as_numpy_f64_arrays(arg)[0]
            status = ct.c_int()
            self._set_id(self.ausaxs._lib.functions.iterative_fit_init_userq(
                mol._get_id(),
                q_vals.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(len(q_vals)),
                ct.byref(status)
            ))
            _check_error_code(status, "iterative_fit_init_userq")
        else:
            status = ct.c_int()
            self._set_id(self.ausaxs._lib.functions.iterative_fit_init(
                mol._get_id(),
                ct.byref(status)
            ))
            _check_error_code(status, "iterative_fit_init")

    @overload
    def evaluate(self, params: np.ndarray | list[float] | tuple) -> np.ndarray: ...
    @overload
    def evaluate(self, params: np.ndarray | list[float] | tuple, q: np.ndarray | list[float]) -> np.ndarray: ...

    def evaluate(self, params: np.ndarray | list[float] | tuple, q: np.ndarray | list[float] = None) -> np.ndarray: # type: ignore[reportInconsistentOverload]
        """Perform one fitting iteration and return the current I(q)."""
        _check_array_inputs(params)
        params_array = _as_numpy_f64_arrays(params)[0]
        if q is not None:
            _check_array_inputs(q)
            q_array = _as_numpy_f64_arrays(q)[0]
            out_ptr = np.zeros(len(q_array), dtype=np.float64)
            status = ct.c_int()
            self.ausaxs._lib.functions.iterative_fit_evaluate_userq(
                self._get_id(),
                params_array.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(len(params_array)),
                q_array.ctypes.data_as(ct.POINTER(ct.c_double)),
                out_ptr.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(len(q_array)),
                ct.byref(status)
            )
            _check_error_code(status, "iterative_fit_evaluate_userq")
            return out_ptr

        else:
            out_ptr = ct.POINTER(ct.c_double)()
            out_n = ct.c_int()
            status = ct.c_int()
            self.ausaxs._lib.functions.iterative_fit_evaluate(
                self._get_id(),
                params_array.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(len(params_array)),
                ct.byref(out_ptr),
                ct.byref(out_n),
                ct.byref(status)
            )
            _check_error_code(status, "iterative_fit_evaluate")
            return _ptr_to_array(out_ptr, out_n.value)

@overload
def manual_fit(mol: Molecule, q_vals: list[float] | np.ndarray) -> IterativeFit: ...
@overload
def manual_fit(mol: Molecule, data: Datafile) -> IterativeFit: ...
@overload
def manual_fit(mol: Molecule) -> IterativeFit: ...

def manual_fit(mol: Molecule, arg=None) -> IterativeFit: # type: ignore[reportInconsistentOverload]
    """Start a fitting session with manual control over the fitting session."""
    return IterativeFit(mol, arg) if arg is not None else IterativeFit(mol)
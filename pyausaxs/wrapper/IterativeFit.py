from .AUSAXS import call, _check_array_inputs
from .BackendObject import BackendObject
from .Molecule import Molecule
from .Datafile import Datafile
from pyausaxs.signatures import register, array, length, out, out_array, buffer, STATUS
import ctypes as ct
import numpy as np
from typing import overload

register({
    "iterative_fit_init":       ([ct.c_int, STATUS], ct.c_int),
    "iterative_fit_init_userq": ([ct.c_int, array("q"), length("q"), STATUS], ct.c_int),
    "iterative_fit_evaluate": (
        [ct.c_int, array("params"), length("params"), out_array("I", ct.c_double, n="n"), out("n", ct.c_int), STATUS],
        None
    ),
    "iterative_fit_evaluate_userq": (
        [ct.c_int, array("params"), length("params"), array("q"), buffer("I", n="q"), length("q"), STATUS],
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
        if isinstance(arg, Datafile):
            arg = arg.q()
        if arg is not None:
            self._set_id(call("iterative_fit_init_userq", mol._get_id(), arg))
        else:
            self._set_id(call("iterative_fit_init", mol._get_id()))

    @overload
    def evaluate(self, params: np.ndarray | list[float] | tuple) -> np.ndarray: ...
    @overload
    def evaluate(self, params: np.ndarray | list[float] | tuple, q: np.ndarray | list[float]) -> np.ndarray: ...

    def evaluate(self, params: np.ndarray | list[float] | tuple, q: np.ndarray | list[float] = None) -> np.ndarray: # type: ignore[reportInconsistentOverload]
        """Perform one fitting iteration and return the current I(q)."""
        _check_array_inputs(params)
        if q is not None:
            _check_array_inputs(q)
            return call("iterative_fit_evaluate_userq", self._get_id(), params, q)
        return call("iterative_fit_evaluate", self._get_id(), params)

@overload
def manual_fit(mol: Molecule, q_vals: list[float] | np.ndarray) -> IterativeFit: ...
@overload
def manual_fit(mol: Molecule, data: Datafile) -> IterativeFit: ...
@overload
def manual_fit(mol: Molecule) -> IterativeFit: ...

def manual_fit(mol: Molecule, arg=None) -> IterativeFit: # type: ignore[reportInconsistentOverload]
    """Start a fitting session with manual control over the fitting session."""
    return IterativeFit(mol, arg) if arg is not None else IterativeFit(mol)
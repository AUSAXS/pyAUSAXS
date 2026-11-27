from .AUSAXS import call, _check_array_inputs, _check_similar_length
from pyausaxs.signatures import register, array, length, buffer, STATUS
import numpy as np

register({
    "debye_no_ff": (
        [array("q"), array("x"), array("y"), array("z"), array("w"), length("q"), length("w"), buffer("I", n="q"), STATUS],
        None
    ),
})

class sasview:
    @staticmethod
    def debye_no_ff(
        q_vector: list[float] | np.ndarray, 
        atom_x: list[float] | np.ndarray, atom_y: list[float] | np.ndarray, atom_z: list[float] | np.ndarray, 
        weights: list[float] | np.ndarray
    ):
        """
        Compute the Debye scattering intensity I(q) for given q values and atomic coordinates.
        No form factors or excluded volume effects are considered; only the pure Debye formula is evaluated. 
        """
        _check_array_inputs(q_vector, atom_x, atom_y, atom_z, weights)
        _check_similar_length(atom_x, atom_y, atom_z, weights, msg="Atomic coordinates and weights must have the same length")
        return call("debye_no_ff", q_vector, atom_x, atom_y, atom_z, weights)

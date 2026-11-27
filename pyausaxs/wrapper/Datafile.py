from .AUSAXS import call, _check_array_inputs, _check_similar_length
from .BackendObject import BackendObject
from pyausaxs.signatures import register, array, length, out, out_array, STATUS, HOLDER
from typing import overload
import ctypes as ct
import numpy as np

register({
    "data_read":     ([ct.c_char_p, STATUS], ct.c_int),
    "data_create":   ([array("q"), array("I"), array("Ierr"), length("q"), STATUS], ct.c_int),
    "data_get_data": (
        [ct.c_int, *(out_array(name, ct.c_double, n="n") for name in ("q", "I", "Ierr")), out("n", ct.c_int), STATUS],
        HOLDER
    ),
})

class Datafile(BackendObject):
    __slots__ = ['_data']

    @overload
    def __init__(self, filename: str): ...
    @overload
    def __init__(self, q: list[float] | np.ndarray, I: list[float] | np.ndarray, Ierr: list[float] | np.ndarray): ...
    def __init__(self, *args): # type: ignore[reportInconsistentOverload]
        super().__init__()
        self._data: dict[str, np.ndarray] = {}

        def init_filename(filename: str):
            self._read_data(filename)

        def init_arrays(q: list[float] | np.ndarray, I: list[float] | np.ndarray, Ierr: list[float] | np.ndarray):
            _check_array_inputs(q, I, Ierr, names=['q', 'I', 'Ierr'])
            _check_similar_length(q, I, Ierr, msg="q, I, and Ierr must have the same length")
            # register a backend dataset so the object can be passed to fit(), just like one read from a file;
            # the arrays are then read back from it, so both constructors report the same (q-range limited) points
            self._object_id = call("data_create", q, I, Ierr)

        if len(args) == 1 and isinstance(args[0], str):
            init_filename(args[0])
        elif len(args) == 3:
            init_arrays(args[0], args[1], args[2])
        else:
            raise TypeError("Datafile constructor accepts either a filename (str) or three arrays (q, I, Ierr).")

    def _read_data(self, filename: str) -> None:
        self._object_id = call("data_read", filename)

    def _get_data(self) -> None:
        if self._data: return
        self._data = call("data_get_data", self._get_id())._asdict()

    def q(self) -> np.ndarray:
        """Get the q-vector (scattering vector) as numpy array."""
        self._get_data()
        return self._data['q']

    def I(self) -> np.ndarray:
        """Get the scattering intensity as numpy array."""
        self._get_data()
        return self._data['I']

    def Ierr(self) -> np.ndarray:
        """Get the intensity error values as numpy array."""
        self._get_data()
        return self._data['Ierr']

    def dict(self) -> dict[str, np.ndarray]:
        """Get all data arrays as a dictionary with keys: 'q', 'I', 'Ierr'."""
        self._get_data()
        return self._data

    def data(self) -> list[np.ndarray]:
        """Get all data arrays as a list of numpy arrays: (q, I, Ierr)."""
        self._get_data()
        return [self._data['q'], self._data['I'], self._data['Ierr']]

def read_data(filename: str) -> Datafile:
    """Read a data file and return a DataFile object."""
    return Datafile(filename)

def create_datafile(q: list[float] | np.ndarray, I: list[float] | np.ndarray, Ierr: list[float] | np.ndarray) -> Datafile:
    """
    Create a DataFile object from given q, I, and Ierr arrays.
    q must be in [1/A]. Points outside the configured q-range are removed, exactly as when reading a file.
    """
    return Datafile(q, I, Ierr)
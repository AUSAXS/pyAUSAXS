from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _as_numpy_f64_arrays, _check_similar_length
from .BackendObject import BackendObject
from typing import overload
import ctypes as ct
import numpy as np

class Datafile(BackendObject):
    __slots__ = ['_data']

    @overload
    def __init__(self, filename: str): ...
    @overload
    def __init__(self, q: list[float] | np.ndarray, I: list[float] | np.ndarray, Ierr: list[float] | np.ndarray): ...
    def __init__(self, *args):
        super().__init__()
        self._data: dict[str, np.ndarray] = {}

        def init_filename(filename: str):
            self._read_data(filename)

        def init_arrays(q: list[float] | np.ndarray, I: list[float] | np.ndarray, Ierr: list[float] | np.ndarray):
            _check_array_inputs(q, I, Ierr, names=['q', 'I', 'Ierr'])
            _check_similar_length(q, I, Ierr, msg="q, I, and Ierr must have the same length")
            q_arr, I_arr, Ierr_arr = _as_numpy_f64_arrays(q, I, Ierr)
            self._data['q'] = q_arr
            self._data['I'] = I_arr
            self._data['Ierr'] = Ierr_arr

        if len(args) == 1 and isinstance(args[0], str):
            init_filename(args[0])
        elif len(args) == 3:
            init_arrays(args[0], args[1], args[2])
        else:
            raise TypeError("Datafile constructor accepts either a filename (str) or three arrays (q, I, Ierr).")

    def _read_data(self, filename: str) -> None:
        ausaxs = AUSAXS()
        filename_c = filename.encode('utf-8')
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.data_read(
            filename_c,
            ct.byref(status)
        )
        _check_error_code(status, "read_data")

    def _get_data(self) -> None:
        if self._data: return
        ausaxs = AUSAXS()
        q_ptr = ct.POINTER(ct.c_double)()
        I_ptr = ct.POINTER(ct.c_double)()
        Ierr_ptr = ct.POINTER(ct.c_double)()
        n_points = ct.c_int()
        status = ct.c_int()

        data_id = ausaxs.lib().functions.data_get_data(
            self._object_id,
            ct.byref(q_ptr),
            ct.byref(I_ptr),
            ct.byref(Ierr_ptr),
            ct.byref(n_points),
            ct.byref(status)
        )
        _check_error_code(status, "data_get_data")

        n = n_points.value
        self._data["q"]    = np.array([q_ptr[i] for i in range(n)],    dtype=np.float64)
        self._data["I"]    = np.array([I_ptr[i] for i in range(n)],    dtype=np.float64)
        self._data["Ierr"] = np.array([Ierr_ptr[i] for i in range(n)], dtype=np.float64)
        ausaxs.deallocate(data_id)

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
    """Create a DataFile object from given q, I, and Ierr arrays."""
    return Datafile(q, I, Ierr)
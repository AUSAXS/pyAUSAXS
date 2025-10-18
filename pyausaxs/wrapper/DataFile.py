from .AUSAXS import AUSAXS, _check_error_code
import ctypes as ct
import numpy as np

class DataFile:
    def __init__(self, filename: str):
        self._object_id: ct.c_int32 = None
        self._data_id: ct.c_int32 = None
        self._data: dict[str, np.ndarray] = {}
        self._read_data(filename)

    def __del__(self):        
        ausaxs = AUSAXS()
        ausaxs.deallocate(self._object_id)
        if self._data_id: ausaxs.deallocate(self._data_id)

    def _read_data(self, filename: str) -> None:
        ausaxs = AUSAXS()
        filename_c = filename.encode('utf-8')
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.data_read(
            filename_c,
            ct.byref(status)
        )
        _check_error_code(status, "read_data")

    def _get_data(self):
        ausaxs = AUSAXS()
        if self._data_id: ausaxs.deallocate(self._data_id)

        q_ptr = ct.POINTER(ct.c_double)()
        I_ptr = ct.POINTER(ct.c_double)()
        Ierr_ptr = ct.POINTER(ct.c_double)()
        n_points = ct.c_int()
        status = ct.c_int()

        self._data_id = ausaxs.lib().functions.data_get_data(
            self._object_id,
            ct.byref(q_ptr),
            ct.byref(I_ptr),
            ct.byref(Ierr_ptr),
            ct.byref(n_points),
            ct.byref(status)
        )
        _check_error_code(status, "data_get_data")

        n = n_points.value
        self._data["q"] = [q_ptr[i] for i in range(n)]
        self._data["I"] = [I_ptr[i] for i in range(n)]
        self._data["Ierr"] = [Ierr_ptr[i] for i in range(n)]

    def q(self) -> np.ndarray:
        """Get q-vector (scattering vector) as numpy array."""
        if not self._data_id: self._get_data()
        return np.array(self._data['q'], dtype=np.float64)

    def I(self) -> np.ndarray:
        """Get scattering intensity as numpy array."""
        if not self._data_id: self._get_data()
        return np.array(self._data['I'], dtype=np.float64)

    def Ierr(self) -> np.ndarray:
        """Get intensity error values as numpy array."""
        if not self._data_id: self._get_data()
        return np.array(self._data['Ierr'], dtype=np.float64)

def read_data(filename: str) -> DataFile:
    """Read a data file and return a DataFile object."""
    return DataFile(filename)
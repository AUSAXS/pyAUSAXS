from .AUSAXS import AUSAXS, _check_error_code
import ctypes as ct
import numpy as np

class PDBFile:
    def __init__(self, filename: str):
        self._object_id: ct.c_int32 = None
        self._data_id: ct.c_int32 = None
        self._data: dict[str, np.ndarray] = {}
        self._read_pdb(filename)

    def __del__(self):
        ausaxs = AUSAXS()
        ausaxs.deallocate(self._object_id)
        if self._data_id: ausaxs.deallocate(self._data_id)

    def _read_pdb(self, filename: str) -> None:
        """Read a pdb (or cif) data file"""
        ausaxs = AUSAXS()
        filename_c = filename.encode('utf-8')
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.pdb_read(
            filename_c,
            ct.byref(status)
        )
        _check_error_code(status, "read_pdb")

    def _get_data(self):
        # AUSAXS stores the PDB data in a manner which is hard to access directly.
        # This call therefore constructs a new data object and links the pointers to it.
        # We therefore store the data object id separately, so we can deallocate it later.
        ausaxs = AUSAXS()
        if self._data_id: ausaxs.deallocate(self._data_id)

        serial_ptr = ct.POINTER(ct.c_int)()
        name_ptr = ct.POINTER(ct.c_char_p)()
        altLoc_ptr = ct.POINTER(ct.c_char_p)()
        resName_ptr = ct.POINTER(ct.c_char_p)()
        chainID_ptr = ct.POINTER(ct.c_char)()
        resSeq_ptr = ct.POINTER(ct.c_int)()
        iCode_ptr = ct.POINTER(ct.c_char_p)()
        x_ptr = ct.POINTER(ct.c_double)()
        y_ptr = ct.POINTER(ct.c_double)()
        z_ptr = ct.POINTER(ct.c_double)()
        occupancy_ptr = ct.POINTER(ct.c_double)()
        tempFactor_ptr = ct.POINTER(ct.c_double)()
        element_ptr = ct.POINTER(ct.c_char_p)()
        charge_ptr = ct.POINTER(ct.c_char_p)()
        n_atoms = ct.c_int()
        status = ct.c_int()

        self._data_id = ausaxs.lib().functions.pdb_get_data(
            self._object_id,
            ct.byref(serial_ptr),
            ct.byref(name_ptr), 
            ct.byref(altLoc_ptr),
            ct.byref(resName_ptr),
            ct.byref(chainID_ptr),
            ct.byref(resSeq_ptr),
            ct.byref(iCode_ptr),
            ct.byref(x_ptr),
            ct.byref(y_ptr),
            ct.byref(z_ptr),
            ct.byref(occupancy_ptr),
            ct.byref(tempFactor_ptr),
            ct.byref(element_ptr),
            ct.byref(charge_ptr),
            ct.byref(n_atoms),
            ct.byref(status)
        )
        _check_error_code(status, "pdb_get_data")

        n = n_atoms.value
        self._data["serial"] = [serial_ptr[i] for i in range(n)]
        self._data["name"] = [name_ptr[i].decode('utf-8') for i in range(n)]
        self._data["altLoc"] = [altLoc_ptr[i].decode('utf-8') for i in range(n)]
        self._data["resName"] = [resName_ptr[i].decode('utf-8') for i in range(n)]
        self._data["chainID"] = [chainID_ptr[i].decode('utf-8') for i in range(n)]
        self._data["resSeq"] = [resSeq_ptr[i] for i in range(n)]
        self._data["iCode"] = [iCode_ptr[i].decode('utf-8') for i in range(n)]
        self._data["x"] = [x_ptr[i] for i in range(n)]
        self._data["y"] = [y_ptr[i] for i in range(n)]
        self._data["z"] = [z_ptr[i] for i in range(n)]
        self._data["occupancy"] = [occupancy_ptr[i] for i in range(n)]
        self._data["tempFactor"] = [tempFactor_ptr[i] for i in range(n)]
        self._data["element"] = [element_ptr[i].decode('utf-8') for i in range(n)]
        self._data["charge"] = [charge_ptr[i].decode('utf-8') for i in range(n)]

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get atomic coordinates as numpy arrays."""
        if not self._data_id: self._get_data()
        return (
            np.array(self._data['x'], dtype=np.float64),
            np.array(self._data['y'], dtype=np.float64),
            np.array(self._data['z'], dtype=np.float64)
        )

def read_pdb(filename: str) -> PDBFile:
    """Convenience function to read a PDB file and return a PDBFile instance."""
    return PDBFile(filename)
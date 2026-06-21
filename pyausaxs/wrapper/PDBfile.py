from .AUSAXS import AUSAXS, _check_error_code, _ptr_to_array, _ptr_to_str_array
from .BackendObject import BackendObject
from .Models import ExvModel
from .Datafile import Datafile
from .FitResult import FitResult
from pyausaxs.signatures import register
import ctypes as ct
import numpy as np

register({
    "pdb_read": (
        [
            ct.c_char_p,         # filename
            ct.POINTER(ct.c_int) # status (0 = success)
        ],
        ct.c_int                 # return pdb id
    ),
    "pdb_get_data": (
        [
            ct.c_int,                            # object id
            ct.POINTER(ct.POINTER(ct.c_int)),    # serial (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # name (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # altLoc (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # resName (output)
            ct.POINTER(ct.POINTER(ct.c_char)),   # chainID (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # resSeq (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # iCode (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # x (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # y (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # z (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # occupancy (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # tempFactor (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # element (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # charge (output)
            ct.POINTER(ct.c_int),                # n_atoms (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
    "pdb_debye_fit": (
        [
            ct.c_int,            # pdb id
            ct.c_int,            # data id
            ct.POINTER(ct.c_int) # status (0 = success)
        ],
        None
    ),
})

class PDBfile(BackendObject):
    def __init__(self, filename: str):
        super().__init__()
        self._data: dict[str, np.ndarray] = {}
        self._read_pdb(filename)

    def _read_pdb(self, filename: str) -> None:
        """Read a pdb (or cif) data file"""
        ausaxs = AUSAXS()
        filename_c = filename.encode('utf-8')
        status = ct.c_int()
        self._set_id(ausaxs.lib().functions.pdb_read(
            filename_c,
            ct.byref(status)
        ))
        _check_error_code(status, "read_pdb")

    def _get_data(self) -> None:
        if self._data: return
        ausaxs = AUSAXS()
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

        data_id = ausaxs.lib().functions.pdb_get_data(
            self._get_id(),
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
        self._data["serial"]     = _ptr_to_array(serial_ptr, n, dtype=np.int32)
        self._data["name"]       = _ptr_to_str_array(name_ptr, n)
        self._data["altLoc"]     = _ptr_to_str_array(altLoc_ptr, n)
        self._data["resName"]    = _ptr_to_str_array(resName_ptr, n)
        self._data["chainID"]    = _ptr_to_str_array(chainID_ptr, n)
        self._data["resSeq"]     = _ptr_to_array(resSeq_ptr, n, dtype=np.int32)
        self._data["iCode"]      = _ptr_to_str_array(iCode_ptr, n)
        self._data["x"]          = _ptr_to_array(x_ptr, n)
        self._data["y"]          = _ptr_to_array(y_ptr, n)
        self._data["z"]          = _ptr_to_array(z_ptr, n)
        self._data["occupancy"]  = _ptr_to_array(occupancy_ptr, n)
        self._data["tempFactor"] = _ptr_to_array(tempFactor_ptr, n)
        self._data["element"]    = _ptr_to_str_array(element_ptr, n)
        self._data["charge"]     = _ptr_to_str_array(charge_ptr, n)
        ausaxs.deallocate(data_id)

    def fit(self, data: Datafile, model: ExvModel | str = ExvModel.simple) -> FitResult:
        """
        Fit the Debye scattering intensity of the PDB data to the provided data.
        Returns: chi-squared value of the fit.
        """
        ausaxs = AUSAXS()
        model_ptr = ct.c_char_p(model.value.encode('utf-8'))
        status = ct.c_int()
        res_id = ausaxs.lib().functions.pdb_debye_fit(
            self._get_id(),
            data._get_id(),
            model_ptr,
            ct.byref(status)
        )
        _check_error_code(status, "pdb_fit")
        return FitResult(res_id)

    def serial(self) -> np.ndarray:
        """Get atom serial numbers as numpy array."""
        self._get_data()
        return self._data['serial']

    def names(self) -> np.ndarray:
        """Get atom names as numpy array."""
        self._get_data()
        return self._data['name']

    def resnames(self) -> np.ndarray:
        """Get residue names as numpy array."""
        self._get_data()
        return self._data['resName']

    def chain_ids(self) -> np.ndarray:
        """Get chain IDs as numpy array."""
        self._get_data()
        return self._data['chainID']

    def res_seqs(self) -> np.ndarray:
        """Get residue sequence numbers as numpy array."""
        self._get_data()
        return self._data['resSeq']

    def icodes(self) -> np.ndarray:
        """Get insertion codes as numpy array."""
        self._get_data()
        return self._data['iCode']

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get atomic coordinates as numpy arrays: (x, y, z)."""
        self._get_data()
        return (
            self._data['x'],
            self._data['y'],
            self._data['z']
        )

    def occupancies(self) -> np.ndarray:
        """Get atomic occupancies as numpy array."""
        self._get_data()
        return self._data['occupancy']

    def temp_factors(self) -> np.ndarray:
        """Get atomic temperature factors as numpy array."""
        self._get_data()
        return self._data['tempFactor']

    def elements(self) -> np.ndarray:
        """Get atomic elements as numpy array."""
        self._get_data()
        return self._data['element']

    def charges(self) -> np.ndarray:
        """Get atomic charges as numpy array."""
        self._get_data()
        return self._data['charge']

    def dict(self) -> dict[str, np.ndarray]:
        """Get all parsed PDB data as a dictionary of numpy arrays."""
        self._get_data()
        return self._data

    def data(self) -> list[np.ndarray]:
        """
        Get all parsed PDB data as a list of numpy arrays
        (serial, name, altloc, resname, chain_id, resseq, icode, x, y, z, occupancy, tempFactor, element, charge).
        """
        self._get_data()
        return [
            self._data['serial'],
            self._data['name'],
            self._data['altLoc'],
            self._data['resName'],
            self._data['chainID'],
            self._data['resSeq'],
            self._data['iCode'],
            self._data['x'],
            self._data['y'],
            self._data['z'],
            self._data['occupancy'],
            self._data['tempFactor'],
            self._data['element'],
            self._data['charge']
        ]

def read_pdb(filename: str) -> PDBfile:
    """Convenience function to read a PDB file and return a PDBFile instance."""
    return PDBfile(filename)
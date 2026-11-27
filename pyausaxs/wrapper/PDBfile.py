from .AUSAXS import call
from .BackendObject import BackendObject
from pyausaxs.signatures import register, out, out_array, STATUS, HOLDER
import ctypes as ct
import numpy as np

_FIELDS = [
    ("serial", ct.c_int), ("name", ct.c_char_p), ("altLoc", ct.c_char_p), ("resName", ct.c_char_p), ("chainID", ct.c_char),
    ("resSeq", ct.c_int), ("iCode", ct.c_char_p), ("x", ct.c_double), ("y", ct.c_double), ("z", ct.c_double),
    ("occupancy", ct.c_double), ("tempFactor", ct.c_double), ("element", ct.c_char_p), ("charge", ct.c_char_p),
]

register({
    "pdb_read":     ([ct.c_char_p, STATUS], ct.c_int),
    "pdb_get_data": ([ct.c_int, *(out_array(name, t, n="n") for name, t in _FIELDS), out("n", ct.c_int), STATUS], HOLDER),
})

class PDBfile(BackendObject):
    def __init__(self, filename: str):
        super().__init__()
        self._data: dict[str, np.ndarray] = {}
        self._read_pdb(filename)

    def _read_pdb(self, filename: str) -> None:
        """Read a pdb (or cif) data file"""
        self._set_id(call("pdb_read", filename))

    def _get_data(self) -> None:
        if self._data: return
        self._data = call("pdb_get_data", self._get_id())._asdict()

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
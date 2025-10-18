from .AUSAXS import AUSAXS, _check_error_code
from .PDBfile import PDBfile
import ctypes as ct
import numpy as np
from typing import overload

class Molecule:
    def __init__(self, *args):
        self._object_id: int = None
        self._atom_data: dict[str, np.ndarray] = {}
        self._water_data: dict[str, np.ndarray] = {}
        self._create_molecule(*args)

    def __del__(self):        
        ausaxs = AUSAXS()
        ausaxs.deallocate(self._object_id)

    def _create_molecule_from_file(self, filename: str) -> None:
        ausaxs = AUSAXS()
        filename_c = filename.encode('utf-8')
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.molecule_from_file(
            filename_c,
            ct.byref(status)
        )
        _check_error_code(status, "_create_molecule_from_file")

    def _create_molecule_from_pdb(self, pdb: PDBfile) -> None:
        if not isinstance(pdb, PDBfile):
            raise TypeError(f"pdb must be of type PDBfile, got {type(pdb)} instead.")
        if pdb._object_id is None:
            raise ValueError("PDBfile object is not properly initialized.")
        ausaxs = AUSAXS()
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.molecule_from_pdb_id(
            pdb._object_id,
            ct.byref(status)
        )
        _check_error_code(status, "_create_molecule_from_pdb")

    def _create_molecule_from_arrays(
        self, x: np.ndarray | list[float], y: np.ndarray | list[float], z: np.ndarray | list[float], weights: np.ndarray | list[float]
    ) -> None:
        ausaxs = AUSAXS()
        n_atoms = ct.c_int(len(x))
        x_c = x.astype(np.float64)
        y_c = y.astype(np.float64)
        z_c = z.astype(np.float64)
        weights_c = weights.astype(np.float64)
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.molecule_from_arrays(
            x_c.ctypes.data_as(ct.POINTER(ct.c_double)),
            y_c.ctypes.data_as(ct.POINTER(ct.c_double)),
            z_c.ctypes.data_as(ct.POINTER(ct.c_double)),
            weights_c.ctypes.data_as(ct.POINTER(ct.c_double)),
            n_atoms,
            ct.byref(status)
        )
        _check_error_code(status, "_create_molecule_from_arrays")

    def _create_molecule(self, *args) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            self._create_molecule_from_file(args[0])
        elif len(args) == 1 and isinstance(args[0], PDBfile):
            self._create_molecule_from_pdb(args[0])
        elif len(args) == 4:
            self._create_molecule_from_arrays(args[0], args[1], args[2], args[3])
        else:
            raise TypeError(
                "Invalid arguments to create Molecule. " \
                "Expected (filename: str), (pdb: PDBfile), or (x: array, y: array, z: array, weights: array)."
            )

    def _get_data(self) -> None:
        if self._atom_data: return
        ausaxs = AUSAXS()
        ax_ptr = ct.POINTER(ct.c_double)()
        ay_ptr = ct.POINTER(ct.c_double)()
        az_ptr = ct.POINTER(ct.c_double)()
        aw_ptr = ct.POINTER(ct.c_double)()
        aff_ptr = ct.POINTER(ct.c_char_p)()
        wx_ptr = ct.POINTER(ct.c_double)()
        wy_ptr = ct.POINTER(ct.c_double)()
        wz_ptr = ct.POINTER(ct.c_double)()
        ww_ptr = ct.POINTER(ct.c_double)()
        n_atoms = ct.c_int()
        n_weights = ct.c_int()
        status = ct.c_int()

        data_id = ausaxs.lib().functions.molecule_get_data(
            self._object_id,
            ct.byref(ax_ptr),
            ct.byref(ay_ptr),
            ct.byref(az_ptr),
            ct.byref(aw_ptr),
            ct.byref(aff_ptr),
            ct.byref(wx_ptr),
            ct.byref(wy_ptr),
            ct.byref(wz_ptr),
            ct.byref(ww_ptr),
            ct.byref(n_atoms),
            ct.byref(n_weights),
            ct.byref(status)
        )
        _check_error_code(status, "molecule_get_data")

        n = n_atoms.value
        self._atom_data["x"]       = np.array([ax_ptr[i] for i in range(n)],                    dtype=np.float64)
        self._atom_data["y"]       = np.array([ay_ptr[i] for i in range(n)],                    dtype=np.float64)
        self._atom_data["z"]       = np.array([az_ptr[i] for i in range(n)],                    dtype=np.float64)
        self._atom_data["weights"] = np.array([aw_ptr[i] for i in range(n)],                    dtype=np.float64)
        self._atom_data["ff_type"] = np.array([aff_ptr[i].decode('utf-8') for i in range(n)],    dtype=np.str_   )
        m = n_weights.value
        self._water_data["x"]       = np.array([wx_ptr[i] for i in range(m)],                    dtype=np.float64)
        self._water_data["y"]       = np.array([wy_ptr[i] for i in range(m)],                    dtype=np.float64)
        self._water_data["z"]       = np.array([wz_ptr[i] for i in range(m)],                    dtype=np.float64)
        self._water_data["weights"] = np.array([ww_ptr[i] for i in range(m)],                    dtype=np.float64)
        self._water_data["ff_type"] = "OH"
        ausaxs.deallocate(data_id)

    def hydrate(self) -> None:
        """Add a hydration shell to the molecule."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.molecule_hydrate(
            self._object_id,
            ct.byref(status)
        )
        _check_error_code(status, "molecule_hydrate")

        # invalidate cached data to refresh data on next access
        self._atom_data = {}
        self._water_data = {}

    def distance_histogram(self) -> dict[str, np.ndarray]:
        """Get the partial distance histogram of the molecule."""
        ausaxs = AUSAXS()
        aa_ptr = ct.POINTER(ct.c_double)()
        aw_ptr = ct.POINTER(ct.c_double)()
        ww_ptr = ct.POINTER(ct.c_double)()
        ax_ptr = ct.POINTER(ct.c_double)()
        xx_ptr = ct.POINTER(ct.c_double)()
        wx_ptr = ct.POINTER(ct.c_double)()
        n_bins = ct.c_int()
        delta_r = ct.c_double()
        exv_hists = ct.c_bool()
        status = ct.c_int()
        tmp_id = ausaxs.lib().functions.molecule_distance_histogram(
            self._object_id,
            ct.byref(aa_ptr),
            ct.byref(aw_ptr),
            ct.byref(ww_ptr),
            ct.byref(ax_ptr),
            ct.byref(xx_ptr),
            ct.byref(wx_ptr),
            ct.byref(n_bins),
            ct.byref(delta_r),
            ct.byref(exv_hists),
            ct.byref(status)
        )
        _check_error_code(status, "molecule_distance_histogram")

        res = {}
        n = n_bins.value
        res["bins"] = np.array([delta_r.value * (i + 0.5) for i in range(n)], dtype=np.float64)
        res["aa"] = np.array([aa_ptr[i] for i in range(n)], dtype=np.float64)
        res["aw"] = np.array([aw_ptr[i] for i in range(n)], dtype=np.float64)
        res["ww"] = np.array([ww_ptr[i] for i in range(n)], dtype=np.float64)
        if exv_hists.value:
            res["ax"] = np.array([ax_ptr[i] for i in range(n)], dtype=np.float64)
            res["xx"] = np.array([xx_ptr[i] for i in range(n)], dtype=np.float64)
            res["wx"] = np.array([wx_ptr[i] for i in range(n)], dtype=np.float64)
        else:
            res["ax"] = None
            res["xx"] = None
            res["wx"] = None
        ausaxs.deallocate(tmp_id)
    
    def histogram(self) -> dict[str, np.ndarray]:
        return self.distance_histogram()

    def debye(self, q_vals: list[float] | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the Debye scattering intensity of the molecule."""
        ausaxs = AUSAXS()
        if q_vals:
            q = np.array(q_vals, dtype=np.float64)
            n_q = ct.c_int(len(q_vals))
            i_ptr = ct.POINTER(ct.c_double)()
            status = ct.c_int()
            tmp_id = ausaxs.lib().functions.debye_from_molecule_userq(
                self._object_id,
                q.ctypes.data_as(ct.POINTER(ct.c_double)),
                n_q,
                ct.byref(i_ptr),
                ct.byref(status)
            )
            _check_error_code(status, "molecule_debye_q")

            i = np.array([i_ptr[i] for i in range(n_q.value)], dtype=np.float64)
            ausaxs.deallocate(tmp_id)
            return q, i
        else:
            q_ptr = ct.POINTER(ct.c_double)()
            i_ptr = ct.POINTER(ct.c_double)()
            n_q = ct.c_int()
            status = ct.c_int()
            tmp_id = ausaxs.lib().functions.debye_from_molecule(
                self._object_id,
                ct.byref(q_ptr),
                ct.byref(i_ptr),
                ct.byref(n_q),
                ct.byref(status)
            )
            _check_error_code(status, "molecule_debye")

            n = n_q.value
            q = np.array([q_ptr[i] for i in range(n)], dtype=np.float64)
            i = np.array([i_ptr[i] for i in range(n)], dtype=np.float64)
            ausaxs.deallocate(tmp_id)
            return q, i

    def atoms(self) -> list[np.ndarray]:
        """Get atomic data as a list of numpy arrays: (x, y, z, weights, ff_type)."""
        self._get_data()
        return [
            self._atom_data["x"],
            self._atom_data["y"],
            self._atom_data["z"],
            self._atom_data["weights"],
            self._atom_data["ff_type"]
        ]

    def waters(self) -> list[np.ndarray]:
        """Get water data as a list of numpy arrays: (x, y, z, weights)."""
        self._get_data()
        return [
            self._water_data["x"],
            self._water_data["y"],
            self._water_data["z"],
            self._water_data["weights"]
        ]

    def atomic_dict(self) -> dict[str, np.ndarray]:
        """Get atomic data as a dictionary with keys: 'x', 'y', 'z', 'weights', 'ff_type'."""
        self._get_data()
        return self._atom_data

    def water_dict(self) -> dict[str, np.ndarray]:
        """Get water data as a dictionary with keys: 'x', 'y', 'z', 'weights' 'ff_type'."""
        self._get_data()
        return self._water_data

@overload
def create_molecule(filename: str) -> Molecule: ...
@overload
def create_molecule(pdb: PDBfile) -> Molecule: ...
@overload
def create_molecule(
    x: np.ndarray | list[float], 
    y: np.ndarray | list[float], 
    z: np.ndarray | list[float], 
    weights: np.ndarray | list[float]
) -> Molecule: ...

def create_molecule(*args) -> Molecule:
    return Molecule(*args)
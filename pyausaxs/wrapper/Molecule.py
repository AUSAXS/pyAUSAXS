from .AUSAXS import AUSAXS, _check_error_code, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays
from .BackendObject import BackendObject
from .PDBfile import PDBfile
from .Histogram import Histogram
from .Datafile import Datafile
from .FitResult import FitResult
import ctypes as ct
import numpy as np
from typing import overload

class Molecule(BackendObject):
    __slots__ = ['_atom_data', '_water_data']
    def __init__(self, *args):
        super().__init__()
        self._atom_data: dict[str, np.ndarray] = {}
        self._water_data: dict[str, np.ndarray] = {}
        self._create_molecule(*args)

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
        self, x: np.ndarray | list[float], y: np.ndarray | list[float], z: np.ndarray | list[float], w: np.ndarray | list[float]
    ) -> None:
        _check_array_inputs(
            x, y, z, w,
            names=['x', 'y', 'z', 'w']
        )
        _check_similar_length(x, y, z, w, msg="Atomic coordinates and weights must have the same length")
        x, y, z, w = _as_numpy_f64_arrays(x, y, z, w)

        ausaxs = AUSAXS()
        n_atoms = ct.c_int(len(x))
        status = ct.c_int()
        self._object_id = ausaxs.lib().functions.molecule_from_arrays(
            x.ctypes.data_as(ct.POINTER(ct.c_double)),
            y.ctypes.data_as(ct.POINTER(ct.c_double)),
            z.ctypes.data_as(ct.POINTER(ct.c_double)),
            w.ctypes.data_as(ct.POINTER(ct.c_double)),
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
        self._atom_data = {}
        self._water_data = {}

    def clear_hydration(self) -> None:
        """Remove the hydration shell from the molecule."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.molecule_clear_hydration(
            self._object_id,
            ct.byref(status)
        )
        _check_error_code(status, "molecule_clear_hydration")
        self._atom_data = {}
        self._water_data = {}

    def radius_of_gyration(self) -> float:
        """Get the radius of gyration of the molecule."""
        ausaxs = AUSAXS()
        rg = ct.c_double()
        status = ct.c_int()
        ausaxs.lib().functions.molecule_Rg(
            self._object_id,
            ct.byref(rg),
            ct.byref(status)
        )
        _check_error_code(status, "molecule_radius_of_gyration")
        return rg.value

    def Rg(self) -> float:
        return self.radius_of_gyration()

    def distance_histogram(self) -> Histogram:
        """Get the partial distance histogram of the molecule."""
        ausaxs = AUSAXS()
        aa_ptr = ct.POINTER(ct.c_double)()
        aw_ptr = ct.POINTER(ct.c_double)()
        ww_ptr = ct.POINTER(ct.c_double)()
        n_bins = ct.c_int()
        delta_r = ct.c_double()
        status = ct.c_int()
        tmp_id = ausaxs.lib().functions.molecule_distance_histogram(
            self._object_id,
            ct.byref(aa_ptr),
            ct.byref(aw_ptr),
            ct.byref(ww_ptr),
            ct.byref(n_bins),
            ct.byref(delta_r),
            ct.byref(status)
        )
        _check_error_code(status, "molecule_distance_histogram")

        n = n_bins.value
        hist = Histogram(
            np.array([delta_r.value*i for i in range(n)], dtype=np.float64),
            np.array([aa_ptr[i] for i in range(n)], dtype=np.float64),
            np.array([aw_ptr[i] for i in range(n)], dtype=np.float64),
            np.array([ww_ptr[i] for i in range(n)], dtype=np.float64),
        )
        ausaxs.deallocate(tmp_id)
        return hist
    
    def histogram(self) -> Histogram:
        return self.distance_histogram()

    def debye(self, q_vals: list[float] | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Debye scattering intensity of the molecule.
        Returns: (q, I)
        """
        ausaxs = AUSAXS()
        if q_vals:
            q = np.array(q_vals, dtype=np.float64)
            i = np.zeros_like(q, dtype=np.float64)
            n_q = ct.c_int(len(q_vals))
            status = ct.c_int()
            tmp_id = ausaxs.lib().functions.molecule_debye_userq(
                self._object_id,
                q.ctypes.data_as(ct.POINTER(ct.c_double)),
                i.ctypes.data_as(ct.POINTER(ct.c_double)),
                n_q,
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
            tmp_id = ausaxs.lib().functions.molecule_debye(
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

    def fit(self, data: Datafile) -> float:
        """
        Fit the Debye scattering intensity of the molecule to the provided data.
        Returns: chi-squared value of the fit.
        """
        ausaxs = AUSAXS()
        status = ct.c_int()
        res_id = ausaxs.lib().functions.molecule_debye_fit(
            self._object_id,
            data._object_id,
            ct.byref(status)
        )
        _check_error_code(status, "molecule_fit")
        return FitResult(res_id)

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
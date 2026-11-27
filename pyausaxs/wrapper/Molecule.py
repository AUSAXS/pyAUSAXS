from .AUSAXS import call, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays
from .BackendObject import BackendObject
from .PDBfile import PDBfile
from .Histogram import Histogram
from .Datafile import Datafile
from .FitResult import FitResult
from .Models import ExvModel, WeightedBins
from .settings import settings
from pyausaxs.signatures import register, arg, array, length, out, out_array, buffer, STATUS, HOLDER
import ctypes as ct
import numpy as np
from typing import overload

# a profile on the default q axis, and its _userq variant at the given q values
PROFILE = ([ct.c_int, out_array("I", ct.c_double, n="n"), out("n", ct.c_int), STATUS], HOLDER)
PROFILE_USERQ = ([ct.c_int, array("q"), buffer("I", n="q"), length("q"), STATUS], None)

register({
    "molecule_from_file":    ([ct.c_char_p, STATUS], ct.c_int),
    "molecule_from_pdb_id":  ([ct.c_int, STATUS], ct.c_int),
    "molecule_from_arrays":  ([array("x"), array("y"), array("z"), array("w"), length("x"), STATUS], ct.c_int),
    "molecule_get_data": (
        [
            ct.c_int,
            out_array("x", ct.c_double, n="na"), out_array("y", ct.c_double, n="na"), out_array("z", ct.c_double, n="na"),
            out_array("weights", ct.c_double, n="na"), out_array("ff_type", ct.c_char_p, n="na"),
            out_array("wx", ct.c_double, n="nw"), out_array("wy", ct.c_double, n="nw"), out_array("wz", ct.c_double, n="nw"),
            out_array("ww", ct.c_double, n="nw"),
            out("na", ct.c_int), out("nw", ct.c_int), STATUS
        ],
        HOLDER
    ),
    "molecule_hydrate":         ([ct.c_int, STATUS], None),
    "molecule_clear_hydration": ([ct.c_int, STATUS], None),
    "molecule_Rg":              ([ct.c_int, out("Rg", ct.c_double), STATUS], None),
    "molecule_distance_histogram": (
        [
            ct.c_int,
            out_array("aa", ct.c_double, n="n"), out_array("aw", ct.c_double, n="n"), out_array("ww", ct.c_double, n="n"),
            out_array("axis", ct.c_double, n="n"), out("n", ct.c_int), STATUS
        ],
        HOLDER
    ),
    "molecule_distance_histogram_raw": (
        [ct.c_int, out_array("counts", ct.c_double, n="n"), out_array("axis", ct.c_double, n="n"), out("n", ct.c_int), STATUS],
        HOLDER
    ),
    "molecule_set_coordinates": ([ct.c_int, array("x"), array("y"), array("z"), length("x"), STATUS], None),
    "molecule_debye_raw_vjp": (
        [
            ct.c_int, array("q"), array("v"), length("q"),
            buffer("gx", n="n_atoms"), buffer("gy", n="n_atoms"), buffer("gz", n="n_atoms"), arg("n_atoms", ct.c_int), STATUS
        ],
        None
    ),
    "molecule_debye":                    PROFILE,
    "molecule_debye_userq":              PROFILE_USERQ,
    "molecule_debye_raw":                PROFILE,
    "molecule_debye_raw_userq":          PROFILE_USERQ,
    "molecule_amplitude_centroid":       PROFILE,
    "molecule_amplitude_centroid_userq": PROFILE_USERQ,
    "molecule_debye_fit": ([ct.c_int, ct.c_int, STATUS], ct.c_int),
})

def debye_profile(name: str, molecule_id: int, q_vals: list[float] | np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the backend profile function `name` on the default q axis, or `name`_userq at the given q values.
    Returns: (q, I)
    """
    if q_vals is not None:
        q = _as_numpy_f64_arrays(q_vals)[0]
        return q, call(f"{name}_userq", molecule_id, q)
    return settings.q_axis(), call(name, molecule_id)

class Molecule(BackendObject):
    __slots__ = ['_atom_data', '_water_data']
    def __init__(self, *args):
        super().__init__()
        self._atom_data: dict[str, np.ndarray] = {}
        self._water_data: dict[str, np.ndarray] = {}
        self._create_molecule(*args)

    def _create_molecule_from_file(self, filename: str) -> None:
        self._set_id(call("molecule_from_file", filename))

    def _create_molecule_from_pdb(self, pdb: PDBfile) -> None:
        self._set_id(call("molecule_from_pdb_id", pdb._get_id()))

    def _create_molecule_from_arrays(
        self, x: np.ndarray | list[float], y: np.ndarray | list[float], z: np.ndarray | list[float], w: np.ndarray | list[float]
    ) -> None:
        _check_array_inputs(
            x, y, z, w,
            names=['x', 'y', 'z', 'w']
        )
        _check_similar_length(x, y, z, w, msg="Atomic coordinates and weights must have the same length")
        self._set_id(call("molecule_from_arrays", x, y, z, w))

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
        data = call("molecule_get_data", self._get_id())
        self._atom_data = {k: getattr(data, k) for k in ("x", "y", "z", "weights", "ff_type")}
        self._water_data = {"x": data.wx, "y": data.wy, "z": data.wz, "weights": data.ww, "ff_type": "HOH"}

    def hydrate(self) -> None:
        """Add a hydration shell to the molecule."""
        call("molecule_hydrate", self._get_id())
        self._atom_data = {}
        self._water_data = {}

    def clear_hydration(self) -> None:
        """Remove the hydration shell from the molecule."""
        call("molecule_clear_hydration", self._get_id())
        self._atom_data = {}
        self._water_data = {}

    def radius_of_gyration(self) -> float:
        """Get the radius of gyration of the molecule."""
        return call("molecule_Rg", self._get_id())

    def Rg(self) -> float:
        return self.radius_of_gyration()

    def distance_histogram(self) -> Histogram:
        """Get the partial distance histogram of the molecule."""
        h = call("molecule_distance_histogram", self._get_id())
        return Histogram(h.axis, h.aa, h.aw, h.ww)
    
    def histogram(self) -> Histogram:
        return self.distance_histogram()

    def distance_histogram_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distance histogram of the molecule. No form factors or excluded volume effects will be applied.
        Returns: (bins, counts)
        """
        h = call("molecule_distance_histogram_raw", self._get_id())
        return h.axis, h.counts

    def debye(self, q_vals: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Debye scattering intensity of the molecule. Form factors and excluded volume effects will be applied
        Returns: (q, I)
        """
        return debye_profile("molecule_debye", self._get_id(), q_vals)

    def amplitude(self, q_vals: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the orientationally averaged scattering amplitude <A(q)> of the molecule, with the same form factors and excluded
        volume model as debye(). The amplitude is taken about the centroid of the excess scattering length.

        Together with the intensity from debye() this gives the beta(q) factor of the decoupling approximation,
            beta(q) = <A(q)>^2 / <|A(q)|^2>,   I(q) = I_0(q) [1 + beta(q) (S(q) - 1)],
        which accounts for the anisotropy of the molecule when combining it with a structure factor S(q).

        Only the excluded volume models simple, none, vacuum and fraser are supported.
        Returns: (q, A)
        """
        return debye_profile("molecule_amplitude_centroid", self._get_id(), q_vals)

    def debye_raw(
        self, q_vals: list[float] | np.ndarray | None = None, form_factors: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Debye scattering intensity of the molecule without excluded volume effects.
        Every atom and water is a point scatterer with its weight. With form_factors, each instead carries its atomic form factor;
        this requires a molecule with form factor information, e.g. one read from a structure file, and raises otherwise.
        Returns: (q, I)
        """
        if not form_factors:
            return debye_profile("molecule_debye_raw", self._get_id(), q_vals)

        # weighted bins, since the profile must hold at any q, far beyond the range auto judges unweighted bins by
        exv, weighted_bins = settings.get("exv_model"), settings.get("weighted_bins")
        try:
            settings.exv(ExvModel.vacuum)
            settings.set("weighted_bins", WeightedBins.true.value)
            return debye_profile("molecule_debye", self._get_id(), q_vals)
        finally:
            settings.set("exv_model", exv)
            settings.set("weighted_bins", weighted_bins.value)

    def set_coordinates(self, coords: np.ndarray) -> None:
        """
        Update the atomic coordinates with the (n_atoms, 3) array coords, in the order of atoms().
        Scattering weights and form factors are unchanged.
        """
        coords = np.asarray(coords, dtype=np.float64)
        n = len(self.atoms()[0])
        if coords.shape != (n, 3):
            raise ValueError(f"Expected coordinates of shape ({n}, 3), got {coords.shape}.")
        x, y, z = (np.ascontiguousarray(coords[:, i]) for i in range(3))
        call("molecule_set_coordinates", self._get_id(), x, y, z)
        self._atom_data["x"], self._atom_data["y"], self._atom_data["z"] = x, y, z

    def debye_raw_vjp(self, q_vals: list[float] | np.ndarray, v: list[float] | np.ndarray) -> np.ndarray:
        """
        Vector-Jacobian product of debye_raw(q_vals) with respect to the atomic coordinates, for point scatterers only (no form factors). 
        Given v = dL/dI at q_vals, returns dL/dr as an (n_atoms, 3) array, in the order of atoms(). 
        This is the derivative of the exact Debye sum, not of its binned approximation.
        """
        q, v = _as_numpy_f64_arrays(q_vals, v)
        _check_similar_length(q, v, msg="q_vals and v must have the same length")
        g = call("molecule_debye_raw_vjp", self._get_id(), q, v, len(self.atoms()[0]))
        return np.stack([g.gx, g.gy, g.gz], axis=1)

    def fit(self, data: str | Datafile) -> FitResult:
        """
        Fit the Debye scattering intensity of the molecule to the provided data.
        Returns: chi-squared value of the fit.
        """
        if isinstance(data, str):
            data = Datafile(data)

        return FitResult(call("molecule_debye_fit", self._get_id(), data._get_id()))

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

def create_molecule(*args) -> Molecule: # type: ignore[reportInconsistentOverload]
    return Molecule(*args)
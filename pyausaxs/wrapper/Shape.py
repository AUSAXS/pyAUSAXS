from .AUSAXS import call, _as_numpy_f64_arrays
from .settings import settings
from pyausaxs.signatures import register, array, length, buffer, STATUS
import ctypes as ct
import numpy as np

register({
    "shape_debye_userq": (
        [array("x"), array("y"), array("z"), length("x"), ct.c_double, array("q"), buffer("I", n="q"), length("q"), STATUS],
        None
    ),
})

class Lattice:
    """A homogeneous body made of the cells of a cubic lattice, each with a Gaussian form factor."""
    def __init__(self, cells: np.ndarray, spacing: float):
        """
        cells:   an (N, 3) array of cell centres, all on one lattice of the given spacing.
        spacing: the lattice spacing in Å.
        """
        cells = np.asarray(cells, dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError(f"cells must be an (N, 3) array, got shape {cells.shape}")
        self.cells = cells
        self.spacing = float(spacing)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def volume(self) -> float:
        """The total volume of the cells in Å³."""
        return len(self.cells)*self.spacing**3

    def debye(self, q_vals: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the scattering intensity of the body. Without q values, the default q-axis is used.
        Returns: (q, I)
        """
        x, y, z = [np.ascontiguousarray(self.cells[:, k]) for k in range(3)]
        q = settings.q_axis() if q_vals is None else _as_numpy_f64_arrays(q_vals)[0]
        return q, call("shape_debye_userq", x, y, z, self.spacing, q)

class shape():
    Lattice = Lattice

    @staticmethod
    def lattice(inside, extent: float | tuple[float, float, float], spacing: float = 1.0) -> Lattice:
        """
        Fill a shape with cells of a cubic lattice.
        inside:  a vectorised function (x, y, z) -> bool, true inside the shape.
        extent:  the half-width of the box to search, either one value or one per axis, in Å.
        spacing: the lattice spacing in Å.
        """
        ex, ey, ez = (extent,)*3 if np.isscalar(extent) else extent
        axes = [np.arange(-np.ceil(e/spacing), np.ceil(e/spacing) + 1)*spacing for e in (ex, ey, ez)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        mask = np.asarray(inside(X, Y, Z), dtype=bool)
        return Lattice(np.column_stack((X[mask], Y[mask], Z[mask])), spacing)

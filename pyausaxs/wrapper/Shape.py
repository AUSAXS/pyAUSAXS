from .AUSAXS import AUSAXS, _check_error_code, _as_numpy_f64_arrays
from pyausaxs.signatures import register
import ctypes as ct
import numpy as np

register({
    "shape_debye_userq": (
        [
            ct.POINTER(ct.c_double), # x
            ct.POINTER(ct.c_double), # y
            ct.POINTER(ct.c_double), # z
            ct.c_int,                # n_cells
            ct.c_double,             # spacing
            ct.POINTER(ct.c_double), # q
            ct.POINTER(ct.c_double), # I (output)
            ct.c_int,                # n_q
            ct.POINTER(ct.c_int)     # status (0 = success)
        ],
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

    def debye(self, q_vals: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the scattering intensity of the body.
        Returns: (q, I)
        """
        x, y, z = [np.ascontiguousarray(self.cells[:, k]) for k in range(3)]
        q = _as_numpy_f64_arrays(q_vals)[0]
        i = np.zeros_like(q, dtype=np.float64)
        status = ct.c_int()
        AUSAXS().lib().functions.shape_debye_userq(
            x.ctypes.data_as(ct.POINTER(ct.c_double)),
            y.ctypes.data_as(ct.POINTER(ct.c_double)),
            z.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(len(x)),
            ct.c_double(self.spacing),
            q.ctypes.data_as(ct.POINTER(ct.c_double)),
            i.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(len(q)),
            ct.byref(status)
        )
        _check_error_code(status, "shape_debye")
        return q, i

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

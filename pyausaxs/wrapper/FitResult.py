from .AUSAXS import AUSAXS, _check_error_code
from .BackendObject import BackendObject
from typing import Any, TYPE_CHECKING
import ctypes as ct
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

class FitResult(BackendObject):
    def __init__(self, id: int):
        super().__init__()
        self._set_id(id)
        self._fit_info: dict[str, Any] = {}
        self._fit_curves: list[np.ndarray] = []

    def _get_fit_curves(self) -> None:
        if self._fit_curves: return
        ausaxs = AUSAXS()
        q_ptr        = ct.POINTER(ct.c_double)()
        I_data_ptr   = ct.POINTER(ct.c_double)()
        I_err_ptr    = ct.POINTER(ct.c_double)()
        I_model_ptr  = ct.POINTER(ct.c_double)()
        n_points     = ct.c_int()
        status       = ct.c_int()

        data_id = ausaxs.lib().functions.fit_get_fit_curves(
            self._get_id(),
            ct.byref(q_ptr),
            ct.byref(I_data_ptr),
            ct.byref(I_err_ptr),
            ct.byref(I_model_ptr),
            ct.byref(n_points),
            ct.byref(status)
        )
        _check_error_code(status, "fit_get_fit_curves")

        n = n_points.value
        self._fit_curves = [
            np.array([q_ptr[i] for i in range(n)],        dtype=np.float64),
            np.array([I_data_ptr[i] for i in range(n)],   dtype=np.float64),
            np.array([I_err_ptr[i] for i in range(n)],    dtype=np.float64),
            np.array([I_model_ptr[i] for i in range(n)],  dtype=np.float64)
        ]
        ausaxs.deallocate(data_id)

    def _get_fit_info(self) -> None:
        if self._fit_info: return
        ausaxs = AUSAXS()
        pars_ptr     = ct.POINTER(ct.c_char_p)()
        pvals_ptr    = ct.POINTER(ct.c_double)()
        perr_min_ptr = ct.POINTER(ct.c_double)()
        perr_max_ptr = ct.POINTER(ct.c_double)()
        n_pars       = ct.c_int()
        chi2         = ct.c_double()
        dof          = ct.c_int()
        status       = ct.c_int()
        data_id = ausaxs.lib().functions.fit_get_fit_info(
            self._get_id(),
            ct.byref(pars_ptr),
            ct.byref(pvals_ptr),
            ct.byref(perr_min_ptr),
            ct.byref(perr_max_ptr),
            ct.byref(n_pars),
            ct.byref(chi2),
            ct.byref(dof),
            ct.byref(status)
        )
        _check_error_code(status, "fit_get_fit_info")

        n = n_pars.value
        self._fit_info["pars"]      = [pars_ptr[i].decode('utf-8') for i in range(n)]
        self._fit_info["pvals"]     = [pvals_ptr[i] for i in range(n)]
        self._fit_info["perr_min"]  = [perr_min_ptr[i] for i in range(n)]
        self._fit_info["perr_max"]  = [perr_max_ptr[i] for i in range(n)]
        self._fit_info["chi2"]      = chi2.value
        self._fit_info["dof"]       = dof.value
        ausaxs.deallocate(data_id)

    def chi2(self) -> float:
        """Absolute chi-squared value for the fit."""
        self._get_fit_info()
        return self._fit_info["chi2"]

    def chi2r(self) -> float:
        """Reduced chi-squared value for the fit."""
        self._get_fit_info()
        return self._fit_info["chi2"] / self._fit_info["dof"]

    def dof(self) -> int:
        """Number of degrees of freedom for the fit."""
        self._get_fit_info()
        return self._fit_info["dof"]

    def fit_parameters(self) -> dict[str, tuple[float, float, float]]:
        """Returns a dictionary of fit parameters with their values and errors."""
        self._get_fit_info()
        params = {}
        for i, name in enumerate(self._fit_info["pars"]):
            params[name] = (
                self._fit_info["pvals"][i],
                self._fit_info["perr_min"][i],
                self._fit_info["perr_max"][i]
            )
        return params

    def fit_curves(self) -> list[np.ndarray]:
        """Returns q, I_data, I_err, I_model arrays."""
        self._get_fit_curves()
        return self._fit_curves

    def fitted_curve(self) -> np.ndarray:
        """Returns the fitted intensity curve."""
        self._get_fit_curves()
        return self._fit_curves[3]

    def data_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns q, I, Ierr arrays from the data."""
        self._get_fit_curves()
        return self._fit_curves[0], self._fit_curves[1], self._fit_curves[2]

    def residuals(self) -> np.ndarray:
        """Returns the normalized residuals."""
        self._get_fit_curves()
        return (self._fit_curves[1] - self._fit_curves[3])/self._fit_curves[2]

    def plot(self) -> "Figure":
        """Plot the fit results using matplotlib."""

        import importlib.util
        plt = importlib.util.find_spec("matplotlib")
        if plt is None:
            raise ImportError("matplotlib is required for plotting fit results.")
        import matplotlib.figure as fig
        import matplotlib.pyplot as plt

        self._get_fit_curves()
        q = self._fit_curves[0]
        I_data = self._fit_curves[1]
        I_err = self._fit_curves[2]
        I_model = self._fit_curves[3]

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
        ax[0].errorbar(q, I_data, yerr=I_err, fmt='ko', markersize=4)
        ax[0].plot(q, I_model, '-', label=f'$\\chi^2_r = {self.chi2r():.2f}$', color='red', lw=2, zorder=10)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Intensity')
        ax[0].legend()

        residuals = (I_data - I_model) / I_err
        ax[1].plot(q, residuals, 'ko', markersize=4)
        ax[1].axhline(0, color='k', linestyle='--')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('q [$\\AA^{-1}$]')
        ax[1].set_ylabel('Residuals')

        plt.tight_layout()
        return fig
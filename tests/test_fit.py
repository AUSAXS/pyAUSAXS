import numpy as np
import pyausaxs as ausaxs

from scipy.optimize import least_squares


def test_automatic_fit():
    data = ausaxs.read_data("tests/files/2epe.dat")
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    fit_result = mol.fit(data)
    q_fit, I_data, I_err, I_model = fit_result.fit_curves()
    chi2 = fit_result.chi2()
    dof = fit_result.dof()
    params = fit_result.fit_parameters()
    assert len(q_fit) == len(I_data) == len(I_err) == len(I_model), "Fitted curves should have same length"
    assert chi2 > 0, "Chi2 should be positive"
    assert dof > 0, "Degrees of freedom should be positive"
    assert len(params) > 0, "Should have some fit parameters"


def test_manual_fit_trivial():
    # trivial fit with single scaling parameter
    data = ausaxs.read_data("tests/files/2epe.dat")
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    fit = ausaxs.manual_fit(mol, data)
    pars = [1.0]
    I_fitted = fit.evaluate(pars)
    _, I_data, _ = data.data()
    assert len(I_fitted) == len(I_data), "Fitted I(q) should match data length"


def test_manual_fit_works_with_scipy():
    data = ausaxs.read_data("tests/files/2epe.dat")
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    fit = ausaxs.manual_fit(mol, data)

    ausaxs.settings.fit(fit_hydration=True)
    _, I, Ierr = data.data()

    def linear_fit(I_model, I_data, I_err):
        weights = 1.0 / I_err
        return np.polyfit(I_model, I_data, 1, w=weights)

    def residuals(pars):
        I_model = fit.evaluate(pars)
        a, b = linear_fit(I_model, I, Ierr)
        residual = (a*I_model+b - I) / Ierr
        return residual

    initial_pars = [1.0]
    result = least_squares(residuals, initial_pars)
    fitted_pars = result.x
    I_fitted = fit.evaluate(fitted_pars)
    a, b = linear_fit(I_fitted, I, Ierr)
    I_fitted = a*I_fitted+b

    afit = mol.fit(data)
    assert result.success, "Optimization should succeed"
    assert len(I_fitted) == len(data.data()[0]), "Fitted I(q) should match data length"
    assert np.sum(np.square(residuals(fitted_pars))) - afit.chi2() < 1e-3, "Chi2 from manual fit should match automatic fit"

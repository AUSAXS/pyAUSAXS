import numpy as np
import pytest
import pyausaxs as ausaxs


@pytest.fixture
def exv_model():
    """Restore the default excluded volume model after the test."""
    yield
    ausaxs.settings.exv(ausaxs.ExvModel.simple)


@pytest.mark.parametrize("model", [ausaxs.ExvModel.simple, ausaxs.ExvModel.none, ausaxs.ExvModel.vacuum, ausaxs.ExvModel.fraser])
def test_amplitude_consistent_with_intensity(exv_model, model):
    ausaxs.settings.exv(model)
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    mol.hydrate()
    q, I = mol.debye()
    qa, A = mol.amplitude()
    assert np.allclose(q, qa), "amplitude() and debye() should share the default q axis"

    beta = A**2/I
    assert beta[0] == pytest.approx(1, rel=1e-3), "beta(q) -> 1 as q -> 0"
    assert np.all(beta <= 1.01), "<A>^2 <= <A^2> at every q"
    assert beta[-1] < 0.9, "an anisotropic molecule should have beta < 1 at large q"


def test_amplitude_userq():
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    q = np.linspace(0.01, 0.5, 20)
    q_out, A = mol.amplitude(q)
    assert np.allclose(q_out, q)
    _, A_list = mol.amplitude(list(q))
    assert np.allclose(A, A_list)

    # any q is supported, also beyond the default axis
    _, A_high = mol.amplitude([5, 10])
    assert np.all(np.isfinite(A_high))


def test_amplitude_unsupported_model(exv_model):
    ausaxs.settings.exv(ausaxs.ExvModel.grid)
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    with pytest.raises(Exception):
        mol.amplitude()


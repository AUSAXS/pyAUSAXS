import numpy as np
import pyausaxs as ausaxs


def test_amplitude_default_axis():
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    q, I = mol.debye()
    qa, A = mol.amplitude()
    assert np.allclose(q, qa), "amplitude() and debye() should share the default q axis"
    assert A.shape == I.shape
    assert not np.allclose(A, I), "amplitude() should call the amplitude backend function, not the intensity one"


def test_amplitude_userq():
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    q = np.linspace(0.01, 0.5, 20)
    q_out, A = mol.amplitude(q)
    assert np.allclose(q_out, q)
    assert A.shape == q.shape
    _, A_list = mol.amplitude(list(q))
    assert np.allclose(A, A_list)

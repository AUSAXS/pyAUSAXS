import math
import numpy as np
import pyausaxs as ausaxs


def test_Rg():
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    Rg = mol.Rg()
    assert math.isclose(Rg, 13.89, abs_tol=0.1), f"Radius of gyration mismatch: expected 16.2162, got {Rg}"


def test_custom_q_range():
    q1 = [0.01*i for i in range(1, 51)]
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    q1, I1 = mol.debye(q1)

    q2 = np.linspace(0.01, 0.5, 50)
    _, I2 = mol.debye(q2)

    assert np.allclose(q1, q2, atol=1e-6), "q values should match for list and array input"
    assert np.allclose(I1, I2, atol=1e-6), "I(q) values should match for list and array input"

    q3 = np.linspace(0.01, 5, 100)
    q3, I3 = mol.debye(q3)
    assert q3[-1] > 2


def test_custom_bin_width():
    ausaxs.settings.histogram(bin_width = 0.1)
    hist = ausaxs.create_molecule("tests/files/2epe.pdb").histogram()
    bins = hist.bins()
    assert math.isclose(bins[1]-bins[0], 0.1, abs_tol=1e-6), "Histogram bin width should be 0.1"

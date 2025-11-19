import numpy as np
import pyausaxs as ausaxs
from .helpers import simple_cube


def test_histogram():
    atoms = simple_cube.points()
    distances, counts_exact = simple_cube.hist()
    mol = ausaxs.create_molecule(*atoms)
    hist = mol.histogram()
    bins, counts = hist.bins(), hist.counts()
    w = (bins[1] - bins[0])
    for d, c_exact in zip(distances, counts_exact):
        index = np.round(d / w).astype(int)
        c_hist = counts[index]
        assert index < len(bins), f"Distance {d} out of histogram bin range"
        assert abs(bins[index] - d) <= w/2, f"Bin mismatch for distance {d}: found {bins[index]}"
        assert abs(c_hist - c_exact) <= 1e-6, f"Count mismatch for distance {d}: expected {c_exact}, got {c_hist}"


def test_debye():
    atoms = simple_cube.points()
    mol = ausaxs.create_molecule(*atoms)
    q, I = mol.debye()
    I_expected = simple_cube.debye(q) * np.array([np.exp(-qi*qi) for qi in q])
    assert np.allclose(I, I_expected, atol=1e-6), "Debye intensity mismatch"


def test_debye_raw_and_exact():
    ausaxs.settings.histogram(qmax=1)
    atoms = simple_cube.points()
    mol = ausaxs.create_molecule(*atoms)
    q, I = mol.debye_raw()
    I_expected = simple_cube.debye(q)
    assert np.allclose(I, I_expected, atol=1e-6), "Debye raw intensity mismatch"

    ausaxs.settings.histogram(bin_width=0.1)
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    mol.clear_hydration()
    q, I = mol.debye_raw()
    I_expected = ausaxs.unoptimized.debye_exact(mol, q)[1]
    assert np.allclose(I, I_expected, rtol=0.01, atol=1e-6), "Debye raw intensity mismatch for 2epe"


def test_debye_exact_and_fit():
    atoms = simple_cube.points()
    mol = ausaxs.create_molecule(*atoms)
    q = np.linspace(0.01, 2.0, 100)
    _, I = ausaxs.unoptimized.debye_exact(mol, q)
    I_expected = simple_cube.debye(q)
    assert np.allclose(I, I_expected, atol=1e-6), "Debye exact intensity mismatch"

    # run fit plot (non-interactive) to ensure no exceptions
    mol2 = ausaxs.create_molecule("tests/files/2epe.pdb")
    data = ausaxs.read_data("tests/files/2epe.dat")
    res = mol2.fit(data)
    res.plot()

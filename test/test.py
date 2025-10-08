import sys
import math
import numpy as np
from pyausaxs import AUSAXS

def read_pdb(filename):
    """
    Simple PDB reader that extracts coordinates and atom information.
    """
    xx, yy, zz, an, rn, e = [], [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip()
                if not element:
                    element = ''.join(c for c in atom_name if c.isalpha())[:1]
                    if not element:
                        element = 'C'

                xx.append(x)
                yy.append(y)
                zz.append(z)
                an.append(atom_name)
                rn.append(res_name)
                e.append(element)
    return (np.array(xx), np.array(yy), np.array(zz), an, rn, e)

def test_cube_debye():
    q = np.linspace(0.0, 1.0, 100)

    # build coordinates for 8 cube corners and center (0,0,0) -> total 9 points
    corners = [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1),
               (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]
    pts = corners + [(0.0, 0.0, 0.0)]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    zs = np.array([p[2] for p in pts], dtype=float)
    ws = np.ones(len(pts), dtype=float)

    # exact distances (unique) and multiplicities
    d_vals = [0.0, math.sqrt(3.0), 2.0, math.sqrt(8.0), math.sqrt(12.0)]
    multiplicities = [9, 16, 24, 24, 8]

    # compute expected I(q) from multiplicities and distances (weights=1)
    I_expected = np.zeros_like(q, dtype=float)
    for m, r in zip(multiplicities, d_vals):
        if r == 0.0:
            I_expected += m
        else:
            qr = q*r
            term = np.empty_like(qr)
            mask = qr == 0
            term[mask] = 1.0
            term[~mask] = np.sin(qr[~mask]) / qr[~mask]
            I_expected += m*term

    ausaxs = AUSAXS()
    I_native = ausaxs.debye(q, xs, ys, zs, ws)
    assert(np.allclose(I_native, I_expected, rtol=1e-5, atol=1e-8))

def test_fit():
    ausaxs = AUSAXS()
    def manual_fitting():
        q, I, Ierr = np.loadtxt("test/2epe.dat", usecols=(0,1,2), unpack=True)
        x, y, z, atom_names, res_names, elements = read_pdb("test/2epe.pdb")
        fitter = ausaxs.manual_fit(
            q, I, Ierr,
            x, y, z, atom_names, res_names, elements
        )
        Iq = fitter.step([1, 2])
        assert len(Iq) == len(I)

    def automatic_fitting():
        q, I, Ierr = np.loadtxt("test/2epe.dat", usecols=(0,1,2), unpack=True)
        x, y, z, atom_names, res_names, elements = read_pdb("test/2epe.pdb")
        Iq = ausaxs.fit(
            q, I, Ierr,
            x, y, z, atom_names, res_names, elements
        )
        assert len(Iq) == len(I)

    manual_fitting()
    automatic_fitting()

def test_pdb_reader():
    """Test that the PDB reader correctly parses atomic information."""
    x, y, z, atom_names, res_names, elements = read_pdb("test/2epe.pdb")
    assert len(x) > 0, "Should have parsed some atoms"
    assert len(x) == len(y) == len(z) == len(atom_names) == len(res_names) == len(elements), "All arrays should have same length"
    assert all(isinstance(name, str) for name in atom_names), "Atom names should be strings"
    assert all(isinstance(name, str) for name in res_names), "Residue names should be strings" 
    assert all(isinstance(elem, str) for elem in elements), "Elements should be strings"
    return True

def test_singleton():
    """Test that AUSAXS instances are the same object."""
    instance1 = AUSAXS()
    instance2 = AUSAXS()
    instance3 = AUSAXS()
    assert instance1 is instance2, "Instance 1 and 2 should be the same object"
    assert instance2 is instance3, "Instance 2 and 3 should be the same object"
    assert instance1 is instance3, "Instance 1 and 3 should be the same object"
    assert instance1.ready() == instance2.ready(), "All instances should have the same ready state"
    assert instance1.init_error() == instance2.init_error(), "All instances should have the same error state"

def test_reset_singleton():
    """Test that reset_singleton works correctly."""
    instance1 = AUSAXS()
    ready1 = instance1.ready()
    AUSAXS.reset_singleton()
    instance2 = AUSAXS()
    ready2 = instance2.ready()
    assert instance1 is not instance2, "After reset, new instance should be a different object"
    assert ready1 == ready2, "Ready state should be consistent across resets"

if __name__ == '__main__':
    import pyausaxs
    print(f"AUSAXS version {pyausaxs.__version__}")
    test_cube_debye()
    test_pdb_reader()
    test_singleton()
    test_reset_singleton()
    test_fit()
    print("All tests passed")
    sys.exit(0)
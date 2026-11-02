import math
import numpy as np
import pytest
import pyausaxs as ausaxs


def cell_ff(q, spacing):
    V = spacing**3
    return V*np.exp(-V**(2/3)*q*q/(4*math.pi))


def test_shape_lattice():
    cells = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= 1.5**2, extent=2, spacing=1).cells
    assert cells.shape == (19, 3)  # the centre, 6 faces and 12 edges of a 3x3x3 block; the 8 corners are at sqrt(3) > 1.5
    assert np.all(np.sum(cells**2, axis=1) <= 1.5**2)


def test_shape_debye_two_cells():
    cells = np.array([[0, 1, -2], [3, 1, 2]], dtype=float)
    q = np.array([0, 0.1, 0.3, 0.7])
    _, I = ausaxs.shape.Lattice(cells, spacing=1).debye(q)
    d = 5
    sinc = np.array([1] + [math.sin(qi*d)/(qi*d) for qi in q[1:]])
    assert np.allclose(I, (2 + 2*sinc)*cell_ff(q, 1)**2, rtol=1e-5)


def test_shape_debye_sphere_forward_scattering():
    sphere = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= 10**2, extent=10, spacing=0.5)
    _, I = sphere.debye([0])
    assert math.isclose(I[0], sphere.volume**2, rel_tol=1e-6)


def test_shape_debye_off_lattice():
    with pytest.raises(Exception):
        ausaxs.shape.Lattice(np.array([[0, 0, 0], [0, 0, 1.5]]), spacing=1).debye([0.1])


def test_shape_debye_default_q():
    sphere = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= 5**2, extent=5)
    q, I = sphere.debye()
    q_mol, _ = ausaxs.create_molecule("tests/files/2epe.pdb").debye()
    assert np.allclose(q, q_mol)
    assert len(I) == len(q)

import math
import numpy as np
import pyausaxs as ausaxs


def test_molecule_and_atoms():
    ausaxs.settings.molecule(implicit_hydrogens=False)
    mol1 = ausaxs.create_molecule("tests/files/2epe.pdb")
    x1, y1, z1, w1, ff1 = mol1.atoms()
    assert math.isclose(x1[0], -3.462, abs_tol=1e-6),   "x coordinate mismatch"
    assert math.isclose(y1[0], 69.119, abs_tol=1e-6),   "y coordinate mismatch" 
    assert math.isclose(z1[0], -8.662, abs_tol=1e-6),   "z coordinate mismatch"
    assert ff1[0].strip() == "N",                       "form factor type mismatch"

    # check correct form factors with implicit hydrogens
    ausaxs.settings.molecule(implicit_hydrogens=True)
    mol2 = ausaxs.create_molecule("tests/files/2epe.pdb")
    x2, y2, z2, w2, ff2 = mol2.atoms()
    assert math.isclose(x1[0], -3.462, abs_tol=1e-6),   "x coordinate mismatch"
    assert math.isclose(y1[0], 69.119, abs_tol=1e-6),   "y coordinate mismatch" 
    assert math.isclose(z1[0], -8.662, abs_tol=1e-6),   "z coordinate mismatch"
    assert ff2[0].strip() == "NH",                      "form factor type mismatch"

    # create molecule from PDBfile
    pdb = ausaxs.read_pdb("tests/files/2epe.pdb")
    mol3 = ausaxs.create_molecule(pdb)
    x3, y3, z3, w3, ff3 = mol3.atoms()
    assert np.allclose(x3, x2, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(y3, y2, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(z3, z2, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(w3, w2, atol=1e-6),      "Molecule weights should match PDB reader"
    assert np.array_equal(ff3, ff2),            "Molecule form factor types should match PDB reader"

    # create molecule from coordinates
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0])
    mol4 = ausaxs.create_molecule(x, y, z, weights)
    x4, y4, z4, w4, ff4 = mol4.atoms()
    assert np.allclose(x4, x, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(y4, y, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(z4, z, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(w4, weights, atol=1e-6), "Molecule weights should match input"


def test_hydrate():
    mol = ausaxs.create_molecule("tests/files/2epe.pdb")
    mol.clear_hydration()
    assert len(mol.waters()[0]) == 0, "Should have no hydration waters after clear_hydration"
    mol.hydrate()
    assert len(mol.waters()[0]) > 0, "Should have hydration waters after hydrate"

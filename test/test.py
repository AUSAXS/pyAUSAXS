import sys
import math
import numpy as np
import pyausaxs as ausaxs

class simple_cube:
    @staticmethod
    def points():
        corners = [
            (-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1),
            (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)
            (0, 0, 0)
        ]
        w = np.ones(len(corners), dtype=float)
        xs = np.array([p[0] for p in corners], dtype=float)
        ys = np.array([p[1] for p in corners], dtype=float)
        zs = np.array([p[2] for p in corners], dtype=float)
        return xs, ys, zs, w
    
    @staticmethod
    def hist(): 
        return [
            [0.0, math.sqrt(3.0), 2.0, math.sqrt(8.0), math.sqrt(12.0)],
            [9, 16, 24, 24, 8]
        ]

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

def test_read_datafile():
    # first two lines of 2epe.dat:
    # 9.81300045E-03 6.67934353E-03 1.33646582E-03 1
    # 1.06309997E-02 7.27293547E-03 1.01892441E-03 1
    data = ausaxs.read_data("test/2epe.dat")
    q, I, Ierr = data.data()
    assert math.isclose(q[0], 9.81300045E-03, abs_tol=1e-6),    "First q value mismatch"
    assert math.isclose(I[0], 6.67934353E-03, abs_tol=1e-6),    "First I value mismatch"
    assert math.isclose(Ierr[0], 1.33646582E-03, abs_tol=1e-6), "First Ierr value mismatch"
    assert math.isclose(q[1], 1.06309997E-02, abs_tol=1e-6),    "Second q value mismatch"
    assert math.isclose(I[1], 7.27293547E-03, abs_tol=1e-6),    "Second I value mismatch"
    assert math.isclose(Ierr[1], 1.01892441E-03, abs_tol=1e-6), "Second Ierr value mismatch"

def test_read_pdbfile():
    # first line of 2epe.pdb (ignoring header stuff):
    # ATOM      1  N   LYS A   1      -3.462  69.119  -8.662  1.00 19.81           N  
    pdb = ausaxs.read_pdb("test/2epe.pdb")
    serial, name, altloc, resname, chain_id, resseq, icode, x, y, z, occupancy, tempFactor, element, charge = pdb.data()
    assert serial[0] == 1,                                  "serial number mismatch"
    assert name[0].strip() == "N",                          "atom name mismatch"
    assert altloc[0].strip() == "",                         "altLoc mismatch"
    assert resname[0].strip() == "LYS",                     "resName mismatch"
    assert chain_id[0].strip() == "A",                      "chainID mismatch"
    assert resseq[0] == 1,                                  "resSeq mismatch"
    assert icode[0].strip() == "",                          "iCode mismatch"
    assert math.isclose(x[0], -3.462, abs_tol=1e-6),        "x coordinate mismatch"
    assert math.isclose(y[0], 69.119, abs_tol=1e-6),        "y coordinate mismatch" 
    assert math.isclose(z[0], -8.662, abs_tol=1e-6),        "z coordinate mismatch"
    assert math.isclose(occupancy[0], 1.00, abs_tol=1e-6),  "occupancy mismatch"
    assert math.isclose(tempFactor[0], 19.81, abs_tol=1e-6),"tempFactor mismatch"
    assert element[0].strip() == "N",                       "element mismatch"
    assert charge[0].strip() == "",                         "charge mismatch"

def test_read_ciffile():
    # first data line of 6LYZ.cif:
    # ATOM   1    N N   . LYS A 1 1   ? 3.287   10.092 10.329 1.00 5.89  ? 1   LYS A N   1 
    pdb = ausaxs.read_pdb("test/6LYZ.cif")
    serial, name, altloc, resname, chain_id, resseq, icode, x, y, z, occupancy, tempFactor, element, charge = pdb.data()
    assert serial[0] == 1,                                  "serial number mismatch"
    assert name[0].strip() == "N",                          "atom name mismatch"
    assert altloc[0].strip() == ".",                        "altLoc mismatch"
    assert resname[0].strip() == "LYS",                     "resName mismatch"
    assert chain_id[0].strip() == "A",                      "chainID mismatch"
    assert resseq[0] == 1,                                  "resSeq mismatch"
    assert icode[0].strip() == "?",                         "iCode mismatch"
    assert math.isclose(x[0], 3.287, abs_tol=1e-6),         "x coordinate mismatch"
    assert math.isclose(y[0], 10.092, abs_tol=1e-6),        "y coordinate mismatch" 
    assert math.isclose(z[0], 10.329, abs_tol=1e-6),        "z coordinate mismatch"
    assert math.isclose(occupancy[0], 1.00, abs_tol=1e-6),  "occupancy mismatch"
    assert math.isclose(tempFactor[0], 5.89, abs_tol=1e-6), "tempFactor mismatch"
    assert element[0].strip() == "N",                       "element mismatch"
    assert charge[0].strip() == "?",                        "charge mismatch"

def test_molecule():
    # first line of 2epe.pdb (ignoring header stuff):
    # ATOM      1  N   LYS A   1      -3.462  69.119  -8.662  1.00 19.81           N  
    mol1 = ausaxs.create_molecule("test/2epe.pdb")
    x1, y1, z1, w1, ff1 = mol1.atoms()
    assert math.isclose(x1[0], -3.462, abs_tol=1e-6),   "x coordinate mismatch"
    assert math.isclose(y1[0], 69.119, abs_tol=1e-6),   "y coordinate mismatch" 
    assert math.isclose(z1[0], -8.662, abs_tol=1e-6),   "z coordinate mismatch"
    assert ff1[0].strip() == "NH",                      "form factor type mismatch"

    # create molecule from PDBfile
    pdb = ausaxs.read_pdb("test/2epe.pdb")
    mol2 = ausaxs.create_molecule(pdb)
    x2, y2, z2, w2, ff2 = mol2.atoms()
    assert np.allclose(x2, x1, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(y2, y1, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(z2, z1, atol=1e-6),      "Molecule coordinates should match PDB reader"
    assert np.allclose(w2, w1, atol=1e-6),      "Molecule weights should match PDB reader"
    assert np.array_equal(ff2, ff1),            "Molecule form factor types should match PDB reader"

    # create molecule from coordinates
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0])
    mol2 = ausaxs.create_molecule(x, y, z, weights)
    x2, y2, z2, w2, ff2 = mol2.atoms()
    assert np.allclose(x2, x, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(y2, y, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(z2, z, atol=1e-6),       "Molecule coordinates should match input"
    assert np.allclose(w2, weights, atol=1e-6), "Molecule weights should match input"

def test_histogram():
    atoms = simple_cube.points()
    hist_expected = simple_cube.hist()
    mol = ausaxs.create_molecule(*atoms)
    hist_native = mol.histogram() # hist should be its own class probably, with easy access to a total() method

if __name__ == '__main__':
    import pyausaxs
    print(f"AUSAXS version {pyausaxs.__version__}")
    # test_cube_debye()
    # test_pdb_reader()
    # test_singleton()
    # test_reset_singleton()
    # test_fit()
    test_read_pdbfile()
    test_read_ciffile()
    test_read_datafile()
    test_molecule()
    print("All tests passed")
    sys.exit(0)
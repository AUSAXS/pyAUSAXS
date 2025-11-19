import math
import pyausaxs as ausaxs


def test_read_datafile():
    data = ausaxs.read_data("tests/files/2epe.dat")
    q, I, Ierr = data.data()
    assert math.isclose(q[0], 9.81300045E-03, abs_tol=1e-6),    "First q value mismatch"
    assert math.isclose(I[0], 6.67934353E-03, abs_tol=1e-6),    "First I value mismatch"
    assert math.isclose(Ierr[0], 1.33646582E-03, abs_tol=1e-6), "First Ierr value mismatch"
    assert math.isclose(q[1], 1.06309997E-02, abs_tol=1e-6),    "Second q value mismatch"
    assert math.isclose(I[1], 7.27293547E-03, abs_tol=1e-6),    "Second I value mismatch"
    assert math.isclose(Ierr[1], 1.01892441E-03, abs_tol=1e-6), "Second Ierr value mismatch"


def test_read_pdbfile():
    pdb = ausaxs.read_pdb("tests/files/2epe.pdb")
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
    pdb = ausaxs.read_pdb("tests/files/Ag_crystal.cif")
    cell_a, cell_b, cell_c = 26.3448, 26.3448, 26.3448
    coords = pdb.coordinates()
    assert math.isclose(coords[0][0], 0.194097*cell_a, abs_tol=1e-3), "First atom x coordinate mismatch"
    assert math.isclose(coords[1][0], 0.424123*cell_b, abs_tol=1e-3), "First atom y coordinate mismatch"
    assert math.isclose(coords[2][0], 0.424123*cell_c, abs_tol=1e-3), "First atom z coordinate mismatch"
    assert math.isclose(coords[0][1], 0.269368*cell_a, abs_tol=1e-3), "Second atom x coordinate mismatch"
    assert math.isclose(coords[1][1], 0.346183*cell_b, abs_tol=1e-3), "Second atom y coordinate mismatch"
    assert math.isclose(coords[2][1], 0.422698*cell_c, abs_tol=1e-3), "Second atom z coordinate mismatch"
    ausaxs.settings.molecule(implicit_hydrogens=True)

    pdb = ausaxs.read_pdb("tests/files/6LYZ.cif")
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

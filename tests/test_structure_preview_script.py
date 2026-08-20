"""The structure pane builds its own script to render the preview from.

It must carry the structure — the load block and every element that reshapes the bodies — and nothing else. Run-time elements
have no effect on what is drawn, but they are still parsed, so one that no longer suits the current structure (symmetry
amplitudes left over from a structure that had symmetries) would otherwise take the whole preview down with it.
"""

import pytest

structure_pane = pytest.importorskip("pyausaxs.gui.structure_pane")

strip = structure_pane._structural_only

SCRIPT = """output /tmp/out/
load {
    pdb /tmp/p.pdb
    saxs /tmp/p.dat
    split 20
}
save initial_state.pdb
symmetry c2
rename b1 left
parameter {
    iterations 100
    sym_translate 5
    sym_rotate 1.5
}
print "Initial chi2: {chi2_no_penalty}"
loop
    optimize_once
    end
end
save final_state.pdb
"""


def test_the_load_block_survives_whole():
    kept = strip(SCRIPT)
    assert "load {" in kept
    assert "pdb /tmp/p.pdb" in kept
    assert "saxs /tmp/p.dat" in kept   # the load element requires it, structural or not
    assert "split 20" in kept


def test_structural_elements_survive():
    kept = strip(SCRIPT)
    assert "symmetry c2" in kept
    assert "rename b1 left" in kept


def test_the_parameter_element_is_dropped():
    # the bug this filter exists for: sym_* amplitudes on a structure with no symmetries are refused by the
    # backend, which failed the preview even though the element cannot affect it
    kept = strip(SCRIPT)
    assert "parameter" not in kept
    assert "sym_translate" not in kept
    assert "iterations" not in kept


@pytest.mark.parametrize("dropped", ["output", "save", "print", "loop", "optimize_once", "end"])
def test_run_only_elements_are_dropped(dropped):
    assert dropped not in strip(SCRIPT)


def test_declaration_order_is_preserved():
    # setup elements are applied in the order the backend reads them, so a rename after a symmetry must stay after it
    kept = strip(SCRIPT)
    assert kept.index("load {") < kept.index("symmetry c2") < kept.index("rename b1 left")


def test_every_kept_element_is_on_its_own_line():
    kept = strip(SCRIPT)
    assert kept.endswith("\n")
    assert "\n\n" not in kept


def test_indented_elements_are_kept():
    # the pane writes its own elements unindented, but a hand-written script may indent them
    kept = strip("load {\n    pdb /tmp/p.pdb\n}\n    symmetry c2\n    merge b1 b2\n")
    assert "symmetry c2" in kept
    assert "merge b1 b2" in kept


def test_constraints_are_kept():
    # constraints are drawn in the preview, so they are structural as far as this pane is concerned
    kept = strip("load {\n    pdb /tmp/p.pdb\n}\nautoconstrain\nconstrain {\n    b1 b2\n}\nparameter {\n iterations 5\n}\n")
    assert "autoconstrain" in kept
    assert "constrain {" in kept
    assert "b1 b2" in kept
    assert "parameter" not in kept


def test_a_script_with_nothing_structural_yields_nothing():
    assert strip("print \"hello\"\nsave out.pdb\n") == ""

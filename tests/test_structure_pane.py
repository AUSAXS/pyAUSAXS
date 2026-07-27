from pyausaxs.gui.structure_pane import _insert_elements, _structure_signature
from pyausaxs.gui.rigidbody_pane import _STRUCTURALLY_RELEVANT_RE


BASE_SCRIPT = """load {
    pdb structure.pdb
    saxs data.dat
}
autoconstrain backbone
"""

PARAMETER_GENERATOR = """parameter_generator {
    iterations 100
    translate 1
    rotate 1
	symmetry 1
}
"""


def test_staged_structure_elements_precede_existing_constraints():
    result = _insert_elements(
        BASE_SCRIPT,
        ["rename b1 protein", "split b1 100", "symmetry b1 {", "    copy b1", "}"],
    )

    constraint_at = result.index("autoconstrain backbone")
    assert result.index("rename b1 protein") < constraint_at
    assert result.index("split b1 100") < constraint_at
    assert result.index("symmetry b1") < constraint_at
    assert result.index("copy b1") < constraint_at


def test_staged_constraints_remain_after_existing_setup():
    base = BASE_SCRIPT + "rename b1 existing\n"

    result = _insert_elements(base, ["split b1 100", "constrain b1 b2"])

    assert result.index("split b1 100") < result.index("autoconstrain backbone")
    assert result.index("constrain b1 b2") > result.index("rename b1 existing")


def test_parameter_generator_symmetry_is_not_structural():
    script = BASE_SCRIPT.replace("autoconstrain", PARAMETER_GENERATOR + "autoconstrain")
    signature = _structure_signature(script)

    assert "symmetry 1" not in "".join(signature)


def test_parameter_generator_does_not_affect_setup_anchor():
    base = PARAMETER_GENERATOR + BASE_SCRIPT

    result = _insert_elements(base, ["split b1 100"])

    assert result.index("split b1 100") < result.index("autoconstrain backbone")


def test_parameter_generator_symmetry_does_not_trigger_rigidbody_preview():
    parameter_only = PARAMETER_GENERATOR
    top_level_symmetry = "symmetry b1 c4\n"

    assert not list(_STRUCTURALLY_RELEVANT_RE.finditer(parameter_only))
    assert [match.group(0) for match in _STRUCTURALLY_RELEVANT_RE.finditer(top_level_symmetry)] == [
        top_level_symmetry.rstrip("\n")
    ]
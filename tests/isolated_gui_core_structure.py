"""The structure-management model driven against the real backend: every action rebuilds the structure through it.

Not collected directly (no test_ prefix): test_isolated.py runs it in a child process — see isolated_gui_core_runner.py for why."""

import os

from pyausaxs.gui.core import structure

FILES = os.path.join(os.path.dirname(__file__), "files")
PDB, DAT = os.path.join(FILES, "2epe.pdb"), os.path.join(FILES, "2epe.dat")

BASE = f"""output out/
load {{
    pdb {PDB}
    saxs {DAT}
    split 50 100
}}
parameters {{
    iterations 10
}}
"""


def make_model(base=BASE):
    holder = {"script": base}
    model = structure.StructureModel(PDB, splits="50 100", base_script=lambda: holder["script"])
    ok, msg = model.rebuild()
    assert ok, msg
    return model, holder


def names(model):
    return [b["name"] for b in model.bodies]


def test_initial_build_reads_the_bodies():
    model, _ = make_model()
    assert names(model) == ["b1", "b2", "b3"]
    assert model.built and not model.is_stale()


def test_merge_stages_an_element_and_drops_the_body():
    model, _ = make_model()
    result = model.apply_merge("b1 b2")
    assert result == structure.ActionResult(True, "Applied: merge b1 b2", rebuilt=True)
    assert model.elements == ["merge b1 b2"]
    assert names(model) == ["b1", "b3"]


def test_a_rejected_element_changes_nothing():
    model, _ = make_model()
    result = model.apply_delete("no_such_body")
    assert not result.ok and not result.rebuilt
    assert model.elements == [] and names(model) == ["b1", "b2", "b3"]


def test_merge_from_the_selection_then_remove_it_again():
    model, _ = make_model()
    model.toggle_highlight(0, None)
    model.toggle_highlight(2, None, ctrl=True)
    assert model.apply_merge("").ok
    assert model.elements == ["merge b1 b3"]
    assert model.highlighted == set()  # the merge renumbered the bodies, so the selection can't survive it
    assert model.remove_element(0).ok
    assert names(model) == ["b1", "b2", "b3"]


def test_rename_keeps_the_selection():
    model, _ = make_model()
    model.toggle_highlight(1, None)
    assert model.apply_rename("core").ok
    assert names(model) == ["b1", "core", "b3"]
    assert model.highlighted == {(1, None)}


def test_resplit_reverts_on_a_bad_value():
    model, _ = make_model()
    result, shown = model.apply_splits("50")
    assert result.ok and shown == "50" and len(model.bodies) == 2
    result, shown = model.apply_splits("1000000")
    assert not result.ok and shown == "50" and model.splits == "50"


def test_body_split_from_clicked_residues_extends_its_own_element():
    model, _ = make_model()
    model.toggle_selected_residue(0, 20)
    assert model.apply_body_split("").ok
    assert model.elements == ["split b1 20"]
    assert model.selected_residues == set()
    # a residue in one of the fragments is still the original body's business, so the staged element is extended rather than duplicated
    fragment = next(b["index"] for b in model.bodies if b["res"] and b["res"][0] <= 35 <= b["res"][1])
    model.toggle_selected_residue(fragment, 35)
    result = model.apply_body_split("")
    assert result.ok and result.message.startswith("Extended:")
    assert model.elements == ["split b1 20 35"]


def test_staleness_and_refresh():
    model, holder = make_model()
    model.apply_merge("b1 b2")
    holder["script"] = BASE.replace("split 50 100", "split 50")
    assert model.is_stale()
    ok, _ = model.refresh_from_base()
    assert ok and model.elements == [] and model.splits == "50" and not model.is_stale()


def test_send_script_keeps_the_rest_of_the_script():
    model, holder = make_model()
    assert model.send_script() is None
    model.apply_merge("b1 b2")
    old, new = model.send_script()
    assert old == BASE and "merge b1 b2" in new and "iterations 10" in new
    holder["script"] = new
    ok, _ = model.sent()
    assert ok and model.elements == [] and names(model) == ["b1", "b3"]

"""The toolkit-independent GUI core: script rewriting, syntax classification, and input-file companions.

These run without a display or tkinter, since the core is shared with front-ends built on other toolkits."""

import os

import pytest

from pyausaxs.gui.core import files, script, syntax

SCRIPT = """output out/
load {
    pdb /tmp/p.pdb
    saxs /tmp/p.dat
    split 20, 40
}
symmetry c2
parameters {
    iterations 100
}
loop
    optimize_once
    end
end
"""


# ── load block ─────────────────────────────────────────────────────────────
def test_load_value_reads_unquoted_directives():
    assert script.load_value(SCRIPT, "pdb") == "/tmp/p.pdb"
    assert script.load_value('load {\n    pdb "/my dir/p.pdb"\n}', "pdb") == "/my dir/p.pdb"
    assert script.load_value(SCRIPT, "missing") is None
    assert script.load_value("print 1\n", "pdb") is None


def test_set_load_directive_replaces_only_its_own_line():
    new = script.set_load_directive(SCRIPT, "saxs", "/tmp/other.dat")
    assert script.load_value(new, "saxs") == "/tmp/other.dat"
    assert script.load_value(new, "pdb") == "/tmp/p.pdb"
    assert new.endswith(SCRIPT[SCRIPT.index("symmetry c2"):])


def test_set_load_directive_quotes_paths_with_spaces():
    new = script.set_load_directive(SCRIPT, "pdb", "/my dir/p.pdb")
    assert 'pdb "/my dir/p.pdb"' in new
    assert script.load_value(new, "pdb") == "/my dir/p.pdb"


def test_set_load_directive_with_an_empty_value_removes_it():
    new = script.set_load_directive(SCRIPT, "split", "")
    assert script.load_value(new, "split") is None


def test_set_load_directive_creates_a_missing_load_block():
    new = script.set_load_directive("print 1\n", "pdb", "/tmp/p.pdb")
    assert new.startswith("load {\n    pdb /tmp/p.pdb\n}\n")
    assert script.set_load_directive("print 1\n", "pdb", "") is None


def test_with_split_touches_only_the_split_line():
    new = script.with_split(SCRIPT, "10 30")
    assert script.load_split(new) == "10 30"
    assert new.replace("split 10 30", "split 20, 40") == SCRIPT


def test_with_split_is_a_no_op_for_equal_splits():
    assert script.with_split(SCRIPT, "20 40") is SCRIPT


# ── splits and signatures ──────────────────────────────────────────────────
def test_split_residues_combines_the_load_block_and_split_elements():
    # duplicates are allowed: the split element pattern also matches the load block's own split line, and draw_structure deduplicates
    assert sorted(set(script.split_residues(SCRIPT + "split b1 70 90\n"))) == [20, 40, 70, 90]


def test_preview_signature_ignores_run_time_edits():
    edited = SCRIPT.replace("iterations 100", "iterations 5")
    assert script.preview_signature(edited) == script.preview_signature(SCRIPT)
    assert script.preview_signature(SCRIPT.replace("symmetry c2", "symmetry c3")) != script.preview_signature(SCRIPT)


def test_structure_signature_covers_body_renames():
    renamed = SCRIPT + "rename b1 core\n"
    assert script.structure_signature(renamed) != script.structure_signature(SCRIPT)


def test_has_live_update():
    assert not script.has_live_update(SCRIPT)
    assert script.has_live_update(script.DEFAULT_RIGIDBODY_SCRIPT)


# ── structure-management scripts ──────────────────────────────────────────
def test_staged_structural_elements_precede_existing_constraints():
    base = "load {\n    pdb p.pdb\n}\nsymmetry c2\nautoconstrain\nprint 1\n"
    new = script.insert_elements(base, ["merge b1 b2", "constrain b1 b3"])
    assert new.index("symmetry c2") < new.index("merge b1 b2") < new.index("autoconstrain") < new.index("constrain b1 b3")


def test_preview_script_is_structural_only():
    preview = script.structure_preview_script(SCRIPT, "/tmp/p.pdb", "20 40", ["rename b1 core"])
    assert "rename b1 core" in preview
    assert "iterations" not in preview


def test_preview_script_falls_back_to_a_synthetic_load_block():
    preview = script.structure_preview_script(None, "/my dir/p.pdb", "15", [])
    assert 'pdb "/my dir/p.pdb"' in preview
    assert "split 15" in preview


def test_send_script_keeps_every_element():
    old, new = script.structure_send_script(SCRIPT, "/tmp/p.pdb", "20 40", ["rename b1 core"])
    assert old is SCRIPT
    assert "rename b1 core" in new
    assert "iterations 100" in new


def test_send_script_without_changes_is_unchanged():
    old, new = script.structure_send_script(SCRIPT, "/tmp/p.pdb", "20, 40", [])
    assert old == new


# ── whole-script rewrites ──────────────────────────────────────────────────
def test_reset_keeps_the_load_block():
    reset = script.reset_script(SCRIPT)
    assert script.LOAD_BLOCK_RE.search(reset).group(0) == script.LOAD_BLOCK_RE.search(SCRIPT).group(0)
    assert "symmetry c2" not in reset


def test_reset_keeps_windows_paths_intact():
    # the load block is inserted verbatim: as a regex replacement string, `\U` would raise and `\n` would become a line break
    current = 'load {\n    pdb "C:\\Users\\John Doe\\new.pdb"\n}\n'
    assert 'pdb "C:\\Users\\John Doe\\new.pdb"' in script.reset_script(current)


def test_reset_without_a_load_block_gives_the_default():
    assert script.reset_script("print 1\n") == script.DEFAULT_RIGIDBODY_SCRIPT


def test_absolutize_script_paths(tmp_path):
    text = 'output out/\nload {\n    pdb data/p.pdb\n    saxs "my data/p.dat"\n}\n'
    new = script.absolutize_script_paths(text, str(tmp_path))
    assert script.load_value(new, "pdb") == os.path.join(str(tmp_path), "data", "p.pdb")
    assert script.load_value(new, "saxs") == os.path.join(str(tmp_path), "my data", "p.dat")
    assert script.OUTPUT_RE.search(new).group(2).endswith(os.sep)


def test_absolutize_leaves_absolute_paths_alone():
    absolute = SCRIPT.replace("output out/", "output /tmp/out/")
    assert script.absolutize_script_paths(absolute, "/elsewhere") == absolute


# ── syntax ─────────────────────────────────────────────────────────────────
OPS, KWS = {"load", "print", "save"}, {"pdb"}


def test_classify_line_kinds():
    assert syntax.classify_line("print 1 # note", 0, OPS, KWS)[0] == [
        syntax.Span(8, 14, "comment"), syntax.Span(0, 5, "operation")]
    assert syntax.classify_line("    pdb x", 0, OPS, KWS)[0] == [syntax.Span(4, 7, "keyword")]
    assert syntax.classify_line("bogus", 0, OPS, KWS)[0] == [syntax.Span(0, 5, "error")]
    assert syntax.classify_line("}", 0, OPS, KWS)[0] == []


def test_scopes_nest_and_end_shares_its_openers_depth():
    depth, kinds = 0, []
    for line in ("loop", "optimize_once", "end", "end"):
        spans, depth = syntax.classify_line(line, depth, OPS, KWS)
        kinds.append((spans[0].kind, spans[0].depth))
    assert kinds == [("scope", 0), ("scope", 1), ("scope", 1), ("scope", 0)]
    assert depth == 0


def test_scope_partner():
    lines = ["loop", "  optimize_once", "  end", "end", "print 1"]
    assert syntax.scope_partner(lines, 0) == 3
    assert syntax.scope_partner(lines, 3) == 0
    assert syntax.scope_partner(lines, 1) == 2
    assert syntax.scope_partner(lines, 2) == 1
    assert syntax.scope_partner(lines, 4) is None
    assert syntax.scope_partner(["loop"], 0) is None


# ── files ──────────────────────────────────────────────────────────────────
def test_find_companion_prefers_the_same_stem(tmp_path):
    (tmp_path / "p.pdb").touch()
    (tmp_path / "p.dat").touch()
    (tmp_path / "q.dat").touch()
    assert files.find_companion(str(tmp_path / "p.pdb"), files.SAXS_EXTENSIONS) == str(tmp_path / "p.dat")


def test_find_companion_returns_a_full_path_for_a_lone_candidate(tmp_path, monkeypatch):
    # the companion is written into the script, which the backend resolves against the working directory, not the structure's
    (tmp_path / "p.pdb").touch()
    (tmp_path / "data.dat").touch()
    monkeypatch.chdir(tmp_path.parent)
    assert files.find_companion(str(tmp_path / "p.pdb"), files.SAXS_EXTENSIONS) == str(tmp_path / "data.dat")


def test_find_companion_refuses_ambiguity(tmp_path):
    (tmp_path / "p.pdb").touch()
    (tmp_path / "a.dat").touch()
    (tmp_path / "b.dat").touch()
    assert files.find_companion(str(tmp_path / "p.pdb"), files.SAXS_EXTENSIONS) is None


# ── structure model: selections (no backend needed) ───────────────────────
from pyausaxs.gui.core import structure


def model_with_bodies(*copies_per_body):
    """A model whose body list is filled in by hand, as a rebuild would, so the selection logic can be checked without the backend."""
    model = structure.StructureModel("/tmp/p.pdb")
    model.bodies = [{"index": i, "name": f"b{i + 1}", "atoms": 10 * (i + 1), "res": (1, 10), "colour": "#000", "copies": list(range(n))}
                    for i, n in enumerate(copies_per_body)]
    model.replica_info = {(i, c): {"type": "c2", "name": f"b{i + 1}r{c}"} for i, n in enumerate(copies_per_body) for c in range(1, n)}
    return model


def test_plain_click_selects_and_deselects():
    model = model_with_bodies(1, 1)
    model.toggle_highlight(0, None)
    assert model.highlighted == {(0, None)}
    model.toggle_highlight(1, None)
    assert model.highlighted == {(1, None)}
    model.toggle_highlight(1, None)
    assert model.highlighted == set()


def test_ctrl_click_adds_and_shift_click_selects_a_range():
    model = model_with_bodies(1, 1, 1, 1)
    model.toggle_highlight(0, None)
    model.toggle_highlight(2, None, ctrl=True)
    assert model.selected_body_names() == ["b1", "b3"]
    model.toggle_highlight(3, None, shift=True)
    assert model.selected_body_names() == ["b3", "b4"]


def test_shift_range_follows_the_visual_order_of_unfolded_replicas():
    # the replicas of an unfolded body sit directly beneath it, so a range ending on the next body passes through them
    model = model_with_bodies(3, 1)
    model.expanded_bodies.add(0)
    assert model.row_order() == [(0, None), (0, 1), (0, 2), (1, None)]
    model.toggle_highlight(0, 1)
    model.toggle_highlight(1, None, shift=True)
    assert model.highlighted == {(0, 1), (0, 2), (1, None)}


def test_selecting_a_replica_asks_for_copies_to_be_shown():
    model = model_with_bodies(2)
    assert model.toggle_highlight(0, 1) is True
    assert model.toggle_highlight(0, None) is False


def test_a_body_and_its_own_replica_can_be_selected_together():
    model = model_with_bodies(2, 1)
    model.toggle_highlight(0, None)
    model.toggle_highlight(0, 1, ctrl=True)
    assert model.selected_body_names() == ["b1"]  # replicas don't count as whole bodies


def test_selection_stands_in_for_typed_bodies():
    model = model_with_bodies(1, 1, 1)
    model.toggle_highlight(0, None)
    model.toggle_highlight(1, None, ctrl=True)
    assert model.with_selected_bodies("cm", exact=2) == ["b1", "b2", "cm"]
    assert model.with_selected_bodies("cm", exact=1) == ["cm"]
    assert model.with_selected_bodies("b3 b1 cm", exact=2) == ["b3", "b1", "cm"]  # typed bodies win
    assert model.can_merge("") and model.can_delete("")
    assert not model.can_rename("core")
    model.toggle_highlight(0, None)  # plain click: just b1
    assert model.can_rename("core")


def test_validation_messages_need_no_backend():
    model = model_with_bodies(1, 1)
    assert model.apply_merge("b1") == structure.ActionResult(False, "Merge needs a target body and at least one other.")
    assert not model.apply_add_constraint("b1 b2 attract").ok
    assert not model.apply_convert_symmetry("b1 c2", tolerance="wide").ok
    assert model.rename("b1", "b1") == structure.ActionResult(True)
    assert model.rename("b1", "my core") == structure.ActionResult(False, "A name cannot contain spaces.")
    assert not model.apply_body_split("b1 ten").ok


def test_side_by_side_diff_aligns_both_sides():
    rows = structure.side_by_side_diff("a\nb\nc\n", "a\nx\ny\nc\n")
    assert rows[0] == structure.DiffRow("a", False, "a", False)
    assert [(r.old, r.old_changed, r.new, r.new_changed) for r in rows[1:3]] == [("b", True, "x", True), ("", False, "y", True)]
    assert rows[-1] == structure.DiffRow("c", False, "c", False)

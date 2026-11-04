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

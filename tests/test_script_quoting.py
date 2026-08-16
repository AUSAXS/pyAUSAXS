"""Sequencer-script values must survive the trip through the GUI.

The backend splits a script line on whitespace unless the value is quoted, so every path the GUI writes into
a script has to be quoted when it contains a space — which on Windows is the normal case, since the user
profile directory routinely contains one.
"""

import pytest

widgets = pytest.importorskip("pyausaxs.gui.widgets")
rigidbody_pane = pytest.importorskip("pyausaxs.gui.rigidbody_pane")
structure_pane = pytest.importorskip("pyausaxs.gui.structure_pane")

quote = widgets.quote_script_value
unquote = widgets.unquote_script_value

WINDOWS_DIR = "C:\\Users\\John Doe\\out\\"
POSIX_FILE = "/home/me/my folder/protein.pdb"


@pytest.mark.parametrize("value", [WINDOWS_DIR, POSIX_FILE, "plain.pdb", "C:\\out\\", ""])
def test_quote_round_trip(value):
    assert unquote(quote(value)) == value


@pytest.mark.parametrize("value", [WINDOWS_DIR, POSIX_FILE])
def test_values_with_spaces_are_quoted(value):
    assert quote(value) == f'"{value}"'


@pytest.mark.parametrize("value", ["plain.pdb", "C:\\out\\", "relative/out/"])
def test_values_without_spaces_are_left_alone(value):
    assert quote(value) == value


def test_quoting_is_idempotent():
    assert quote(quote(POSIX_FILE)) == quote(POSIX_FILE)


def test_unquoting_a_bare_value_is_a_no_op():
    assert unquote(POSIX_FILE) == POSIX_FILE


def test_path_directives_are_quoted_but_splits_are_not():
    assert rigidbody_pane._directive_value("pdb", POSIX_FILE) == f'"{POSIX_FILE}"'
    assert rigidbody_pane._directive_value("saxs", POSIX_FILE) == f'"{POSIX_FILE}"'
    # split is a list of residue ids: quoting it would hand the backend one token instead of several
    assert rigidbody_pane._directive_value("split", "10, 20") == "10, 20"


@pytest.mark.parametrize("line, expected", [
    ("output out/rigidbody/", "out/rigidbody/"),
    (f'output "{WINDOWS_DIR}"', WINDOWS_DIR),
    ("output 'my out/'", "my out/"),
])
def test_output_directive_captures_quoted_paths(line, expected):
    match = rigidbody_pane._OUTPUT_RE.search(line)
    assert match is not None
    assert unquote(match.group(2)) == expected


def test_synthetic_load_block_quotes_the_structure_path():
    block = structure_pane._synth_load_block(POSIX_FILE, "10")
    assert f'pdb "{POSIX_FILE}"' in block
    assert "split 10" in block

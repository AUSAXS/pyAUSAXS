# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Reading and rewriting rigid-body sequencer scripts.

Every function here takes a script (or a fragment of one) as a string and returns a new string; none of them touch an editor widget, so the
script stays the single source of truth regardless of the front-end showing it.
"""

import os
import re
from typing import Optional

DEFAULT_RIGIDBODY_SCRIPT = """\
output output/rigidbody/
load {
    pdb
    saxs
}
save initial_state.pdb
save trajectory.xyz
parameters {
    iterations 100
    translate 1
    rotate 1
}

print "Initial chi2: {chi2_no_penalty}"
loop
    optimize_once
        on_improvement
            print {
                msg "{iteration}/{iterations_total}: Accepted with new chi2 {chi2_no_penalty}"
                colour green
            }
            update structure
            save trajectory.xyz
        end
    end
end
save final_state.pdb
"""

# ── regex building blocks ──────────────────────────────────────────────────
# one or more spaces/tabs — separates a directive from its argument
_TABS_OR_SPACES = r"[ \t]+"
# a brace block `{ ... }` (non-greedy; spans newlines under re.DOTALL)
_MANDATORY_BLOCK_FORMAT = r"\s*\{.*?\}"
# a directive body that is either a brace block or a single inline line
_OPTIONAL_BLOCK_FORMAT = r"(?:\s*\{.*?\}|[^\n]*)"
# start of a line, skipping any leading indentation (needs re.MULTILINE)
_LINE_START = r"^[ \t]*"

# `load { ... }` — the mandatory brace block listing the bodies to refine
LOAD_BLOCK_RE = re.compile(f"load{_MANDATORY_BLOCK_FORMAT}", re.DOTALL)
# a top-level 'symmetry' element: either a brace block (symmetry { ... }) or a single inline line
# (symmetry c6 / symmetry b1 c6), anchored to column zero so the parameters block's indented
# `symmetry 1` value is not treated as a structural update
_SYMMETRY_RE = re.compile(
    rf"^symmetry\b{_OPTIONAL_BLOCK_FORMAT}", re.MULTILINE | re.DOTALL
)
# a constraint element — autoconstrain/autoconstraints (inline) or constrain/constraint
# (inline or a brace block) — anchored to the first token on a line. The whole { ... } block
# is captured so edits to its arguments (type, bodies, …) refresh the preview too.
_PREVIEW_CONSTRAINT_RE = re.compile(
    rf"{_LINE_START}(?:auto)?constrain(?:ts?)?\b{_OPTIONAL_BLOCK_FORMAT}",
    re.MULTILINE | re.DOTALL,
)
# an 'update' element (e.g. `update structure`) as the first token on a line; its presence makes
# the GUI poll the backend for the live structure during a run
_UPDATE_RE = re.compile(rf"{_LINE_START}update\b", re.MULTILINE)
# a directive's value: either a quoted string — the only way to write a path containing spaces — or a bare
# run of non-space characters
_VALUE = r"\"[^\"\n]*\"|'[^'\n]*'|\S+"
# the top-level 'output' directive (output <dir>), captured as (prefix, path) so the path can be
# rewritten to an absolute one at boot
OUTPUT_RE = re.compile(rf"({_LINE_START}output{_TABS_OR_SPACES})({_VALUE})", re.MULTILINE)
# the load-block directives whose value is a path, and so must be quoted when it contains spaces. 'split'
# is a list of residue ids and has to stay bare.
PATH_DIRECTIVES = ("pdb", "saxs")
# a top-level 'split' element: `split <body> <ids…>` on its own line. This is the sequencer element that
# partitions an already-loaded body into new bodies at the given residue ids (distinct from the load-block
# 'split' directive, which lists ids for the load-time BodySplitter). Both change the body set, so both must
# refresh the preview; anchored to the first token on a line so it won't match 'splitter' or a mid-line word.
_SPLIT_RE = re.compile(rf"{_LINE_START}split\b[^\n]*", re.MULTILINE)
# structurally relevant elements: the load block, symmetry block, constraint blocks, delete element, and the
# split element. Used to determine whether a script edit should trigger a preview redraw.
_DELETE_RE = re.compile(rf"{_LINE_START}delete\b{_OPTIONAL_BLOCK_FORMAT}", re.MULTILINE | re.DOTALL)
_STRUCTURALLY_RELEVANT_RE = re.compile(
    rf"(?:{LOAD_BLOCK_RE.pattern}|{_SYMMETRY_RE.pattern}|{_PREVIEW_CONSTRAINT_RE.pattern}|{_DELETE_RE.pattern}|{_SPLIT_RE.pattern})",
    re.MULTILINE | re.DOTALL
)

# Every element the structure pane reads or writes, so that an external edit to any of them (in the main editor) marks the view stale: the
# load block plus all body-affecting setup elements. The tail captures either a whole brace block or the rest of the inline line, so an edit
# inside a symmetry/constraint block counts too. Longer keywords precede the prefixes they contain.
_STALE_TAIL = r"(?:[ \t]*\{.*?\}|[^\n]*)"
_STALE_RE = re.compile(
    r"(?m)^[ \t]*(?:load[ \t]*\{.*?\}|"
    r"(?:merge|delete|rename|convert_to_symmetry|constraint|constrain"
    r"|autoconstraints|autoconstrain|copy_body|copy|split|symmetry)\b)" + _STALE_TAIL,
    re.DOTALL,
)

# Setup elements are emitted in declaration order by the backend. Newly staged structure-pane elements must therefore be appended after existing
# setup elements rather than blindly inserted immediately after `load`; otherwise a staged rename can precede an existing symmetry declaration.
_SETUP_RE = re.compile(
    r"(?m)^[ \t]*(?:merge|delete|rename|convert_to_symmetry|constraint|constrain"
    r"|autoconstraints|autoconstrain|copy_body|copy|split|symmetry)\b" + _STALE_TAIL,
    re.DOTALL,
)

# Constraint declarations and the staged elements that must remain after existing setup declarations. Every other structure element must
# be declared before the first constraint, because constraints are indexed by the structure that exists when they are declared. See
# insert_elements.
_CONSTRAINT_RE = re.compile(
    r"(?m)^[ \t]*(?:autoconstraints|autoconstrain|constraint|constrain)\b" + _STALE_TAIL, re.DOTALL)
_CONSTRAINT_ELEMENTS = frozenset({"autoconstraints", "autoconstrain", "constraint", "constrain"})


# ── quoting ────────────────────────────────────────────────────────────────
# The backend tokenizes a script line on whitespace unless the value is quoted, so any value containing a space has to be written with
# quotes around it. This is near-universal on Windows, where the user profile directory routinely contains one (C:\Users\John Doe\...).
# The backend treats backslash as a path separator rather than an escape character, so a path ending in one needs no special handling here.
_SCRIPT_QUOTES = ('"', "'")


def _is_quoted(value: str) -> bool:
    return len(value) > 1 and value[0] in _SCRIPT_QUOTES and value[-1] == value[0]


def quote_script_value(value: str) -> str:
    """Quote a value for a sequencer script line if it contains whitespace. Values that are already quoted, or
    that need no quoting, are returned unchanged, so this is safe to apply more than once."""
    value = value.strip()
    if not value or _is_quoted(value):
        return value
    return f'"{value}"' if any(c.isspace() for c in value) else value


def unquote_script_value(value: str) -> str:
    """Strip one matching pair of surrounding quotes from a script value — the inverse of quote_script_value,
    giving back the value as the backend's tokenizer would see it."""
    value = value.strip()
    return value[1:-1] if _is_quoted(value) else value


def directive_value(directive: str, value: str) -> str:
    """Render a load-block value for writing into the script, quoting it if it is a path that needs it."""
    return quote_script_value(value) if directive in PATH_DIRECTIVES else value


# ── the load block ─────────────────────────────────────────────────────────
def _load_block_lines(script: str) -> Optional[list[str]]:
    """The lines inside the script's load block, or None if it has none."""
    match = LOAD_BLOCK_RE.search(script)
    if not match:
        return None
    return re.match(r"load\s*\{(.*)\}", match.group(0), re.DOTALL).group(1).splitlines()


def load_value(script: str, directive: str) -> Optional[str]:
    """Return the argument of a directive in the script's load block, or None. The value is unquoted, i.e. as
    the backend's tokenizer would see it, so it can be handed straight to os.path and to file-input widgets."""
    for line in _load_block_lines(script) or []:
        tokens = line.split(None, 1)
        if tokens and tokens[0] == directive:
            return unquote_script_value(tokens[1]) if len(tokens) == 2 else ""
    return None


def load_split(script: str) -> str:
    """The split directive currently in the script's load block, as a raw string ("" if none)."""
    for line in _load_block_lines(script) or []:
        tokens = line.split(None, 1)
        if tokens and tokens[0] == "split":
            return tokens[1].strip() if len(tokens) == 2 else ""
    return ""


def rewrite_directive(block: str, directive: str, value: str) -> str:
    """Return the load block with `directive` set to `value` (or removed if empty), preserving every other directive line and their order."""
    inner = re.match(r"load\s*\{(.*)\}", block, re.DOTALL).group(1)
    keyword = re.compile(rf"^\s*{re.escape(directive)}\b")
    new_line = f"{directive} {directive_value(directive, value)}" if value else None

    kept: list[str] = []
    replaced = False
    for line in inner.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if keyword.match(line):
            if new_line is not None and not replaced:  # replace in place; drop duplicates
                kept.append(new_line)
                replaced = True
        else:
            kept.append(stripped)
    if new_line is not None and not replaced:  # directive was absent: append it
        kept.append(new_line)

    body = "".join(f"    {line}\n" for line in kept)
    return "load {\n" + body + "}"


def set_load_directive(script: str, directive: str, value: str) -> Optional[str]:
    """Return `script` with a single load directive (pdb/saxs/split) set to `value`, replacing any existing line for that directive and
    leaving the rest of the script untouched. An empty value removes the directive. If no load block exists, one is created. Returns None
    when there is nothing to do (an empty value and no load block to remove it from)."""
    value = value.strip()
    match = LOAD_BLOCK_RE.search(script)
    if match:
        new_block = rewrite_directive(match.group(0), directive, value)
        return script[:match.start()] + new_block + script[match.end():]
    if value:
        return f"load {{\n    {directive} {directive_value(directive, value)}\n}}\n" + script
    return None


def synth_load_block(pdb_path: str, splits: str) -> str:
    """A minimal load block for `pdb_path`, split at `splits`, for when there is no script to take one from."""
    inner = [f"    pdb {quote_script_value(pdb_path)}"]
    if splits.strip():
        inner.append(f"    split {splits.strip()}")
    return "load {\n" + "\n".join(inner) + "\n}"


def with_split(script: str, splits: str) -> str:
    """Return `script` with its load block's split directive set to `splits` (added if missing, removed if empty), touching only the split line
    so unrelated formatting is preserved. A no-op when the split is already `splits`, so a diff shows a change only when the user actually
    re-split."""
    match = LOAD_BLOCK_RE.search(script)
    if not match or norm_splits(load_split(script)) == norm_splits(splits):
        return script
    block = match.group(0)
    splits = splits.strip()
    line_re = re.compile(r"^([ \t]*)split\b[^\n]*", re.MULTILINE)
    if line_re.search(block):
        if splits:
            new_block = line_re.sub(lambda m: f"{m.group(1)}split {splits}", block, count=1)
        else:  # drop the split line (and its own line break) entirely
            new_block = re.sub(r"[ \t]*split\b[^\n]*\n?", "", block, count=1)
    elif splits:  # insert a split line before the closing brace, matching pdb's indentation
        pm = re.search(r"^([ \t]*)pdb\b", block, re.MULTILINE)
        indent = pm.group(1) if pm else "    "
        new_block = re.sub(r"\n?\}$", f"\n{indent}split {splits}\n}}", block, count=1)
    else:
        return script
    return script[:match.start()] + new_block + script[match.end():]


# ── splits ─────────────────────────────────────────────────────────────────
def parse_splits(value: Optional[str]) -> list[int]:
    """The residue ids in a splits string. Non-numeric tokens (e.g. a body name) are skipped."""
    if not value:
        return []
    return [int(t) for t in re.split(r"[,\s]+", value.strip()) if t.isdigit()]


def norm_splits(value: str) -> str:
    """Normalise a splits string to a canonical space-separated form, so equal splits written with different spacing/commas compare equal
    (and don't churn the load block)."""
    return " ".join(t for t in re.split(r"[,\s]+", value.strip()) if t)


def split_residues(script: str) -> list[int]:
    """Residue ids to highlight as split points in a preview: the load-block `split` directive (load-time BodySplitter) together with every
    top-level `split` element (which partitions an already-loaded body). Both forms carry residue ids; non-numeric tokens such as the
    element's body name or the `split` keyword itself are ignored by parse_splits, and draw_structure deduplicates, so a residue matched by
    both forms is marked once."""
    ids = parse_splits(load_value(script, "split"))
    for m in _SPLIT_RE.finditer(script):
        ids += parse_splits(m.group(0))
    return ids


def is_residue_id(token: str) -> bool:
    """Whether `token` is a residue sequence id. Ids may legitimately be negative, so a leading minus is allowed."""
    return token.lstrip("-").isdigit() and token not in ("-", "")


def parse_split_element(element: str) -> tuple[str, list[int]] | None:
    """(target body, cut residues) of a `split` element, or None for any other element."""
    tokens = element.split()
    if len(tokens) < 3 or tokens[0] != "split" or not all(is_residue_id(t) for t in tokens[2:]):
        return None
    return tokens[1], [int(t) for t in tokens[2:]]


# ── signatures ─────────────────────────────────────────────────────────────
def preview_signature(script: str) -> tuple:
    """Distil the parts of the script that affect the structure preview — the load block, any symmetry elements, constraint lines, and
    delete/split elements — so edits to unrelated lines (iterations, print, save, ...) don't trigger a redraw or a backend rebuild."""
    return tuple(m.group(0) for m in _STRUCTURALLY_RELEVANT_RE.finditer(script))


def structure_signature(script: str) -> tuple:
    """A fingerprint of the parts of `script` the structure-management view cares about, so staleness is flagged when — and only when — one
    of them changes in the main editor. Broader than preview_signature: it also covers rename/merge/copy, which change the body list even
    when they leave the drawn geometry untouched."""
    return tuple(m.group(0) for m in _STALE_RE.finditer(script))


def has_live_update(script: str) -> bool:
    """Whether the script contains an `update` element, i.e. publishes its structure for a live view during a run."""
    return _UPDATE_RE.search(script) is not None


# ── structure-management scripts ──────────────────────────────────────────
def structural_only(script: str) -> str:
    """`script` reduced to the elements that shape the structure: the load block and the body-affecting setup elements, i.e. exactly what
    structure_signature covers. Everything else — `parameters`, `loop`, `optimize_once`, `save`, `print`, `output`, ... — is dropped.

    None of it changes what the preview shows, and leaving it in makes the preview hostage to parts of the script that have nothing to do
    with the structure: a `parameters` block carrying symmetry amplitudes, left behind from a structure that had symmetries, is refused
    outright by the backend for one that has none, which would fail the whole preview rather than just that element.

    Only ever fed to the backend. The script written back to the editor is composed separately, in structure_send_script, from the
    untouched base — that one must keep every element the user wrote.
    """
    return "".join(m.group(0).rstrip() + "\n" for m in _STALE_RE.finditer(script))


def insert_elements(base: str, elements: list[str]) -> str:
    """Return `base` with staged setup elements inserted among the existing setup declarations.

    The staged list itself remains in the order in which the user applied its elements, and existing declarations are left untouched. Two
    anchors are used, since the two ends of the setup block have opposite requirements:

            * structural elements go immediately *before* the first constraint declaration. Constraints are indexed by the structure that exists
                when they are declared, so later structural changes can be rejected by the backend.
            * staged constraint elements go *after* the last existing setup declaration, so they remain after any existing structural changes.

    With no constraints in the base script the two anchors coincide, and the staged block is appended whole.
    """
    if not elements:
        return base
    match = LOAD_BLOCK_RE.search(base)
    if match is None:  # no load block to anchor to: prepend
        return "".join(f"{e}\n" for e in elements) + base

    setup_end = match.end()
    existing_setup = list(_SETUP_RE.finditer(base, match.end()))
    if existing_setup:
        setup_end = existing_setup[-1].end()
    constraint = _CONSTRAINT_RE.search(base, match.end())
    body_set_at = constraint.start() if constraint else setup_end

    def insert(text: str, at: int, staged: list[str]) -> str:
        if not staged:
            return text
        block = "".join(f"{e}\n" for e in staged)
        sep = "" if not text[:at] or text[:at].endswith("\n") else "\n"
        return text[:at] + sep + block + text[at:]

    if body_set_at == setup_end:  # the anchors coincide, so the staged block stays whole and in the order it was applied
        return insert(base, setup_end, elements)

    structural, constraints = [], []
    for e in elements:
        (constraints if e.split()[:1] and e.split()[0] in _CONSTRAINT_ELEMENTS else structural).append(e)
    # the later insertion first, so the earlier offset is still valid when it is applied
    return insert(insert(base, setup_end, constraints), body_set_at, structural)


def structure_preview_script(base: Optional[str], pdb_path: str, splits: str, elements: list[str]) -> str:
    """The setup-only script fed to the backend to build a structure-management view: `base` (or a synthetic load block for `pdb_path` if it
    is None or has no load block of its own) split at `splits`, reduced to its structural elements, with the staged `elements` added. See
    structural_only: this is deliberately not the script the user gets back — structure_send_script composes that one from the full base."""
    if base is None or LOAD_BLOCK_RE.search(base) is None:
        base = synth_load_block(pdb_path, splits)
    else:  # honour the (possibly edited) splits over whatever the base load block carries
        base = with_split(base, splits)
    # stripped before the staged elements go in, not after, so a staged element is never dropped by the filter
    return insert_elements(structural_only(base), elements)


def structure_send_script(base: Optional[str], pdb_path: str, splits: str, elements: list[str]) -> tuple[str, str]:
    """(old, new): the script a structure-management view writes back, and the one it replaces. Unlike structure_preview_script this keeps
    every element of `base` — stripping the non-structural ones would silently delete the user's parameter/loop/save elements. old == new
    when neither the splits nor the staged elements differ from the base."""
    old = base if base is not None else synth_load_block(pdb_path, splits)
    new = with_split(old, splits) if LOAD_BLOCK_RE.search(old) else old
    return old, insert_elements(new, elements)


# ── whole-script rewrites ──────────────────────────────────────────────────
def reset_script(current: str) -> str:
    """The default script, carrying over `current`'s load block (if it has one) so a reset keeps the chosen inputs."""
    current_load = LOAD_BLOCK_RE.search(current)
    if not current_load:
        return DEFAULT_RIGIDBODY_SCRIPT
    return LOAD_BLOCK_RE.sub(lambda _m: current_load.group(0), DEFAULT_RIGIDBODY_SCRIPT, count=1)


def absolutize_script_paths(script: str, base_dir: str) -> str:
    """Rewrite the script's relative file paths (the load block's pdb/saxs and the top-level output directive) into absolute paths anchored
    at `base_dir`, e.g. for a cached script written from a different working directory. Absolute paths are left as-is."""
    def absolutize(path: str) -> str:
        path = unquote_script_value(path)
        if not path or os.path.isabs(path):
            return path
        # preserve a trailing separator: the backend appends file stems to the output dir
        absolute = os.path.normpath(os.path.join(base_dir, path))
        return absolute + os.sep if path.endswith(("/", os.sep)) else absolute

    new_script = OUTPUT_RE.sub(lambda m: m.group(1) + quote_script_value(absolutize(m.group(2))), script, count=1)
    for directive in PATH_DIRECTIVES:
        value = load_value(script, directive)
        if value:
            block = LOAD_BLOCK_RE.search(new_script)
            new_block = rewrite_directive(block.group(0), directive, absolutize(value))
            new_script = new_script[:block.start()] + new_block + new_script[block.end():]
    return new_script

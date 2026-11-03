# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Tokenizing a rigid-body sequencer script for syntax highlighting.

Only the first token of each line carries meaning to the sequencer, so highlighting is a per-line classification of that token plus any
trailing `# comment`. Scope openers (loop, optimize_once, on_improvement) and their closing `end` nest, so each line is classified at a
depth carried over from the line before — which maps directly onto Qt's per-block state in QSyntaxHighlighter, and onto a top-to-bottom
scan for widgets without one.
"""

import re
from typing import NamedTuple, Optional

SCOPE_KEYWORDS = ("loop", "optimize_once", "on_improvement")
_FIRST_TOKEN_RE = re.compile(r"\S+")


class Span(NamedTuple):
    """A highlighted run within one line: columns [start, end) of the given kind.

    kind is one of "comment", "scope", "operation", "keyword" or "error". depth is the scope nesting depth the span is drawn at, used to pick
    a colour for "scope" spans (an opener and its `end` share one), and 0 for every other kind."""
    start: int
    end: int
    kind: str
    depth: int = 0


def fetch_vocabulary() -> tuple[set, set]:
    """Ask the backend for the valid script elements, split into line operations (the dict keys) and argument keywords (values not
    themselves keys). Returns empty sets if the backend is unavailable, in which case a highlighter should still colour scopes/comments
    but flag nothing."""
    try:
        from ...wrapper.Rigidbody import Rigidbody
        mapping = Rigidbody.get_valid_elements_and_arguments()
    except Exception:
        return set(), set()
    operations = set(mapping)
    keywords = {arg for args in mapping.values() for arg in args} - operations
    return operations, keywords


def classify_line(line: str, depth: int, operations, keywords, *, flag_unknown: bool = True) -> tuple[list[Span], int]:
    """Classify one script line entered at scope `depth`, returning its spans and the depth the next line is entered at.

    A comment wins over any keyword inside it. Unrecognised first tokens are flagged as "error" only when `flag_unknown` is set — callers
    without a known vocabulary should clear it, so a missing backend doesn't paint every line red."""
    spans = []
    hash_col = line.find("#")
    if hash_col != -1:
        spans.append(Span(hash_col, len(line), "comment"))
        scannable = line[:hash_col]
    else:
        scannable = line

    match = _FIRST_TOKEN_RE.search(scannable)
    if not match:
        return spans, depth
    token, start, end = match.group(), match.start(), match.end()
    if token == "end":  # decrease first so `end` shares its opener's colour
        depth = max(0, depth - 1)
        spans.append(Span(start, end, "scope", depth))
    elif token in SCOPE_KEYWORDS:
        spans.append(Span(start, end, "scope", depth))
        depth += 1
    elif token in operations:
        spans.append(Span(start, end, "operation"))
    elif token in keywords:
        spans.append(Span(start, end, "keyword"))
    elif token != "}" and flag_unknown:
        spans.append(Span(start, end, "error"))
    return spans, depth


def _first_token(lines: list[str], index: int) -> str:
    tokens = lines[index].split() if 0 <= index < len(lines) else []
    return tokens[0] if tokens else ""


def scope_partner(lines: list[str], index: int) -> Optional[int]:
    """The (0-based) index of the line pairing with line `index` — the `end` closing a scope opener, or the opener an `end` closes — or None
    if that line is neither, or is unmatched."""
    token = _first_token(lines, index)
    if token in SCOPE_KEYWORDS:
        step, opener_first = 1, True
    elif token == "end":
        step, opener_first = -1, False
    else:
        return None
    stack = 0
    n = index + step
    while 0 <= n < len(lines):
        token = _first_token(lines, n)
        if token == "end" or token in SCOPE_KEYWORDS:
            # a token of the same kind as the starting line nests one level deeper; one of the opposite kind closes a level
            nests = (token == "end") != opener_first
            if nests:
                stack += 1
            elif stack == 0:
                return n
            else:
                stack -= 1
        n += step
    return None

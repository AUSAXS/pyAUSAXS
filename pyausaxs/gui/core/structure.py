# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""The model behind the structure-management view: a structure's bodies, the setup elements staged against it, and the selections the
user makes on it.

The body list is authoritative: every edit is applied by rebuilding the setup script through the backend and re-reading the bodies that
remain, so a merge or delete really does make bodies disappear exactly as it will at run time. Edits are staged here as setup elements and
only reach the user's script through structure_send_script, once confirmed.

Front-ends hand the model what the user typed and selected, and show the ActionResult it returns; everything about what an action means —
which selection stands in for an empty field, how a staged split is extended, the wording of the messages — lives here.
"""

import difflib
from typing import Callable, NamedTuple, Optional

from .loads import RELAXED_LOADS, RelaxedLoads
from .plotting import _BODY_COLORS, residue_ca_mask
from .script import (
    LOAD_BLOCK_RE, is_residue_id, load_split, norm_splits, parse_split_element, parse_splits, structure_preview_script,
    split_element_ids, structure_send_script, structure_signature, synth_load_block,
)

Selector = tuple[int, Optional[int]]  # (body index, copy index); copy None selects the whole body, i.e. every one of its copies


class ActionResult(NamedTuple):
    """What an action did: whether it succeeded, a status message to show (empty when there is nothing to say), and whether the structure
    was rebuilt, i.e. whether the view has to refresh from the model."""
    ok: bool
    message: str = ""
    rebuilt: bool = False


class StructureModel:
    """Bodies and staged setup elements of a single structure.

    `base_script` returns the user's current script, which the staged elements are composed onto (or None to work from `pdb_path` alone,
    with a synthetic load block). `base_signature` reduces it to the fingerprint staleness is judged by.
    """

    def __init__(self, pdb_path: str, *, splits: str = "", base_script: Optional[Callable[[], str]] = None,
                 base_signature: Optional[Callable[[str], object]] = None, relaxed_loads: RelaxedLoads = RELAXED_LOADS):
        self.pdb_path = pdb_path
        self.splits = splits                      # the load-block splits in force
        # relaxed-load grants, shared with every other pane by default, so a structure the user already accepted with relaxed settings
        # elsewhere doesn't start failing again here.
        self._relaxed_loads = relaxed_loads
        self._base_script = base_script
        # reduce the base script to a structural fingerprint, so a later edit to the same body/split setup is detected as "stale". Defaults to
        # the structure signature, which covers every element this model reads or writes (rename/merge/split/symmetry/constraint/...).
        self._base_signature = base_signature or structure_signature
        self._built_sig = None                    # base fingerprint the current view was built from

        self.elements: list[str] = []             # setup elements the user has applied, in order
        # Cα positions each staged `split` element's target body covered when it was staged, keyed by that target's name. Fragments are
        # registered as brand-new bodies and inherit nothing from their parent, so the atoms themselves are the only link back to it; this
        # map is what lets a later split of a fragment be folded into the element that produced it (see apply_body_split). Positions, not
        # residue ids: a chain-split structure repeats its ids across chains, so ids would match a sibling chain's split element too.
        self._split_coverage: dict[str, frozenset[tuple[float, ...]]] = {}
        self.data: dict | None = None             # last good preview-structure dict
        self._names: list[str] = []               # body names aligned to body indices
        self.bodies: list[dict] = []              # per-body summary rows (index/name/atoms/res/colour/copies)
        self.replica_info: dict[tuple[int, int], dict] = {}  # (body, copy) -> {"type", "name"}
        # the view selection isolated in the plot, as a set of (body, copy) selectors: copy None selects the whole body (all its symmetry copies),
        # an int selects just that one replica. Empty = nothing selected. Plain click replaces the whole set with just the clicked row. Ctrl-click
        # toggles that row's membership without touching the rest (matching the OS convention for adding single items to a selection). Shift-click
        # selects every row between the last clicked row (the "anchor") and the one just clicked, in list order — the OS convention for
        # range-selecting. Together these let several bodies be selected at once (e.g. to merge them) without typing every name out.
        self.highlighted: set[Selector] = set()
        # residues click-selected in the preview, as (body, residue id) pairs — an id alone does not identify a residue, since a chain-split
        # structure repeats the same numbering in every chain. Like the body selection above (but for residues, not bodies), it stands in for
        # an empty residue field: applying a split with no residues typed splits at the selected residues instead. Consumed (and cleared)
        # when a split is applied.
        self.selected_residues: set[tuple[int, int]] = set()
        self._select_anchor: Optional[Selector] = None  # last plain/ctrl-clicked row, the shift-click range's fixed end
        self.expanded_bodies: set[int] = set()    # body indices whose replica rows are unfolded in the list

    # ----- backend rebuild ----------------------------------------------------
    def _base(self) -> Optional[str]:
        return self._base_script() if self._base_script else None

    def compose(self, elements: list[str]) -> str:
        """The setup-only script fed to the backend to build the view. See script.structural_only: this is deliberately not the script the
        user gets back — send_script composes that one from the full base."""
        return structure_preview_script(self._base(), self.pdb_path, self.splits, elements)

    def rebuild(self, elements: Optional[list[str]] = None) -> tuple[bool, str]:
        """Load the composed setup script (with `elements` staged, the current ones by default) through the backend and refresh the bodies
        from the ones that remain. Returns (ok, message); on failure nothing is changed. The staged elements themselves are left alone:
        adopting `elements` on success is up to the caller."""
        elements = self.elements if elements is None else elements
        script = self.compose(elements)
        try:
            from ...wrapper.Rigidbody import Rigidbody
            with self._relaxed_loads.applied(self.pdb_path):
                rb = Rigidbody(script)
                data = rb.preview_structure()
                if not len(data["coords"]):
                    return False, "the structure is empty"
                names = rb.body_names()
                layout = rb.symmetry_layout()
        except Exception as e:
            return False, str(e)

        self.data, self._names = data, names
        self.replica_info = {
            (int(body), int(copy)): {"type": t, "name": nm}
            for body, copy, t, nm in zip(layout["body"], layout["copy"], layout["type"], layout["name"])
        }
        before = self._body_structure()
        self._compute_bodies()
        # Bodies are addressed by index, and a merge/delete/split/convert renumbers them: the indices a selection holds would silently
        # re-point at unrelated bodies (and residues) once the operation goes through. So both selections survive only a rebuild that
        # leaves the body/copy structure exactly as it was — e.g. a rename, which changes nothing but a name.
        if self._body_structure() != before:
            self.highlighted, self._select_anchor = set(), None
            self.selected_residues = set()
        # the view now matches this base, so it is no longer stale
        self._built_sig = self._base_sig()
        return True, ""

    def _body_structure(self) -> list[tuple[int, tuple[int, ...]]]:
        """The body/copy layout every selector is addressed against, as (body index, copy indices). Compared across a rebuild to tell a
        structural change (merge/delete/split/…) from one that only relabels — see rebuild."""
        return [(b["index"], tuple(b["copies"])) for b in self.bodies]

    def _compute_bodies(self):
        data = self.data
        body, copy, res = data["body"], data["copy"], data["residue_seq"]
        self.bodies = []
        for k, idx in enumerate(sorted(set(body.tolist()))):
            orig = (body == idx) & (copy == 0)
            resids = res[orig & (res >= 0)]
            self.bodies.append({
                "index": idx,
                "name": self._names[k] if k < len(self._names) else f"b{idx + 1}",
                "atoms": int(orig.sum()),
                "res": (int(resids.min()), int(resids.max())) if len(resids) else None,
                "colour": _BODY_COLORS[idx % len(_BODY_COLORS)],
                # every copy index present for this body; copy 0 is the base, >0 are symmetry replicas
                "copies": sorted(set(copy[body == idx].tolist())),
            })

    # ----- staleness / refresh ------------------------------------------------
    def _base_sig(self):
        base = self._base()
        return self._base_signature(base if base is not None else synth_load_block(self.pdb_path, self.splits))

    @property
    def built(self) -> bool:
        """Whether the structure has ever been built successfully."""
        return self._built_sig is not None

    def is_stale(self) -> bool:
        """Whether the base script has changed since the view was built (e.g. the user edited the load block in the main editor)."""
        return self.built and self._base_sig() != self._built_sig

    def refresh_from_base(self) -> tuple[bool, str]:
        """Discard every staged element and rebuild from the current base script, re-reading its splits so they mirror it again."""
        self._set_elements([])
        base = self._base() or ""
        if LOAD_BLOCK_RE.search(base):
            self.splits = load_split(base)
        return self.rebuild()

    # ----- staged elements ----------------------------------------------------
    def _set_elements(self, elements: list[str]):
        self.elements = elements
        # drop residue coverage for bodies no staged `split` targets any more, e.g. after one is removed from the applied list
        targets = {parsed[0] for e in self.elements if (parsed := parse_split_element(e)) is not None}
        self._split_coverage = {name: r for name, r in self._split_coverage.items() if name in targets}

    def _try(self, candidate: list[str], success: str, failure: str) -> ActionResult:
        ok, msg = self.rebuild(candidate)
        if not ok:
            return ActionResult(False, failure.format(msg=msg))
        self._set_elements(candidate)
        return ActionResult(True, success, rebuilt=True)

    def apply_element(self, element: str) -> ActionResult:
        """Stage `element` after the others, if the backend accepts the result."""
        return self._try(self.elements + [element], f"Applied: {element}", f"Rejected “{element}”: {{msg}}")

    def replace_element(self, i: int, element: str) -> ActionResult:
        """Swap the staged element at `i` for `element`, keeping its position in the order. Used to extend an existing `split` rather than
        stage a second one against a fragment the backend would refuse to split."""
        candidate = self.elements[:i] + [element] + self.elements[i + 1:]
        return self._try(candidate, f"Extended: {element}", f"Rejected “{element}”: {{msg}}")

    def remove_element(self, i: int) -> ActionResult:
        candidate = self.elements[:i] + self.elements[i + 1:]
        return self._try(candidate, "Removed an element.", "Could not remove: {msg}")

    # ----- send to script -----------------------------------------------------
    def send_script(self) -> Optional[tuple[str, str]]:
        """(old, new): the user's script and the one carrying the staged changes, or None when there is nothing to send. Deliberately not
        compose(): that one strips every non-structural element, which is right for a preview and would silently delete the user's
        parameter/loop/save elements here."""
        old, new = structure_send_script(self._base(), self.pdb_path, self.splits, self.elements)
        return None if new == old else (old, new)

    def sent(self) -> tuple[bool, str]:
        """Drop the staged edits once the composed script has been written back: they now live in the base, so keeping them staged would
        double-apply them on the next rebuild. Rebuilds from the new base, re-recording its fingerprint."""
        self._set_elements([])
        return self.rebuild()

    # ----- body selection -----------------------------------------------------
    def row_order(self) -> list[Selector]:
        """Selectors in the order their rows appear in the body list: each base body, with any unfolded replicas beneath it. Needed to
        resolve a shift-click range."""
        order = []
        for b in self.bodies:
            order.append((b["index"], None))
            if b["index"] in self.expanded_bodies:
                order += [(b["index"], c) for c in b["copies"][1:]]
        return order

    def _range_between(self, anchor: Selector, selector: Selector) -> set[Selector]:
        """Every selector between `anchor` and `selector` in list order, inclusive. Falls back to just `selector` if either endpoint isn't
        currently shown (e.g. its body was folded away since the anchor was set)."""
        order = self.row_order()
        if anchor not in order or selector not in order:
            return {selector}
        i, j = order.index(anchor), order.index(selector)
        lo, hi = min(i, j), max(i, j)
        return set(order[lo:hi + 1])

    def toggle_highlight(self, body: int, copy: Optional[int], *, shift: bool = False, ctrl: bool = False) -> bool:
        """Plain click: select just this row, or deselect it if it was the only one already selected — and become the anchor for a future
        shift-click range. Ctrl-click: add/remove this row from the current selection without touching the rest (also becomes the new anchor),
        so several bodies can be picked one at a time (e.g. to merge them — see selected_body_names). Shift-click: select every row between
        the anchor and this one, replacing the current selection, without moving the anchor.

        Returns whether a single replica was just selected — isolating one is pointless while copies are hidden, so a view hiding them
        should reveal them."""
        selector = (body, copy)
        if shift and self._select_anchor is not None:
            self.highlighted = self._range_between(self._select_anchor, selector)
        elif ctrl:
            if selector in self.highlighted:
                self.highlighted.discard(selector)
            else:
                self.highlighted.add(selector)
            self._select_anchor = selector
        else:
            self.highlighted = set() if self.highlighted == {selector} else {selector}
            self._select_anchor = selector
        return copy not in (None, 0) and selector in self.highlighted

    def selected_body_names(self) -> list[str]:
        """Names of the whole bodies (copy=None) currently selected, ordered by body index. Replica-only selections don't count —
        merge/delete operate on whole bodies."""
        names_by_index = {b["index"]: b["name"] for b in self.bodies}
        return [names_by_index[i] for i, c in sorted(self.highlighted, key=lambda s: (s[0], -1 if s[1] is None else s[1]))
                if c is None and i in names_by_index]

    def known_names(self) -> set[str]:
        """Every body/replica display name currently known, used to tell whether a field's first token is already a body identifier the
        user typed themselves."""
        names = {b["name"] for b in self.bodies}
        names |= {info["name"] for info in self.replica_info.values()}
        return names

    def with_selected_bodies(self, text: str, *, exact: Optional[int] = None, minimum: int = 1) -> list[str]:
        """Tokens for an action whose field is a leading run of body identifiers followed by other arguments (a symmetry type, a constraint
        type/distance, ...). If the field's first token is already a recognized body/replica name, it's taken literally — the user typed the
        bodies themselves, so the selection is left out of it entirely. Otherwise, if the body selection satisfies the requirement (exactly
        `exact` whole bodies if given, else at least `minimum`), the selected bodies' names are prepended to whatever's already typed — so
        e.g. selecting two bodies and typing just "cm" is enough for a constraint. Falls back to the typed tokens as-is when neither applies,
        so the existing per-action validation reports the real problem (e.g. "needs a body and a type") rather than silently doing nothing."""
        tokens = text.split()
        if tokens and tokens[0] in self.known_names():
            return tokens
        selected = self.selected_body_names()
        satisfied = len(selected) == exact if exact is not None else len(selected) >= minimum
        if not satisfied:
            return tokens
        return selected + tokens

    def name_of(self, body: int, copy: Optional[int]) -> Optional[str]:
        """Current display name of a body or replica, read fresh from the model rather than a value captured when a row was built — matters
        after an in-place rename, which a view may apply without rebuilding its rows."""
        if copy is None:
            return next((b["name"] for b in self.bodies if b["index"] == body), None)
        info = self.replica_info.get((body, copy))
        return info["name"] if info else None

    def selected_single_name(self) -> Optional[str]:
        """Display name of the one currently selected row — a whole body or a single replica — or None unless exactly one is selected. Used
        to let Rename infer "old" from a click instead of typing it, which (unlike merge/delete) makes sense for a replica selection too."""
        if len(self.highlighted) != 1:
            return None
        body, copy = next(iter(self.highlighted))
        return self.name_of(body, copy)

    def index_of(self, name: str) -> Optional[int]:
        """Index of the body with this display name, or None if no body has it (a replica name included: replicas aren't bodies)."""
        return next((b["index"] for b in self.bodies if b["name"] == name), None)

    # ----- residue selection --------------------------------------------------
    def toggle_selected_residue(self, body: int, resid: int) -> str:
        """Add the residue to the click-selection, or remove it if already selected, returning a status message describing what it now
        offers. It is keyed by the body it was picked in as well as its id, so clicking residue 100 of one chain marks that one residue rather
        than every chain's residue 100. Nothing is applied — the selection just stands in for an empty residue field when a split is applied
        (see apply_splits / apply_body_split), mirroring how a body selection stands in for an empty Merge/Delete field.

        Once the load-block split would override rather than establish the partition (see split_would_override), the per-body split is the
        only thing a click can usefully feed, so a view keeping it folded away should reveal it."""
        pick = (body, resid)
        self.selected_residues.discard(pick) if pick in self.selected_residues else self.selected_residues.add(pick)
        n = len(self.selected_residues)
        if not n:
            return "Selection cleared."
        # Add targets one body, so it is only offered while the selection resolves to one
        target = self.split_target_name()
        hint = f", or Add to split {target} there" if target else ""
        return f"{n} residue{'' if n == 1 else 's'} selected — Apply to split the whole structure there{hint}."

    def _selected_bodies(self) -> set[int]:
        """Body indices the click-selected residues lie in."""
        return {body for body, _resid in self.selected_residues}

    def _selection_body_name(self) -> Optional[str]:
        """Name of the body every click-selected residue lies in, or None when there is no selection or it spans several bodies."""
        bodies = self._selected_bodies()
        return self.name_of(next(iter(bodies)), None) if len(bodies) == 1 else None

    def split_target_name(self) -> Optional[str]:
        """Body a per-body split would cut the click-selection in: the original target of a staged split that already covers the selection
        (whose fragments count as one body's business), otherwise the single live body the residues lie in."""
        staged = self._staged_split_for(self.selected_residues)
        return staged[1] if staged is not None else self._selection_body_name()

    def clear_all_selection(self):
        """Drop both selections in one go: the click-selected residues and the body highlight."""
        self.selected_residues = set()
        self.highlighted, self._select_anchor = set(), None

    def split_would_override(self) -> bool:
        """Whether applying the load-block splits would override work already done, rather than being the plain first split it looks like: a
        split is already in force (including `chain`, which typing residues silently replaces), or staged elements name bodies that a re-split
        renumbers. In both cases the per-body `split` element is what the user actually wants."""
        return bool(self.elements) or bool(self.splits.strip())

    def split_residues(self) -> list[int]:
        """Residue ids to mark as split points in the view: the load-block split plus every `split` element — both the staged ones and any
        already in the base script. Non-numeric tokens (a body name, the keyword itself) are ignored; draw_structure deduplicates."""
        ids = parse_splits(self.splits)
        sources = list(self.elements)
        base = self._base()
        if base is not None:
            sources.append(base)
        for text in sources:
            ids += split_element_ids(text)
        return ids

    # ----- whether an action has enough to go on ------------------------------
    # Each takes the action's typed field, and is True when applying it would have something to act on — typed text, or a selection that
    # can stand in for it — so a view can tell the user at a glance that clicking will actually do something.
    def can_split_body(self, text: str) -> bool:
        return bool(self.selected_residues) or len(text.split()) >= 2

    def can_rename(self, text: str) -> bool:
        return len(text.split()) == 2 or (len(text.split()) == 1 and self.selected_single_name() is not None)

    def can_merge(self, text: str) -> bool:
        return bool(text) or len(self.selected_body_names()) >= 2

    def can_delete(self, text: str) -> bool:
        return bool(text) or len(self.selected_body_names()) >= 1

    # ----- setup actions ------------------------------------------------------
    def apply_splits(self, typed: str) -> tuple[ActionResult, str]:
        """Re-split the structure at the residue numbers `typed` into the load-block splits field, returning the result and the value the
        field should show afterwards. The split lives in the load block, so it is applied by recomposing (see script.with_split); on a bad
        value the field is reverted, so it always mirrors the splits actually in force.

        With the field left empty, the click-selection stands in for it — click residues in the preview and hit Apply, no typing — matching
        how a body selection stands in for an empty Merge/Delete field."""
        typed = typed.strip()
        from_selection = not typed and bool(self.selected_residues)
        # the load-block split cuts the file as a whole, so only the ids matter here; two chains' residue 100 collapse to one cut
        new = " ".join(str(r) for r in sorted({r for _b, r in self.selected_residues})) if from_selection else typed
        if norm_splits(new) == norm_splits(self.splits):
            if from_selection:
                self.selected_residues = set()  # already the splits in force, but consume the selection so it doesn't linger
            return ActionResult(True), new
        prev = self.splits
        self.splits = new
        ok, msg = self.rebuild()
        if not ok:
            self.splits = prev
            return ActionResult(False, f"Could not re-split: {msg}"), prev
        if from_selection:
            self.selected_residues = set()
        return ActionResult(True, f"Re-split at {new}." if new else "Removed all splits.", rebuilt=True), new

    def apply_body_split(self, text: str) -> ActionResult:
        """Split an existing body into fragments at the given residue ids: `split <body> <residues…>`. Distinct from the load-block split
        above (which re-partitions the freshly-read PDB and defines the whole body set): this is a staged setup element that partitions a
        body already in the setup — e.g. one produced by convert_to_symmetry — so its fragments stay tied together in a shared symmetry.
        The body may come from a single body selection or, failing that, from the click-selected residues themselves — each one belongs to a
        body, so clicking a few residues along one body and applying, with nothing typed and nothing else selected, is enough.

        Residues already inside a staged split's target extend that element instead of staging a second one. The backend refuses to split
        a fragment that carries a symmetry shared with its siblings, and one element is enough regardless: BodySplitter cuts at a *set* of
        residue ids taken in atom order, so cutting the original body at the union of both sets yields exactly the same fragments. This
        also lets a selection spanning two fragments of one original body work, since they are one split element's business."""
        tokens = self.with_selected_bodies(text, exact=1)
        if tokens and tokens[0] in self.known_names():
            named, typed_residues = tokens[0], tokens[1:]
        else:  # nothing names a body; whatever was typed is then all residues, and the body follows from them
            named, typed_residues = None, tokens
        from_selection = not typed_residues and bool(self.selected_residues)
        if from_selection:
            picks = set(self.selected_residues)
        else:
            if not typed_residues:
                return ActionResult(False, "Splitting a body needs residue ids — type them or click residues in the preview.")
            if not all(is_residue_id(r) for r in typed_residues):
                return ActionResult(False, "Split residue ids must be integers, e.g. b1 100 200.")
            # typed ids are only residues of the body they are typed against, so pair them with it; with no body to pair them with there is
            # nothing to match or split, and the "needs a body" branch below reports it
            index = self.index_of(named or self._selection_body_name() or "")
            picks = {(index, int(r)) for r in typed_residues} if index is not None else set()
        cuts = {r for _b, r in picks}

        staged = self._staged_split_for(picks)
        if staged is not None:
            i, target, existing = staged
            merged = sorted(existing | cuts)
            if len(merged) == len(existing):
                return ActionResult(False, f"Already splitting {target} at {' '.join(str(c) for c in sorted(cuts))}.")
            result = self.replace_element(i, "split " + target + " " + " ".join(str(c) for c in merged))
        else:
            body = named or self._selection_body_name()
            if body is None:
                return ActionResult(False,
                    "The selected residues span several bodies — split one body at a time." if len(self._selected_bodies()) > 1 else
                    "Splitting a body needs a body — type one, select it in the Bodies list, or click residues on it.")
            coverage = self._coverage_of_body(body)  # read before the rebuild replaces the body with its fragments
            result = self.apply_element("split " + body + " " + " ".join(str(c) for c in sorted(cuts)))
            if result.ok:
                self._split_coverage[body] = coverage
        if result.ok and from_selection:
            self.selected_residues = set()
        return result

    def _staged_split_for(self, picks: set[tuple[int, int]]) -> tuple[int, str, frozenset[int]] | None:
        """(index, target body, existing cuts) of the staged `split` element whose target body contained every one of the (body, residue)
        `picks`, or None when they fall outside every staged split. Targets are whole bodies, so at most one element can ever match."""
        positions = self._ca_positions(residue_ca_mask(self.data, picks))
        if not positions:
            return None
        for i, element in enumerate(self.elements):
            parsed = parse_split_element(element)
            if parsed is not None and positions <= self._split_coverage.get(parsed[0], frozenset()):
                return i, parsed[0], frozenset(parsed[1])
        return None

    def _ca_positions(self, mask) -> frozenset[tuple[float, ...]]:
        """The masked atoms' positions, rounded to a stable key. Positions identify residues where ids and body indices cannot: ids repeat
        across chains, and indices are renumbered by every merge/delete/split — the atoms themselves are only ever re-partitioned."""
        if self.data is None:
            return frozenset()
        return frozenset(tuple(round(float(v), 3) for v in p) for p in self.data["coords"][mask])

    def _coverage_of_body(self, name: str) -> frozenset[tuple[float, ...]]:
        """Every Cα position in the named body, as it stands right now — the coverage a `split` element staged against it takes over."""
        index = self.index_of(name)
        if index is None or self.data is None:
            return frozenset()
        d = self.data
        return self._ca_positions(d["is_ca"] & (d["copy"] == 0) & (d["body"] == index))

    def rename(self, old: str, new: str) -> ActionResult:
        """Rename a body or replica in place: `rename <old> <new>`. The backend keeps the default name too, so a rename can always be undone
        by renaming back. A no-op (empty or unchanged name) succeeds without doing anything."""
        new = new.strip()
        if not new or new == old:
            return ActionResult(True)
        if any(c.isspace() for c in new):
            return ActionResult(False, "A name cannot contain spaces.")
        return self.apply_element(f"rename {old} {new}")

    def apply_rename(self, text: str) -> ActionResult:
        """Rename a body: `rename <old> <new>`. Either typed as two tokens (old new), or — with exactly one row selected — as just the new
        name, taking "old" from that selection."""
        tokens = text.split()
        if len(tokens) == 2:
            old, new = tokens
        elif len(tokens) == 1 and self.selected_single_name() is not None:
            old, new = self.selected_single_name(), tokens[0]
        else:
            return ActionResult(False, "Rename needs the current name and the new name, e.g. b1 core.")
        if old == new:
            return ActionResult(False, "The new name is the same as the current one.")
        return self.apply_element(f"rename {old} {new}")

    def apply_merge(self, text: str) -> ActionResult:
        # an empty field falls back to the body selection, so a shift-clicked group of bodies can be merged without typing every name out
        # (see selected_body_names)
        tokens = text.split() or self.selected_body_names()
        if len(tokens) < 2:
            return ActionResult(False, "Merge needs a target body and at least one other.")
        return self.apply_element("merge " + " ".join(tokens))

    def apply_delete(self, text: str) -> ActionResult:
        tokens = text.split() or self.selected_body_names()
        if not tokens:
            return ActionResult(False, "Delete needs at least one body.")
        return self.apply_element("delete " + " ".join(tokens))

    def apply_add_symmetry(self, text: str) -> ActionResult:
        """Add a symmetry to a single body: `symmetry <body> <type>` (e.g. b1 c4). A lone type is allowed for a single-body system (the backend
        infers the body), or — with exactly one whole body selected — for any system, taking the body from that selection."""
        tokens = self.with_selected_bodies(text, exact=1)
        if not tokens:
            return ActionResult(False, "Add symmetry needs a body and a type, e.g. b1 c4.")
        if len(tokens) > 2:
            return ActionResult(False, "Add symmetry takes one body and one type, e.g. b1 c4.")
        return self.apply_element("symmetry " + " ".join(tokens))

    def apply_convert_symmetry(self, text: str, tolerance: str = "") -> ActionResult:
        """Decompose one or more bodies into one shared symmetry, collapsing the copies into the first body plus a fitted symmetry:
        `convert_to_symmetry { type <type> bodies <b…> }`. A single body is fitted directly — the backend splits it into the symmetry's own
        copy count itself, rather than needing several bodies handed to it as ready-made copies. Typing just the type is enough when one or
        more whole bodies are selected or — with only one body in the whole system — with nothing selected at all, since decomposing it is
        then the only sensible target. An optional `tolerance` (Å) overrides the backend's default when the assembly's residual RMSD to the
        fitted symmetry is just barely out of range. The bodies are reordered so the largest leads — see _most_complete_first."""
        tokens = self.with_selected_bodies(text, minimum=1)
        if len(tokens) == 1 and len(self.bodies) == 1:
            tokens = [self.bodies[0]["name"]] + tokens
        if len(tokens) < 2:
            return ActionResult(False, "Decomposing needs at least one body and a type, e.g. b1 c2.")
        *bodies, sym = tokens
        bodies = self._most_complete_first(bodies)
        tolerance = tolerance.strip()
        extra = ""
        if tolerance:
            try:
                float(tolerance)
            except ValueError:
                return ActionResult(False, "Tolerance must be a number, e.g. 3.5.")
            extra = "\n    tolerance " + tolerance
        return self.apply_element("convert_to_symmetry {\n    type " + sym + "\n    bodies " + " ".join(bodies) + extra + "\n}")

    def _most_complete_first(self, bodies: list[str]) -> list[str]:
        """Reorder a decomposition's bodies so the one with the most atoms leads, leaving the rest in the given order. The backend keeps
        the first-listed body and regenerates the others from it, so that body is the one that survives into the model — and copies of the
        same molecule are routinely modelled to differing extents (a terminus resolved in some chains but not others), making the fullest
        of them the one worth keeping. Left alone if any name is not a body we know the size of, since we then cannot judge; ties keep the
        original leader."""
        atoms = {b["name"]: b["atoms"] for b in self.bodies}
        if not all(name in atoms for name in bodies):
            return bodies
        lead = max(range(len(bodies)), key=lambda i: atoms[bodies[i]])
        return [bodies[lead]] + bodies[:lead] + bodies[lead + 1:]

    def apply_autoconstrain(self, text: str) -> ActionResult:
        """Auto-generate a set of constraints: `autoconstrain <backbone|none>`. Defaults to backbone, the usual choice; `none` clears any
        auto-generated set."""
        return self.apply_element(f"autoconstrain {text.strip() or 'backbone'}")

    def apply_add_constraint(self, text: str) -> ActionResult:
        """Add a distance constraint between two bodies: `<body1> <body2> <type> [distance]`. `bond` and `cm` need only the two bodies;
        `attract` and `repel` also take a target distance (e.g. b1 b2 attract 30). Built as a `constrain { … }` block for the backend.
        Typing just the type (and distance, if needed) is enough when exactly two whole bodies are selected."""
        tokens = self.with_selected_bodies(text, exact=2)
        if len(tokens) < 3:
            return ActionResult(False, "A constraint needs two bodies and a type, e.g. b1 b2 cm.")
        body1, body2, ctype, *rest = tokens
        lines = [f"    first {body1}", f"    second {body2}", f"    type {ctype}"]
        if ctype in ("attract", "repel"):
            if len(rest) != 1:
                return ActionResult(False, f"A {ctype} constraint needs a distance, e.g. b1 b2 {ctype} 30.")
            lines.append(f"    distance {rest[0]}")
        elif rest:
            return ActionResult(False, f"A {ctype} constraint takes no arguments beyond the two bodies.")
        return self.apply_element("constrain {\n" + "\n".join(lines) + "\n}")


class DiffRow(NamedTuple):
    """One row of a side-by-side diff. A side's text is "" where it is padded to keep the two sides aligned."""
    old: str
    old_changed: bool
    new: str
    new_changed: bool


def side_by_side_diff(old: str, new: str) -> list[DiffRow]:
    """Line-aligned rows for showing `old` and `new` next to each other, with removed/added lines marked. Where a change replaces a
    different number of lines, the shorter side is padded with blank rows."""
    old_lines, new_lines = old.splitlines(), new.splitlines()
    rows = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
        a, b = old_lines[i1:i2], new_lines[j1:j2]
        if tag == "equal":
            rows += [DiffRow(x, False, y, False) for x, y in zip(a, b)]
            continue
        for k in range(max(len(a), len(b))):
            rows.append(DiffRow(a[k] if k < len(a) else "", k < len(a), b[k] if k < len(b) else "", k < len(b)))
    return rows

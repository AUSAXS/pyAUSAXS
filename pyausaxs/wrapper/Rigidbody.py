from .AUSAXS import call
from .BackendObject import BackendObject
from pyausaxs.signatures import register, out, out_array, STATUS, HOLDER
import ctypes as ct
import numpy as np
import threading

def _ints(*names: str, n: str) -> list:
    return [out_array(name, ct.c_int, n=n) for name in names]

def _coords(n: str) -> list:
    return [out_array(name, ct.c_double, n=n) for name in ("x", "y", "z")]

register({
    "rigidbody_load_script": ([ct.c_char_p, STATUS], ct.c_int),
    "rigidbody_validate":    ([ct.c_int, STATUS], None),
    "rigidbody_get_preview_structure": (
        [
            ct.c_int, *_coords("n"), *_ints("body", "copy", "residue_seq", "is_ca", n="n"), out("n", ct.c_int),
            out_array("constraints", ct.c_int, n="n_constraints", cols=3), out("n_constraints", ct.c_int), STATUS
        ],
        HOLDER
    ),
    "rigidbody_create_live_poller":     ([STATUS], ct.c_int),
    "rigidbody_poll_live_structure":    ([ct.c_int, *_coords("n"), out("n", ct.c_int), out("version", ct.c_int), STATUS], None),
    "rigidbody_register_live_consumer": ([ct.c_bool, STATUS], None),
    "rigidbody_run": (
        [ct.c_int, *(out_array(name, ct.c_double, n="n") for name in ("q", "I", "Ierr", "Iinterp")), out("n", ct.c_int), STATUS],
        HOLDER
    ),
    "rigidbody_stop_run":            ([STATUS], None),
    "rigidbody_get_valid_elements":  ([out_array("elements", ct.c_char_p, n="n"), out("n", ct.c_int), STATUS], None),
    "rigidbody_get_valid_arguments": ([ct.c_char_p, out_array("arguments", ct.c_char_p, n="n"), out("n", ct.c_int), STATUS], None),
    "rigidbody_get_body_names":      ([ct.c_int, out_array("names", ct.c_char_p, n="n"), out("n", ct.c_int), STATUS], None),
    "rigidbody_get_symmetry_layout": (
        [
            ct.c_int, *_ints("body", "copy", "symmetry", "replica", n="n"),
            out_array("type", ct.c_char_p, n="n"), out_array("name", ct.c_char_p, n="n"), out("n", ct.c_int), STATUS
        ],
        HOLDER
    ),
})

class _LivePoller(BackendObject):
    """Owns the backend buffers that each live-structure poll copies into, so successive polls reuse them."""
    def __init__(self):
        super().__init__()
        self._set_id(call("rigidbody_create_live_poller"))

_live_poller: _LivePoller | None = None
_live_poller_lock = threading.Lock() # every poll rewrites the poller's buffers

class Rigidbody(BackendObject):
    def __init__(self, script: str):
        super().__init__()
        self._read_script(script)

    def _read_script(self, script: str) -> None:
        self._set_id(call("rigidbody_load_script", script))

    def preview_structure(self) -> dict:
        """Get the explicit structure (symmetries realized) annotated with the per-atom metadata needed to draw a preview. 
        Returns a dict with:
            coords      : (N, 3) float array of atom positions
            body        : (N,) int   — index of the body each atom belongs to
            copy        : (N,) int   — symmetry copy index (0 = original)
            residue_seq : (N,) int   — residue number (-1 if unknown)
            is_ca       : (N,) bool  — whether the atom is a Cα
            constraints : (M, 3) int — one row per constraint, [atom_index_1, atom_index_2, type]
                          with type 0=backbone, 1=centre-of-mass, 2=attractor, 3=repulsor.
                          The atom indices point into the rows of `coords` (copy 0 of each body)."""
        p = call("rigidbody_get_preview_structure", self._get_id())
        return {
            "coords": np.column_stack((p.x, p.y, p.z)),
            "body": p.body.astype(int),
            "copy": p.copy.astype(int),
            "residue_seq": p.residue_seq.astype(int),
            "is_ca": p.is_ca.astype(bool),
            "constraints": p.constraints.astype(int),
        }

    def body_names(self) -> list[str]:
        """Names of the bodies currently in the setup, after any merge/delete/convert_to_symmetry
        elements have collapsed or removed bodies. The order matches the body indices reported by
        preview_structure(), so body index i corresponds to body_names()[i]."""
        return call("rigidbody_get_body_names", self._get_id()).tolist()

    def symmetry_layout(self) -> dict:
        """Get the symmetry-replica layout of the current structure, one row per replica (copy > 0),
        keyed to preview_structure()'s (body, copy) pairs so the GUI never has to guess the
        copy -> symmetry mapping. Returns a dict with:
            body     : (R,) int       — body index (matches preview_structure's body)
            copy     : (R,) int       — symmetry copy index (matches preview_structure's copy)
            symmetry : (R,) int       — 0-based symmetry index within the body
            replica  : (R,) int       — replica index within the symmetry
            type     : (R,) list[str] — symmetry type (e.g. "c4", "p2")
            name     : (R,) list[str] — current addressable name, including a rename alias when present (e.g. "b1s1r1" or "my_replica")"""
        s = call("rigidbody_get_symmetry_layout", self._get_id())
        return {
            "body": s.body.astype(int),
            "copy": s.copy.astype(int),
            "symmetry": s.symmetry.astype(int),
            "replica": s.replica.astype(int),
            "type": s.type.tolist(),
            "name": s.name.tolist(),
        }

    @staticmethod
    def set_live_consumer(connected: bool) -> None:
        """Register (or unregister) as a consumer of the live structure. While no consumer is registered, an `update` element does 
        nothing and warns once at parse time, so it wastes no resources. A GUI that polls live_structure() should register on startup."""
        call("rigidbody_register_live_consumer", connected)

    @staticmethod
    def live_structure() -> tuple:
        """Poll the structure most recently published by an `update structure` element during a run. Returns (coords, version), where
        coords is an (N, 3) float array (or None if nothing has been published yet) and version is an int that increments on each 
        publish — the GUI compares it against the previous value to detect new frames. The atom order matches preview_structure(), 
        so a backbone mask computed there can be reused here. Thread-safe."""
        global _live_poller
        with _live_poller_lock:
            if _live_poller is None:
                _live_poller = _LivePoller()
            p = call("rigidbody_poll_live_structure", _live_poller._get_id())
        return (np.column_stack((p.x, p.y, p.z)) if len(p.x) else None), p.version

    def validate(self) -> None:
        """Validate the script and raise an error if it is invalid."""
        call("rigidbody_validate", self._get_id())

    def run(self) -> np.ndarray:
        """Run the rigid-body refinement script."""
        return np.column_stack(call("rigidbody_run", self._get_id()))

    @staticmethod
    def stop_run() -> None:
        """Ask a running refinement to stop. The request is picked up at the start of the next loop iteration, after which run()
        returns normally with the best conformation found so far. Safe to call from a different thread than the one in run(), and
        a no-op if nothing is running."""
        call("rigidbody_stop_run")

    @staticmethod
    def get_valid_elements() -> list[str]:
        """Get a list of valid elements for rigid-body refinement."""
        return call("rigidbody_get_valid_elements").tolist()

    @staticmethod
    def get_valid_arguments(element: str) -> list[str]:
        """Get a list of valid arguments for a given element."""
        return call("rigidbody_get_valid_arguments", element).tolist()

    @staticmethod
    def get_valid_elements_and_arguments() -> dict[str, list[str]]:
        """Get a dictionary of valid elements and their corresponding arguments."""
        if hasattr(Rigidbody.get_valid_elements_and_arguments, "map"):
            return Rigidbody.get_valid_elements_and_arguments.map

        elements = Rigidbody.get_valid_elements()
        element_arguments = {}
        for element in elements:
            arguments = Rigidbody.get_valid_arguments(element)
            element_arguments[element] = arguments
        setattr(Rigidbody.get_valid_elements_and_arguments, "map", element_arguments)
        return element_arguments

def prepare_rigidbody_refinement(script: str) -> Rigidbody:
    """
    Prepare a rigid-body refinement by loading the refinement script.
    """
    rb = Rigidbody(script)
    return rb
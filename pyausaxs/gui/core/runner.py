# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Running a rigid-body refinement script on a worker thread, independently of any GUI toolkit.

The backend call releases the GIL, so the GUI stays responsive while it runs. Its console output arrives through the library's output
callback on the worker thread and is queued; the front-end drains the queue from its own timer on the GUI thread (tkinter's `after`,
a Qt QTimer, ...), which is where every callback passed to drain() is invoked.
"""

import queue
import threading
from typing import Callable, Optional

from .loads import RELAXED_LOADS, RelaxedLoads


class RefinementOutcome:
    """The outcome of a rigid-body validation or refinement."""
    def __init__(self, result, error, error_streamed=False):
        self.result = result    # np.ndarray of [q, I, Ierr, I_model], or None on failure/validation
        self.error = error      # Exception, or None on success
        # True if the error was already streamed to the console by the backend (run-time
        # errors), so the GUI can avoid printing a duplicate
        self.error_streamed = error_streamed


class RefinementWorker:
    """Runs a rigid-body refinement script through the Rigidbody API on a worker thread.

    start() launches a run; the front-end then calls drain() periodically on its GUI thread until it returns False. drain() hands each queued
    output line to on_line(str), and the RefinementOutcome to on_done once the run has finished. For a validation, the outcome's result is
    None and its error indicates success/failure.
    """

    POLL_MS = 100  # a suitable interval for the front-end's drain() timer

    def __init__(self):
        self._busy = False
        self._stopping = False
        self._queue: queue.Queue = queue.Queue()
        self._relaxed_loads: Optional[RelaxedLoads] = None
        self._overrides = None  # settings the active run relaxed, held until it finishes

    def running(self) -> bool:
        return self._busy

    def stopping(self) -> bool:
        return self._stopping

    def request_stop(self) -> bool:
        """Ask a running refinement to stop after its current iteration. The run then finishes normally with the best
        conformation found so far, so on_done still receives a result. Returns False if there is nothing to stop."""
        if not self._busy or self._stopping:
            return False
        from ...wrapper.Rigidbody import Rigidbody
        Rigidbody.stop_run()
        self._stopping = True
        return True

    def start(self, script: str, validate_only: bool, *, structure_path: Optional[str] = None,
              relaxed_loads: RelaxedLoads = RELAXED_LOADS):
        """Launch a validation (validate_only) or a refinement of `script`. Any relaxed load granted for `structure_path` stays applied until
        the run finishes: the run reads the structure on the worker thread. That is safe despite the settings being process-wide, as long as
        the front-end doesn't load structures of its own meanwhile (e.g. it should skip its structure preview while running())."""
        if self._busy:
            raise RuntimeError("a refinement is already running")
        self._busy = True
        self._stopping = False
        self._queue = queue.Queue()

        try:
            from ...wrapper.AUSAXS import AUSAXS
            from ...wrapper.Rigidbody import prepare_rigidbody_refinement
            from ...wrapper.Output import set_output_callback, reset_output_callback
            # instantiate the singleton up front: the output-callback and rigidbody
            # wrappers reach the library via AUSAXS.lib() classmethods, which require
            # the instance to already exist.
            if not AUSAXS().ready():
                raise RuntimeError(f"library failed to initialize: {AUSAXS().init_error()}")
        except Exception as e:
            # reported through the queue like any other outcome, so the front-end handles it in one place
            self._queue.put(f"AUSAXS library unavailable: {e}\n")
            self._queue.put(RefinementOutcome(None, e))
            return

        self._relaxed_loads = relaxed_loads
        self._overrides = relaxed_loads.apply(structure_path)
        if not validate_only:
            # automatically disable weighted bins if they are not required
            # TODO move this to the backend
            from ...wrapper.settings import settings
            settings.set("weighted_bins", "auto")

        threading.Thread(
            target=self._work,
            args=(script, validate_only, prepare_rigidbody_refinement, set_output_callback, reset_output_callback),
            daemon=True,
        ).start()

    def _work(self, script, validate_only, prepare, set_cb, reset_cb):
        result, error, error_streamed = None, None, False
        # the callback fires on this worker thread; just enqueue for the GUI thread
        set_cb(lambda line: self._queue.put(line))
        try:
            rb = prepare(script)
            if validate_only:
                rb.validate()
            else:
                try:
                    result = rb.run()
                except Exception:
                    # the backend prints run-time errors to its output (already streamed
                    # here via the callback); flag so the GUI doesn't print a duplicate
                    error_streamed = True
                    raise
        except Exception as e:
            error = e
        finally:
            reset_cb()
        self._queue.put(RefinementOutcome(result, error, error_streamed))

    def drain(self, on_line: Callable[[str], None], on_done: Callable[[RefinementOutcome], None]) -> bool:
        """Deliver everything queued since the last call. Returns whether the run is still going, i.e. whether to call again."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return self._busy
            if isinstance(item, RefinementOutcome):
                self._busy = False
                self._stopping = False
                if self._relaxed_loads is not None:  # the run is over; put the global settings back
                    self._relaxed_loads.restore(self._overrides)
                self._relaxed_loads = self._overrides = None
                on_done(item)
                return False
            on_line(item)

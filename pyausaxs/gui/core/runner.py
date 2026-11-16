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

    def start(self, script: str, validate_only: bool, *, backend_settings: Optional[dict[str, str]] = None):
        """Launch a validation (validate_only) or a refinement of `script`. The backend is reset to its defaults first, then given
        `backend_settings` (e.g. the q-range and any relaxed load settings). Those stay in force while the run reads them on the worker
        thread, as long as the front-end doesn't enter the backend meanwhile (e.g. it should skip its structure preview while running())."""
        if self._busy:
            raise RuntimeError("a refinement is already running")
        from ...wrapper.Rigidbody import prepare_rigidbody_refinement
        from ...wrapper.Output import set_output_callback, reset_output_callback
        from ...wrapper.settings import settings
        settings.reset_to_defaults()
        for name, value in (backend_settings or {}).items():
            settings.set(name, value)
        self._busy = True
        self._stopping = False
        self._queue = queue.Queue()
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
                on_done(item)
                return False
            on_line(item)

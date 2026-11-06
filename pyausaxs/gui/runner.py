# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Run AUSAXS CLI tools and rigid-body refinements in-process on a worker thread, streaming their output into the GUI."""

import os
import queue
import sys
import threading
from typing import Callable, Optional

from .core.loads import RELAXED_LOADS, RelaxedLoads
from .core.runner import RefinementOutcome, RefinementWorker


class CliRunner:
    """Runs an `ausaxs <tool> ...` command through the backend entrypoint on a worker thread.

    The ctypes call releases the GIL, so the GUI stays responsive. stdout/stderr are
    redirected at the file descriptor level while the tool runs, so the backend's
    console output can be streamed into the GUI. on_line(str) is called for every
    output line and on_done(returncode) once on completion, both on the GUI thread.
    """

    POLL_MS = 100

    def __init__(self, tk_widget):
        self._widget = tk_widget
        self._busy = False
        self._queue: queue.Queue = queue.Queue()
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_done: Optional[Callable[[int], None]] = None

    def running(self) -> bool:
        return self._busy

    def start(self, args: list[str],
              on_line: Callable[[str], None], on_done: Callable[[int], None]):
        """Run `ausaxs *args`, e.g. args = ["fit", "structure.pdb", "data.dat"]."""
        if self._busy:
            raise RuntimeError("a fit is already running")

        from ..wrapper.AUSAXS import AUSAXS
        if not AUSAXS.ready():
            on_line(f"AUSAXS library unavailable: {AUSAXS.init_error()}\n")
            on_done(1)
            return

        self._busy = True
        self._stopping = False
        self._on_line = on_line
        self._on_done = on_done
        self._queue = queue.Queue()

        # redirect stdout/stderr into a pipe for the duration of the call
        read_fd, write_fd = os.pipe()
        saved_fds = (os.dup(1), os.dup(2))
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)

        reader = threading.Thread(target=self._read_pipe, args=(read_fd,), daemon=True)
        reader.start()
        threading.Thread(
            target=self._work, args=(args, saved_fds, reader), daemon=True
        ).start()
        self._widget.after(self.POLL_MS, self._poll)

    def _read_pipe(self, read_fd: int):
        with os.fdopen(read_fd, errors="replace") as pipe:
            for line in pipe:
                self._queue.put(line)

    def _work(self, args: list[str], saved_fds: tuple[int, int], reader: threading.Thread):
        from ..__main__ import run_backend_cli
        returncode = 1
        try:
            returncode = run_backend_cli(args)
        except Exception as e:
            self._queue.put(f"\nError: {e}\n")
        finally:
            # restoring the fds closes the last write end of the pipe, ending the reader
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            os.close(saved_fds[0])
            os.close(saved_fds[1])
        reader.join()
        self._queue.put(returncode)

    def _poll(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, int):
                self._busy = False
                if self._on_done:
                    self._on_done(item)
                return
            if self._on_line:
                self._on_line(item)
        self._widget.after(self.POLL_MS, self._poll)


class RigidbodyRunner:
    """Drives a core RefinementWorker from the tkinter event loop: the same worker, drained on a Tk `after` timer.

    on_line(str) receives each output line and on_done(RefinementOutcome) is called once on completion, both on the GUI thread."""

    def __init__(self, tk_widget):
        self._widget = tk_widget
        self._worker = RefinementWorker()

    def running(self) -> bool:
        return self._worker.running()

    def stopping(self) -> bool:
        return self._worker.stopping()

    def request_stop(self) -> bool:
        """See RefinementWorker.request_stop."""
        return self._worker.request_stop()

    def start(self, script: str, validate_only: bool,
              on_line: Callable[[str], None], on_done: Callable[[RefinementOutcome], None], *,
              structure_path: Optional[str] = None, relaxed_loads: RelaxedLoads = RELAXED_LOADS):
        """See RefinementWorker.start."""
        self._worker.start(script, validate_only, structure_path=structure_path, relaxed_loads=relaxed_loads)

        def poll():
            if self._worker.drain(on_line, on_done):
                self._widget.after(RefinementWorker.POLL_MS, poll)
        self._widget.after(RefinementWorker.POLL_MS, poll)

# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Run AUSAXS CLI tools in-process on a worker thread, streaming their output into the GUI."""

import os
import queue
import sys
import threading
from typing import Callable, Optional


class CliRunner:
    """Calls an AUSAXS CLI function (e.g. cli_saxs_fitter) on a worker thread.

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

    def start(self, func_name: str, argv: list[str],
              on_line: Callable[[str], None], on_done: Callable[[int], None]):
        if self._busy:
            raise RuntimeError("a fit is already running")

        try:
            from ..wrapper.AUSAXS import AUSAXS
            func = getattr(AUSAXS().lib().functions, func_name)
        except Exception as e:
            on_line(f"AUSAXS library unavailable: {e}\n")
            on_done(1)
            return

        self._busy = True
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
            target=self._work, args=(func, argv, saved_fds, reader), daemon=True
        ).start()
        self._widget.after(self.POLL_MS, self._poll)

    def _read_pipe(self, read_fd: int):
        with os.fdopen(read_fd, errors="replace") as pipe:
            for line in pipe:
                self._queue.put(line)

    def _work(self, func, argv: list[str], saved_fds: tuple[int, int], reader: threading.Thread):
        from ..__main__ import _call_cli
        returncode = 1
        try:
            returncode = _call_cli(func, argv)
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


class _Done:
    """Sentinel carrying the outcome of a rigid-body refinement."""
    def __init__(self, result, error, error_streamed=False):
        self.result = result    # np.ndarray of [q, I, Ierr, I_model], or None on failure/validation
        self.error = error      # Exception, or None on success
        # True if the error was already streamed to the console by the backend (run-time
        # errors), so the GUI can avoid printing a duplicate
        self.error_streamed = error_streamed


class RigidbodyRunner:
    """Runs a rigid-body refinement script through the Rigidbody API on a worker thread.

    Backend console output is streamed via the library's output callback (invoked from
    the worker thread) into a thread-safe queue, then drained on the GUI thread.
    on_line(str) receives each output line; on_done(_Done) is called once on completion.
    For validation, result is None and error indicates success/failure.
    """

    POLL_MS = 100

    def __init__(self, tk_widget):
        self._widget = tk_widget
        self._busy = False
        self._queue: queue.Queue = queue.Queue()
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_done: Optional[Callable[["_Done"], None]] = None

    def running(self) -> bool:
        return self._busy

    def start(self, script: str, validate_only: bool,
              on_line: Callable[[str], None], on_done: Callable[["_Done"], None]):
        if self._busy:
            raise RuntimeError("a refinement is already running")

        try:
            from ..wrapper.AUSAXS import AUSAXS
            from ..wrapper.Rigidbody import prepare_rigidbody_refinement
            from ..wrapper.Output import set_output_callback, reset_output_callback
            # instantiate the singleton up front: the output-callback and rigidbody
            # wrappers reach the library via AUSAXS.lib() classmethods, which require
            # the instance to already exist.
            if not AUSAXS().ready():
                raise RuntimeError(f"library failed to initialize: {AUSAXS().init_error()}")
        except Exception as e:
            on_line(f"AUSAXS library unavailable: {e}\n")
            on_done(_Done(None, e))
            return

        self._busy = True
        self._on_line = on_line
        self._on_done = on_done
        self._queue = queue.Queue()

        threading.Thread(
            target=self._work,
            args=(script, validate_only, prepare_rigidbody_refinement,
                  set_output_callback, reset_output_callback),
            daemon=True,
        ).start()
        self._widget.after(self.POLL_MS, self._poll)

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
        self._queue.put(_Done(result, error, error_streamed))

    def _poll(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _Done):
                self._busy = False
                if self._on_done:
                    self._on_done(item)
                return
            if self._on_line:
                self._on_line(item)
        self._widget.after(self.POLL_MS, self._poll)

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

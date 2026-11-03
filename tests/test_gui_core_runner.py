"""Runs isolated_gui_core_runner.py in a child process; see its docstring for why it can't share this one."""

import os
import subprocess
import sys


def test_gui_core_runner_in_isolation():
    cases = os.path.join(os.path.dirname(__file__), "isolated_gui_core_runner.py")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", cases],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-2000:]

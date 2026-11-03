import importlib
import pkgutil
import subprocess
import sys

import pyausaxs


def test_import_all_submodules():
    """Importing every submodule must not raise."""
    # Subpackages that can't be imported in a headless test environment:
    #  - __pyinstaller: build-time hooks that require PyInstaller (an optional dependency)
    #  - gui: gui.py calls matplotlib.use("TkAgg") at import, which needs an interactive display. Its toolkit-independent
    #    core is the exception, and is imported like any other module.
    skip = (".__pyinstaller", ".gui")
    keep = (".gui.core",)

    failures = []
    for module in pkgutil.walk_packages(pyausaxs.__path__, prefix="pyausaxs."):
        if any(s in module.name for s in skip) and not any(k in module.name for k in keep):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as e:  # noqa: BLE001 - we want to report any failure
            failures.append(f"{module.name}: {e!r}")

    assert not failures, "Some modules failed to import:\n" + "\n".join(failures)


def test_gui_core_needs_no_gui_toolkit():
    """The GUI core is shared with front-ends built on other toolkits (e.g. SasView's Qt calculators), so importing it must neither pull
    in tkinter nor select a matplotlib backend for the embedding application."""
    script = (
        "import sys, matplotlib\n"
        "backend = matplotlib.rcParams._get_backend_or_none()\n"
        "import pyausaxs.gui.core.script, pyausaxs.gui.core.syntax, pyausaxs.gui.core.files, pyausaxs.gui.core.loads\n"
        "import pyausaxs.gui.core.palette, pyausaxs.gui.core.plotting\n"
        "assert 'tkinter' not in sys.modules, 'tkinter was imported'\n"
        "assert 'pyausaxs.gui.gui' not in sys.modules, 'the tkinter interface was imported'\n"
        "assert matplotlib.rcParams._get_backend_or_none() == backend, 'a matplotlib backend was selected'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

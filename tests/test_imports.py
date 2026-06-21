import importlib
import pkgutil

import pyausaxs


def test_import_all_submodules():
    """Importing every submodule must not raise."""
    # Subpackages that can't be imported in a headless test environment:
    #  - __pyinstaller: build-time hooks that require PyInstaller (an optional dependency)
    #  - gui: gui.py calls matplotlib.use("TkAgg") at import, which needs an interactive display
    skip = (".__pyinstaller", ".gui")

    failures = []
    for module in pkgutil.walk_packages(pyausaxs.__path__, prefix="pyausaxs."):
        if any(s in module.name for s in skip):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as e:  # noqa: BLE001 - we want to report any failure
            failures.append(f"{module.name}: {e!r}")

    assert not failures, "Some modules failed to import:\n" + "\n".join(failures)

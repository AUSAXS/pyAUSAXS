import importlib
import pkgutil

import pyausaxs


def test_import_all_submodules():
    """Importing every submodule must not raise.

    Besides catching plain import errors, this exercises the lazy backend binding: each
    wrapper registers its function signatures as an import-time side effect, so importing
    every module triggers every register() call (e.g. Filetypes, which is only imported
    lazily by the GUI at runtime).
    """
    failures = []
    for module in pkgutil.walk_packages(pyausaxs.__path__, prefix="pyausaxs."):
        # __pyinstaller holds build-time hooks that require PyInstaller (an optional dependency)
        if ".__pyinstaller" in module.name:
            continue
        try:
            importlib.import_module(module.name)
        except Exception as e:  # noqa: BLE001 - we want to report any failure
            failures.append(f"{module.name}: {e!r}")

    assert not failures, "Some modules failed to import:\n" + "\n".join(failures)

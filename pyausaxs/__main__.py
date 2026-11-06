import argparse
import importlib.util
import sys

# Tools implemented in Python, which are intercepted here: tool -> (help description, pip extra, modules that must be importable).
# A tool without an extra has no optional dependencies. Everything else, including the top-level help, version, and licence, is
# forwarded to the backend's own `ausaxs` entrypoint, which lists these tools after its own.
_PYTHON_TOOLS = {
    "plot":  ("Plot results from other tools",            "plots", ("matplotlib", "scipy")),
    "gui":   ("Start the graphical interface",            "gui",   ("matplotlib", "scipy", "_tkinter")),
    "setup": ("Choose which AUSAXS backend library to use", None,  ()),
}


def _extra_available(tool: str) -> bool:
    """Check if the optional dependencies for a tool are available by trying to find their modules."""
    for module in _PYTHON_TOOLS[tool][2]:
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


def _extra_missing_error(tool: str) -> int:
    extra = _PYTHON_TOOLS[tool][1]
    print(
        f"Error: the '{extra}' extension is not installed."
        f"\nPlease install it by running the command \"pip install pyausaxs[{extra}]\".",
        file=sys.stderr,
    )
    return 1


def _build_setup_parser() -> argparse.ArgumentParser:
    setup = argparse.ArgumentParser(prog="ausaxs setup", description="Configure the AUSAXS backend.")
    grp = setup.add_mutually_exclusive_group()
    grp.add_argument(
        "--relink", metavar="PATH",
        help="use the backend library at PATH instead of the bundled one",
    )
    grp.add_argument(
        "--reset", action="store_true",
        help="forget any relinked path and use the bundled library",
    )
    grp.add_argument(
        "--show", action="store_true",
        help="show which backend library is currently in use (default)",
    )
    return setup


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    match argv[0] if argv else None:
        case "setup":
            return _run_setup(_build_setup_parser().parse_args(argv[1:]))
        case "gui":
            if not _extra_available("gui"):
                return _extra_missing_error("gui")
            print("Warning: The Python GUI is highly experimental. Use at your own risk.", file=sys.stderr)
            from .gui.gui import main as gui_main
            return gui_main(argv[1:])
        case "plot":
            if not _extra_available("plot"):
                return _extra_missing_error("plot")
            from .plot.plot import main as plot_main
            return plot_main(argv[1:])
        case _:
            return run_backend_cli(argv)


def run_backend_cli(args: list[str]) -> int:
    """Run the backend's `ausaxs` entrypoint in-process with the given arguments (excluding the program name)."""
    from .wrapper.AUSAXS import AUSAXS
    if not AUSAXS.ready():
        print(f"Error: the AUSAXS backend failed to load: {AUSAXS.init_error()}", file=sys.stderr)
        return 1

    functions = AUSAXS.lib().functions
    for name, (description, extra, _) in _PYTHON_TOOLS.items():
        if extra is not None and not _extra_available(name):
            description += f" (requires \"pip install pyausaxs[{extra}]\")"
        functions.cli_ausaxs_add_tool(name.encode("utf-8"), description.encode("utf-8"))

    import ctypes as ct
    c_args = [arg.encode("utf-8") for arg in ["ausaxs", *args]]
    c_argv = (ct.c_char_p * len(c_args))(*c_args)
    return functions.cli_ausaxs(len(c_args), c_argv)


def _run_setup(ns):
    from pathlib import Path
    from . import loader
    from .wrapper.AUSAXS import AUSAXS

    if ns.relink:
        previous = loader.get_relink_path()
        try:
            cache = loader.set_relink_path(ns.relink)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # validate the newly relinked library actually loads before committing to it
        if not AUSAXS.ready():
            error = AUSAXS.init_error()
            if previous is not None:
                loader.set_relink_path(previous)
            else:
                loader.clear_relink_path()
            AUSAXS.reset_singleton()
            print(f"Error: '{ns.relink}' failed to load as an AUSAXS backend: {error}", file=sys.stderr)
            return 1

        linked = loader.get_relink_path()
        if "ausaxs" not in Path(linked).stem.lower():
            print(
                f"Warning: '{Path(linked).name}' does not look like an AUSAXS "
                "library (its name should contain 'ausaxs').",
                file=sys.stderr,
            )
        print(f"Relinked AUSAXS backend to: {linked}")
        print(f"(stored in {cache})")
        return 0

    if ns.reset:
        if loader.clear_relink_path():
            print("Reset: now using the bundled AUSAXS backend.")
        else:
            print("No relink configured; already using the bundled backend.")
        return 0

    # default: --show
    path, origin = loader.resolve_lib()
    print(f"AUSAXS backend in use: {path}")
    print({
        "relink": "  (relinked; reset with: ausaxs setup --reset)",
        "environment": "  (set via the AUSAXS_LIB environment variable)",
        "bundled": "  (bundled)",
        "prefix": "  (installed in the environment prefix)",
    }[origin])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

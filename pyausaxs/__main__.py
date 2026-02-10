import argparse
import sys

from .wrapper.Filetypes import _is_pdb_file, _is_saxs_data_file, _is_rigidbody_config_file
from .wrapper.Rigidbody import run_rigidbody_script

from .__init__ import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="PyAUSAXS Command Line Interface")
    parser.add_argument('--version', action='version', version=f'PyAUSAXS {__version__}')
    parser.add_argument('paths', nargs='*', help='File(s) or script to run')
    args = parser.parse_args(argv)

    if not args.paths:
        parser.print_help()
        return 0

    # Currently support a single rigidbody-config script path.
    if len(args.paths) == 1:
        path = args.paths[0]
        try:
            if _is_rigidbody_config_file(path):
                run_rigidbody_script(path)
                return 0
            else:
                print(f"File is not a rigidbody config: {path}", file=sys.stderr)
                return 2
        except Exception as e:
            print(f"Error running rigidbody script: {e}", file=sys.stderr)
            return 1

    print("No supported operation for given arguments.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
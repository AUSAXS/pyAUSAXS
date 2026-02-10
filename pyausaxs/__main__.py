import argparse

from .__init__ import __version__

def main(argv=None):
    parser = argparse.ArgumentParser(description="PyAUSAXS Command Line Interface")
    parser.add_argument('--version', action='version', version=f'PyAUSAXS {__version__}')
    parser.add_argument('-')
    args = parser.parse_args(argv)

if __name__ == "__main__":
    main()
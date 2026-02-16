"""Download plotting scripts from github.com/AUSAXS/AUSAXS.

Downloads `plot.py` and `plot_helper.py` from the AUSAXS repository
scripts directory into this package (default: `pyausaxs/plot`).

Usage:
  python fetch_plot_scripts.py
  python fetch_plot_scripts.py --target ./some/dir
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from typing import List


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/AUSAXS/AUSAXS/master/scripts"
DEFAULT_TARGET = os.path.dirname(__file__)
DEFAULT_FILES: List[str] = ["plot.py", "plot_helper.py"]


def download_file(url: str, dest_path: str) -> None:
    """Download a file from URL to dest_path."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pyAUSAXS"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def fetch_files(base_url: str, filenames: List[str], target_dir: str) -> List[str]:
    """Download files from GitHub into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    saved: List[str] = []
    for name in filenames:
        url = f"{base_url}/{name}"
        dest = os.path.join(target_dir, name)
        print(f"Downloading {url} -> {dest}")
        try:
            download_file(url, dest)
            saved.append(dest)
        except Exception as e:
            print(f"Error downloading {name}: {e}", file=sys.stderr)
    return saved


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download AUSAXS plotting scripts from GitHub")
    p.add_argument("--url", default=GITHUB_RAW_BASE, help="Base URL for raw GitHub files")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target directory")
    p.add_argument("--files", nargs="*", help="Files to download (default: plot.py, plot_helper.py)")
    args = p.parse_args(argv)

    files = args.files if args.files else DEFAULT_FILES
    saved = fetch_files(args.url, files, args.target)
    print(f"Done: downloaded {len(saved)} files to {args.target}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

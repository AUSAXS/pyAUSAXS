#!/usr/bin/env python3
"""
Maintainer utility: sync the plot scripts vendored in pyausaxs/plot with the
canonical copies in the AUSAXS repository. The vendored copies are committed,
so this never runs as part of a build.
"""
import os
import sys
import urllib.request


def fetch_plot_scripts():
	"""Download plot scripts from the AUSAXS GitHub repository."""
	base_url = "https://raw.githubusercontent.com/AUSAXS/AUSAXS/master/scripts"
	files = ["plot.py", "plot_helper.py"]
	target_dir = os.path.join(os.path.dirname(__file__), "..", "pyausaxs", "plot")

	os.makedirs(target_dir, exist_ok=True)

	for filename in files:
		url = f"{base_url}/{filename}"
		dest = os.path.join(target_dir, filename)

		print(f"Downloading {url} -> {dest}")
		try:
			req = urllib.request.Request(url, headers={"User-Agent": "pyAUSAXS-build"})
			with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
				f.write(resp.read())
			print(f"Successfully downloaded {filename}")
		except Exception as e:
			print(f"Warning: Failed to download {filename}: {e}", file=sys.stderr)


if __name__ == "__main__":
	fetch_plot_scripts()

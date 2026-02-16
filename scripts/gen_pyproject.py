#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request


def detect_library_path() -> str:
	plat = sys.platform
	if plat.startswith("linux"):
		return "resources/libausaxs.so"
	if plat == "darwin":
		return "resources/libausaxs.dylib"
	if plat.startswith("win") or plat == "cygwin":
		return "resources/libausaxs.dll"
	else:
		raise RuntimeError(f"Unsupported platform: {plat}")


def fetch_plot_scripts():
	"""Download plot scripts from AUSAXS GitHub repository."""
	base_url = "https://raw.githubusercontent.com/AUSAXS/AUSAXS/master/scripts"
	files = ["plot.py", "plot_helper.py"]
	target_dir = os.path.join(os.path.dirname(__file__), "..", "pyausaxs", "plot")
	
	os.makedirs(target_dir, exist_ok=True)
	
	for filename in files:
		url = f"{base_url}/{filename}"
		dest = os.path.join(target_dir, filename)
		
		# Skip if file already exists
		if os.path.isfile(dest):
			print(f"Plot script already exists: {dest}")
			continue
			
		print(f"Downloading {url} -> {dest}")
		try:
			req = urllib.request.Request(url, headers={"User-Agent": "pyAUSAXS-build"})
			with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
				f.write(resp.read())
			print(f"Successfully downloaded {filename}")
		except Exception as e:
			print(f"Warning: Failed to download {filename}: {e}", file=sys.stderr)


# Fetch plot scripts before generating pyproject.toml
fetch_plot_scripts()

lib = detect_library_path()
lib_relative = lib

template = "pyproject.template.toml"
if not os.path.exists(template):
    raise FileNotFoundError(f"Template not found!")

with open(template, "r", encoding="utf-8") as fh:
    template = fh.read()

content = template.replace("LIBRARY_PLACEHOLDER", lib)

with open("pyproject.toml", "w", encoding="utf-8") as fh:
    fh.write(content)
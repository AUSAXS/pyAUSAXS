<p align="center">
  <a href="https://ausaxs.github.io/pyausaxs/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AUSAXS/pyAUSAXS/media/title_light.png">
      <img alt="pyAUSAXS" src="https://raw.githubusercontent.com/AUSAXS/pyAUSAXS/media/title_dark.png">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyausaxs"><img alt="PyPI version" src="https://img.shields.io/pypi/v/pyausaxs"></a>
  <a href="https://python.org/downloads"><img alt="Python 3.10 or newer" src="https://img.shields.io/badge/python-%E2%89%A53.10-blue"></a>
  <a href="https://pypistats.org/packages/pyausaxs"><img alt="PyPI downloads" src="https://img.shields.io/pypi/dm/pyausaxs"></a>
  <a href="https://ausaxs.github.io/pyausaxs/"><img alt="Documentation" src="https://img.shields.io/badge/docs-ausaxs.github.io-blue"></a>
  <a href="https://doi.org/10.1107/S160057672500562X"><img alt="Paper DOI" src="https://img.shields.io/badge/DOI-10.1107%2FS160057672500562X-blue"></a>
</p>

pyAUSAXS is the Python interface to [AUSAXS](https://github.com/AUSAXS/AUSAXS), a fast calculator of small-angle X-ray
and neutron scattering (SAXS and SANS) profiles from atomic structures. It runs the same C++ engine as the AUSAXS
command-line tools, and returns every result as a NumPy array.

**Documentation, guides and examples you can run in the browser: [ausaxs.github.io/pyausaxs](https://ausaxs.github.io/pyausaxs/)**

## Installation
pyAUSAXS needs Python 3.10 or newer, and comes with prebuilt libraries for Linux, macOS and Windows:
```bash
python -m pip install pyausaxs
```
Install `pyausaxs[plots]` to also get matplotlib for plotting, or `pyausaxs[all]` for the graphical interface as well.
If the installation fails, see the [troubleshooting](https://ausaxs.github.io/getting-started/troubleshooting/) page.

## Example
Read a structure, add a hydration shell, calculate its SAXS profile and fit it to a measurement. The example uses
hen egg-white lysozyme, [2epe.pdb](https://raw.githubusercontent.com/AUSAXS/AUSAXS/master/tests/files/2epe.pdb), and a
SAXS measurement of it, [2epe.dat](https://raw.githubusercontent.com/AUSAXS/AUSAXS/master/tests/files/2epe.dat):
```python
import pyausaxs as ausaxs
import matplotlib.pyplot as plt

mol = ausaxs.create_molecule("2epe.pdb")
mol.hydrate()                      # place a hydration shell around it
q, I = mol.debye()                 # the SAXS profile, as two NumPy arrays
fit = mol.fit("2epe.dat")          # fit it to the measurement
fit.plot()                         # the data, the fitted profile and the residuals
plt.show()
```
The [first steps](https://ausaxs.github.io/pyausaxs/first-steps/) guide goes through it in detail, and you can run it
there without installing anything.

## What it can do
- [Calculate SAXS profiles](https://ausaxs.github.io/pyausaxs/scattering-profiles/) of PDB, mmCIF and XYZ structures,
  with form factors, a hydration shell and excluded volume, or as a pure Debye sum.
- [Fit structures to SAXS data](https://ausaxs.github.io/pyausaxs/fitting/), or use AUSAXS inside your own fit.
- [Calculate SANS profiles](https://ausaxs.github.io/applications/sans/) in any H₂O/D₂O mixture.
- [Process MD trajectories](https://ausaxs.github.io/pyausaxs/trajectories/) and ensembles, frame by frame.
- [Differentiate the Debye equation](https://ausaxs.github.io/pyausaxs/gradients/) with respect to the atomic
  coordinates, for refinement and machine learning.
- [Calculate the scattering of geometric shapes](https://ausaxs.github.io/pyausaxs/shapes/), and of
  [crystals and nanoparticles](https://ausaxs.github.io/applications/crystals/).
- Run the `ausaxs` [command-line tools](https://ausaxs.github.io/documentation/) and their graphical interface, which
  the package installs as well.

Every function and setting is listed in the [API reference](https://ausaxs.github.io/pyausaxs/reference/). The
[examples](https://github.com/AUSAXS/pyAUSAXS/tree/master/examples) folder has complete, runnable scripts.

## Contributing
Found a bug, have a suggestion, or want to contribute? See the
[contributor guidelines](https://github.com/AUSAXS/pyAUSAXS/blob/master/CONTRIBUTING.md).

## Citing
If you use pyAUSAXS in published work, please [cite the relevant AUSAXS paper](https://ausaxs.github.io/documentation/citing/).

pyAUSAXS is free software under the [GNU Lesser General Public License v3.0](https://github.com/AUSAXS/pyAUSAXS/blob/master/LICENSE).

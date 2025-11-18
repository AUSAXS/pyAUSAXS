![title_light](../media/title_dark.png?raw=true#gh-light-mode-only)
![title_dark](../media/title_light.png?raw=true#gh-dark-mode-only)

[![pip install](https://img.shields.io/badge/pip%20install-pyausaxs-blue)](#installation)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://python.org/downloads)
[![PyPI - Version](https://img.shields.io/pypi/v/pyausaxs)](https://pypi.org/project/pyausaxs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyausaxs)](https://pypistats.org/package/pyausaxs)

Welcome to `pyAUSAXS`, the perhaps fastest Python tool for evaluating the scattering intensity of biological samples and crystals. 
`pyAUSAXS` is a Python wrapper around [`AUSAXS`](https://github.com/AUSAXS/AUSAXS), the high-performance C++ backend, offering easy
access to most of its features. 

## Who is this for?
`pyAUSAXS` offers highly efficient calculation of the expected scattering intensity of your structures. These calculations include
form factors, and both hydration shell and excluded volume modeling. For a full overview of how it works, see the `AUSAXS` article:
doi: [10.1107/S160057672500562X](https://doi.org/10.1107/S160057672500562X).

## Installation
To install, simply run:
```bash
pip install pyausaxs
```
and you are good to go!

## Usage
Proper documentation is currently being written. For now, please refer to the runnable examples [here](https://github.com/AUSAXS/pyAUSAXS/tree/master/examples).

## Contributing
Are you encountering problems, have feedback or suggestions, or are you considering contributing to the project? Please check out the [contributor guidelines](CONTRIBUTING.md).

## Citation
If you use `pyAUSAXS` in published work, please cite the following paper:
Small-angle X-ray scattering profile calculation for high-resolution models of biomacromolecules  
(doi: [10.1107/S160057672500562X](https://doi.org/10.1107/S160057672500562X))

*This project is licenced under the GNU Lesser General Public Licence v3.0.*
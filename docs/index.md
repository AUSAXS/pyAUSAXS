# pyAUSAXS API reference

This is the reference for every public function and class of pyAUSAXS. For an introduction, see the [pyAUSAXS guides](https://ausaxs.github.io/pyausaxs/); for runnable scripts, the [examples](https://github.com/AUSAXS/pyAUSAXS/tree/master/examples).

Everything is imported from the top-level module:
```python
import pyausaxs as ausaxs
```

| Name | Description | Page |
|------|-------------|------|
| `create_molecule` | Read a structure, or create one from coordinate arrays | [Molecules](molecules.md) |
| `read_data`, `create_datafile` | Read a measured profile, or create one from arrays | [Data and structure files](data.md#datasets) |
| `read_pdb` | Read the records of a PDB or mmCIF file | [Data and structure files](data.md#structure-files) |
| `Molecule.fit` | Fit a molecule to data | [Fitting](fitting.md) |
| `manual_fit` | Evaluate the fit model for parameters of your choice | [Fitting](fitting.md#manual-fits) |
| `settings`, `ExvModel`, `ExvTable` | Options of all calculations | [Settings](settings.md) |
| `prepare_rigidbody_refinement`, `Rigidbody` | Run a rigid-body refinement script | [Rigid-body refinement](rigidbody.md) |
| `shape` | Scattering of homogeneous bodies | [Utilities](utilities.md#homogeneous-shapes) |
| `form_factor` | Look up atomic form factors and volumes | [Utilities](utilities.md#form-factors) |
| `unoptimized` | The exact Debye sum, for testing | [Utilities](utilities.md#the-exact-debye-sum) |
| `sasview` | The plain Debye sum on raw arrays | [Utilities](utilities.md#plain-debye-sum-on-arrays) |
| `set_output_callback`, `reset_output_callback` | Redirect the messages of AUSAXS | [Utilities](utilities.md#output-messages) |
| `advanced` | Manage the lifetime of the underlying C++ objects | [Utilities](utilities.md#object-lifetime) |

## Conventions
- **Units.** Lengths are in Å, and q in Å⁻¹, in both input and output.
- **Arrays.** Functions accept lists, tuples or NumPy arrays, and return NumPy arrays of `float64`.
- **Errors.** A failure in AUSAXS raises a `RuntimeError` with its message, e.g. for a file that cannot be read.
- **Settings are global.** A change to [`settings`](settings.md) applies to every calculation after it, in every molecule, until it is changed again.
- **Version.** `ausaxs.__version__` is the version of the installed package.

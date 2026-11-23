# Utilities

## Homogeneous shapes
For an introduction, see [scattering from geometric shapes](https://ausaxs.github.io/pyausaxs/shapes/).

### shape.lattice
```python
ausaxs.shape.lattice(inside, extent, spacing=1.0) -> Lattice
```
Fill a shape with the cells of a cubic lattice.
- `inside`: a function `(x, y, z) -> bool`, true inside the shape. It is called once with arrays of coordinates in Å, so it must work on NumPy arrays.
- `extent`: the half-width in Å of the box to search, one value or one per axis.
- `spacing`: the lattice spacing in Å.

### shape.Lattice
```python
ausaxs.shape.Lattice(cells, spacing)
```
A homogeneous body made of lattice cells, each with a Gaussian form factor. `cells` is an `(N, 3)` array of cell centres in Å, all on one lattice with the given `spacing`.

| Member | Description |
|--------|-------------|
| `debye(q=None)` | `(q, I)`: the scattering of the body, on the default q axis or at the given q values. The intensity is for unit density, so at q = 0 it equals the volume squared. |
| `volume` | The total volume of the cells in Å³. |
| `len(body)` | The number of cells. |
| `cells`, `spacing` | The cell centres and the spacing. |

## Form factors
| Function | Returns |
|----------|---------|
| `form_factor.valid_types()` | The names of the form factor types, e.g. `"C"`, `"CH2"`, `"OH"`. |
| `form_factor.get_five_gaussian_coefficients(type)` | `(a, b, c)`: the coefficients of the five-Gaussian approximation of the form factor of `type`, with `a` and `b` arrays of five values. |
| `form_factor.get_current_exv_volume(type)` | The excluded volume of `type` in Å³, from the current excluded volume table. |

## The exact Debye sum
```python
ausaxs.unoptimized.debye_exact(mol, q=None) -> (q, I)
```
The plain Debye sum over every pair of atoms, without the distance histogram, and without form factors. It is much slower than `mol.debye_raw()`, and meant for testing and validation.

## Plain Debye sum on arrays
```python
ausaxs.sasview.debye_no_ff(q, x, y, z, weights) -> I
```
The plain Debye sum of weighted point scatterers at the given q values, from coordinate arrays, without creating a molecule. No form factors or excluded volume are used.

## Output messages
| Function | Description |
|----------|-------------|
| `set_output_callback(function)` | Send every message of AUSAXS to `function(text)` instead of printing it. |
| `reset_output_callback()` | Print the messages again. |

To turn messages off instead, use `settings.general(verbose=False, warnings=False)`.

## Object lifetime
Molecules, datasets and fit results are Python handles to objects in the C++ library, which are freed when the Python object is garbage collected. `ausaxs.advanced` gives manual control, e.g. to free a large molecule right away:

| Function | Description |
|----------|-------------|
| `advanced.deallocate(obj)` | Free the C++ object now. `obj` cannot be used afterwards. |
| `advanced.detach(obj)` | Do not free the C++ object when `obj` is garbage collected. |
| `advanced.attach(obj)` | Free it with `obj` again. |

## The command line
Installing pyAUSAXS installs the `ausaxs` command, also available as `python -m pyausaxs`. Its tools are described in the [documentation](https://ausaxs.github.io/documentation/). Two of them are part of the Python package:
- `ausaxs gui` starts the [graphical interface](https://ausaxs.github.io/documentation/cli/fit/#graphical-user-interface).
- `ausaxs setup` chooses which AUSAXS library pyAUSAXS uses: `--relink PATH` for [your own build](https://ausaxs.github.io/getting-started/compiling/#using-your-build-from-python), `--reset` for the bundled one, and `--show` to print the one in use. The `AUSAXS_LIB` environment variable also selects a library.

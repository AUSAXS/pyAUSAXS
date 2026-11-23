# Molecules

A `Molecule` holds the atoms of a structure with their form factors, and its hydration shell. It calculates the scattering profile, the distance histogram and the radius of gyration, and fits the structure to data. For an introduction, see [SAXS profiles in Python](https://ausaxs.github.io/pyausaxs/scattering-profiles/).

## create_molecule
```python
ausaxs.create_molecule(filename: str) -> Molecule
ausaxs.create_molecule(pdb: PDBfile) -> Molecule
ausaxs.create_molecule(x, y, z, weights) -> Molecule
```
Create a molecule from one of:
- **a structure file**: PDB, mmCIF or XYZ. Water molecules in the file become the hydration shell. Explicit hydrogens are left out, and their scattering added to the atom they are bound to, unless the `keep_hydrogens` [setting](settings.md) is on.
- **a structure read with [`read_pdb`](data.md#structure-files)**.
- **four arrays of equal length**: the coordinates in Å and the scattering weight of each point. These points have no atom type, so use them with `debye_raw()`.

The structure is read with the [settings](settings.md) in effect at that moment, e.g. for the radiation (X-rays or neutrons) and the hydrogens. Change them before creating the molecule.

## Scattering profiles

| Method | Returns | Description |
|--------|---------|-------------|
| `debye(q=None)` | `(q, I)` | The profile with form factors, hydration shell and excluded volume, using the excluded volume model of the [settings](settings.md#excluded-volume). |
| `debye_raw(q=None, form_factors=False)` | `(q, I)` | The plain Debye sum, with every atom and water as a point scatterer with its weight, and no excluded volume. With `form_factors=True`, each atom carries its form factor instead; this needs a molecule read from a file. |
| `debye_raw_vjp(q, v)` | `(n_atoms, 3)` array | The vector-Jacobian product of `debye_raw(q)` with respect to the atomic coordinates: $\sum_q v(q)\,\partial I(q)/\partial \mathbf{r}_i$. It is the derivative of the exact sum over atom pairs, for point scatterers without form factors. See [gradients](https://ausaxs.github.io/pyausaxs/gradients/). |

Without `q`, the profiles are calculated on the default q axis, `settings.q_axis()`. With `q`, an array of values in Å⁻¹, they are calculated at those values. `debye_raw` accepts q beyond 1 Å⁻¹; the solvent models of `debye` are meant for q up to about 1 Å⁻¹.

## Hydration shell

| Method | Description |
|--------|-------------|
| `hydrate()` | Place a new hydration shell around the molecule, replacing the current one. The shell is placed at random, so each call gives a slightly different profile. |
| `clear_hydration()` | Remove the hydration shell. |

## Atoms and coordinates

| Method | Returns | Description |
|--------|---------|-------------|
| `atoms()` | `[x, y, z, weights, form_factors]` | The atoms, as arrays. `form_factors` holds the form factor type of each atom, e.g. `"CH2"` for a carbon with two implicit hydrogens. |
| `waters()` | `[x, y, z, weights]` | The water molecules of the hydration shell. |
| `atomic_dict()` | `dict` | The atoms, keyed `"x"`, `"y"`, `"z"`, `"weights"`, `"ff_type"`. |
| `water_dict()` | `dict` | The waters, with the same keys. |
| `set_coordinates(coords)` | | Replace the coordinates of the atoms with an `(n_atoms, 3)` array, in the order of `atoms()`. The weights and form factors are kept. Call `hydrate()` afterwards for a shell that matches the new coordinates. |
| `Rg()`, `radius_of_gyration()` | `float` | The radius of gyration of the atoms in Å. |

## Distance histograms

| Method | Returns | Description |
|--------|---------|-------------|
| `distance_histogram()`, `histogram()` | `Histogram` | The distance histogram the profile is calculated from, split by atom and water pairs. |
| `distance_histogram_raw()` | `(bins, counts)` | The distance histogram of point scatterers, without form factors or excluded volume. |

A `Histogram` has the methods:

| Method | Description |
|--------|-------------|
| `bins()` | The distance of each bin in Å. With weighted bins, the mean distance of the pairs in the bin rather than its centre. |
| `counts()`, `counts_total()` | The weighted counts of all pairs. |
| `counts_aa()`, `counts_aw()`, `counts_ww()` | The counts of atom-atom, atom-water and water-water pairs. The water counts have an arbitrary scale; multiply by the fitted `cw` to put them on the scale of the atoms. |
| `truncate()` | Remove the empty bins at the end of the histogram. |
| `get_bin_width()` | The bin width in Å, from the `bin_width` setting. |

## Fitting
`fit(data)` fits the molecule to a dataset, and returns a `FitResult`. See [Fitting](fitting.md).

# Settings

`ausaxs.settings` holds the options of every calculation. The settings are global: a change applies to everything calculated after it, in every molecule, until it is changed again or reset. Some, such as the radiation and the hydrogens, are applied when a structure is read, so set them before calling `create_molecule`.

## Grouped options
The most common options are set through a few functions. Each parameter is optional, and only the ones you pass are changed.

### Excluded volume
```python
ausaxs.settings.exv(exv_model=ExvModel.simple)
```
Choose the excluded volume model, as an `ExvModel` or its name:

| `ExvModel` | Description |
|------------|-------------|
| `simple` | The default. Treats all atoms as equivalent; fast, with no excluded volume parameter to fit. |
| `fraser` | A Gaussian sphere per atom, as in CRYSOL, FoXS and Pepsi-SAXS. |
| `grid` | A continuous grid of dummy atoms filling the molecule. |
| `vacuum` | No excluded volume: the atoms with their form factors in vacuum. |
| `none` | No excluded volume, and a shared form factor for all atoms. |

The models are described in the [AUSAXS paper](https://doi.org/10.1107/S160057672500562X), and their command-line options in [ausaxs fit](https://ausaxs.github.io/documentation/cli/fit/).

### Fitting
```python
ausaxs.settings.fit(fit_hydration=None, fit_excluded_volume=None, fit_solvent_density=None)
```
Choose which solvent parameters a [fit](fitting.md) optimises: the hydration shell (`cw`, on by default), the excluded volume (`cx`, off) and the solvent density (`cr`, off).

### q range and histogram
```python
ausaxs.settings.histogram(qmin=None, qmax=None, unit=None, weighted_bins=None, bin_width=None)
```

| Parameter | Description |
|-----------|-------------|
| `qmin`, `qmax` | The q range in Å⁻¹: of the default q axis, and of the data kept when reading a dataset. Default 10⁻⁴ to 0.5. |
| `unit` | The unit of q in data files, `"A"` or `"nm"`. By default it is guessed from the values. |
| `weighted_bins` | Place each distance bin at the mean distance of its pairs rather than at its centre. By default decided per structure. |
| `bin_width` | The width of the distance bins in Å. Default 0.25; use a smaller width for crystals. |

`ausaxs.settings.q_axis()` returns the default q axis.

### Molecules
```python
ausaxs.settings.molecule(throw_on_unknown_atom=None, implicit_hydrogens=None, use_occupancy=None, exv_table=None)
```

| Parameter | Description |
|-----------|-------------|
| `throw_on_unknown_atom` | Raise an error on an atom type AUSAXS does not know, instead of continuing. On by default. |
| `implicit_hydrogens` | Add the scattering of missing hydrogens to the atoms that carry them. On by default. |
| `use_occupancy` | Weight each atom by its occupancy. On by default. |
| `exv_table` | The table of atomic volumes for the excluded volume, as an `ExvTable` or its name: `traube`, `voronoi_implicit_H`, `voronoi_explicit_H`, `minimum_fluctutation_implicit_H` (the default), `minimum_fluctutation_explicit_H` or `vdw`. `"voronoi"` and `"mf"` are short for the implicit-hydrogen versions. |

### Grid
```python
ausaxs.settings.grid(cell_width=None, expansion_factor=None, min_exv_radius=None)
```
The grid used by the hydration shell and the grid excluded volume model: the width of its cells in Å, how far it extends beyond the molecule relative to its size, and the minimum radius of each atom in the grid, which sets the size of the grid excluded volume.

### General
```python
ausaxs.settings.general(offline=None, verbose=None, warnings=None, threads=None)
```

| Parameter | Description |
|-----------|-------------|
| `offline` | Do not go online to look up the hydrogens of unusual residues. |
| `verbose` | Print progress messages. On by default. |
| `warnings` | Print warnings. On by default. |
| `threads` | The number of threads used by the calculations. By default, one less than the number of cores. |

## Any setting by name
Every setting can also be read and changed by its name. `set` takes the value as a string:
```python
ausaxs.settings.set("radiation", "neutron")
ausaxs.settings.set("keep_hydrogens", "true")
ausaxs.settings.get("qmax")          # returns the value with its type, here a float
```
The names are those of the settings files of the command-line tools. To see all of them with their current values, save them to a file.

| Function | Description |
|----------|-------------|
| `get(name)` | The value of a setting. |
| `set(name, value)` | Change a setting; `value` is a string. Raises a `RuntimeError` for an unknown name. |
| `reset()` | Restore every setting to its default. |
| `refresh(new_settings)` | `reset()`, then `set` each name and value of the dictionary `new_settings`. |
| `save(path)` | Write every setting to a text file. |
| `load(path)` | Read the settings from a file written by `save` or by the command-line tools. |
| `q_axis()` | The default q axis, as an array. |

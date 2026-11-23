# Data and structure files

## Datasets
A `Datafile` holds a measured scattering profile: q, the intensity I, and its error.

### read_data
```python
ausaxs.read_data(filename: str) -> Datafile
```
Read a profile from a text file with columns q, I and the error of I. Further columns are ignored. The unit of q is guessed from its values, and converted to Å⁻¹; set the `unit` [setting](settings.md#q-range-and-histogram) to `"A"` or `"nm"` beforehand if the guess is wrong. Points outside the q range of the settings are removed.

### create_datafile
```python
ausaxs.create_datafile(q, I, Ierr) -> Datafile
```
Create a dataset from three arrays of equal length, with q in Å⁻¹. Points outside the q range of the settings are removed, as when reading a file.

### Datafile methods

| Method | Returns |
|--------|---------|
| `q()`, `I()`, `Ierr()` | The q values in Å⁻¹, the intensities, and their errors. |
| `data()` | `[q, I, Ierr]` |
| `dict()` | `{"q": ..., "I": ..., "Ierr": ...}` |

## Structure files
`read_pdb` gives access to the records of a structure file, e.g. to select atoms or read their residue names. To calculate scattering, pass the result to [`create_molecule`](molecules.md#create_molecule), or give `create_molecule` the file name directly.

### read_pdb
```python
ausaxs.read_pdb(filename: str) -> PDBfile
```
Read a PDB or mmCIF file. Water molecules are not included in the records.

### PDBfile methods
Each method returns one array with an entry per atom.

| Method | Content |
|--------|---------|
| `serial()` | Atom serial numbers |
| `names()` | Atom names, e.g. `"CA"` |
| `resnames()` | Residue names, e.g. `"LYS"` |
| `chain_ids()` | Chain identifiers |
| `res_seqs()` | Residue sequence numbers |
| `icodes()` | Insertion codes |
| `coordinates()` | `(x, y, z)`, three arrays in Å |
| `occupancies()` | Occupancies |
| `temp_factors()` | Temperature factors |
| `elements()` | Element symbols |
| `charges()` | Formal charges |
| `dict()` | All of the above in a dictionary, keyed by the PDB field names (`"resName"`, `"chainID"`, `"x"`, ...) |
| `data()` | All of the above in a list, in the order of the PDB format |

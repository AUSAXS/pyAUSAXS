# Rigid-body refinement

pyAUSAXS runs the same rigid-body refinement scripts as `ausaxs rigidbody`. The script language, with every element and argument, is described in the [rigid-body refinement reference](https://ausaxs.github.io/documentation/cli/rigidbody/); for an introduction, see the [rigid-body refinement application](https://ausaxs.github.io/applications/rigid-body-refinement/).

## prepare_rigidbody_refinement
```python
ausaxs.prepare_rigidbody_refinement(script: str) -> Rigidbody
```
Parse a refinement script, given as its text, not its file name. Paths in the script are relative to the folder Python runs in.
```python
with open("dimer.conf") as f:
    rb = ausaxs.prepare_rigidbody_refinement(f.read())
rb.validate()
result = rb.run()
q, I, Ierr, I_fit = result.T
```

## Rigidbody methods

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | | Check the script, and raise a `RuntimeError` if it is invalid. |
| `run()` | `(n, 4)` array | Run the refinement. The columns are q, I and Ierr of the data, and the fitted profile of the final structure. |
| `stop_run()` | | Ask a running refinement to stop. It stops at the start of its next iteration, and `run()` then returns the best structure found so far. Can be called from another thread. |
| `body_names()` | `list[str]` | The names of the bodies, after any merging or deletion in the script. |
| `preview_structure()` | `dict` | The structure with its symmetry copies, for drawing: `coords` (an `(N, 3)` array), and per atom its `body`, symmetry `copy`, `residue_seq` and whether it `is_ca`, plus the `constraints` between atoms. |
| `symmetry_layout()` | `dict` | One row per symmetry copy: its `body`, `copy`, `symmetry` and `replica` index, its symmetry `type` and its `name`. |

## Following a running refinement
An `update structure` element in the script publishes the current structure during a run, e.g. to show it in a viewer. To receive it:

| Function | Description |
|----------|-------------|
| `Rigidbody.set_live_consumer(True)` | Register as a consumer. Without a consumer, `update` elements do nothing. |
| `Rigidbody.live_structure()` | `(coords, version)`: the most recently published coordinates as an `(N, 3)` array, in the order of `preview_structure()`, or `None` if nothing was published yet. `version` increases with every new structure. Safe to call from another thread than `run()`. |

## Valid elements
`Rigidbody.get_valid_elements()` lists the elements a script can contain, `Rigidbody.get_valid_arguments(element)` the arguments of one element, and `Rigidbody.get_valid_elements_and_arguments()` both, as a dictionary.

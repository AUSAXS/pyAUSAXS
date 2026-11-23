# Fitting

For an introduction with examples, see [fitting SAXS data in Python](https://ausaxs.github.io/pyausaxs/fitting/).

## Molecule.fit
```python
mol.fit(data: str | Datafile) -> FitResult
```
Fit the profile of the molecule to a dataset, given as a file name or as a [`Datafile`](data.md#datasets). The model compared with the data is $a \cdot I(q) + b$, where $I(q)$ is the profile of the molecule for the fitted solvent parameters:

| Parameter | Description | Fitted when |
|-----------|-------------|-------------|
| `a` | Intensity scale | Always |
| `b` | Constant background | Always |
| `cw` | Scattering strength of the hydration shell | `settings.fit(fit_hydration=True)`, the default |
| `cx` | Scale of the excluded volume, relative to its default | `settings.fit(fit_excluded_volume=True)`; not with the `simple` excluded volume model |
| `cr` | Density of the bulk solvent, relative to its default | `settings.fit(fit_solvent_density=True)`; discouraged |

The excluded volume model and the other options are chosen in the [settings](settings.md).

## FitResult

| Method | Returns |
|--------|---------|
| `chi2()` | $\chi^2$ of the fit. |
| `dof()` | Degrees of freedom: the number of data points minus the number of fitted parameters. |
| `chi2r()` | Reduced $\chi^2$, `chi2()/dof()`. |
| `fit_parameters()` | `{name: (value, lower error, upper error)}` for each fitted parameter. The lower error is negative. |
| `data_curve()` | `(q, I, Ierr)` of the data used in the fit, after the q range was applied. |
| `fitted_curve()` | The fitted model at the q values of `data_curve()`. |
| `fit_curves()` | `[q, I, Ierr, I_fit]` |
| `residuals()` | The residuals $(I - I_\text{fit})/I_\text{err}$. |
| `plot()` | A matplotlib figure of the data with the fit, and the residuals below. Needs `pyausaxs[plots]`. |

## Manual fits
```python
ausaxs.manual_fit(mol: Molecule) -> IterativeFit
ausaxs.manual_fit(mol: Molecule, data: Datafile) -> IterativeFit
ausaxs.manual_fit(mol: Molecule, q) -> IterativeFit
```
Prepare the model profile of a molecule for evaluation at parameter values of your choice, for use in your own fit. It is evaluated at the q values of `data`, at the given `q`, or without either on the default q axis.

### IterativeFit.evaluate
```python
model.evaluate(params) -> np.ndarray
model.evaluate(params, q) -> np.ndarray
```
The model profile for the solvent parameters `params`, one value per enabled parameter in the order `cw`, `cx`, `cr` (with the default settings, only `cw`). The scale `a` and background `b` are not applied. With `q`, the profile is evaluated at those q values instead, for this call only.

The enabled parameters are those of `settings.fit` when `manual_fit` is called. A wrong number of parameters raises a `RuntimeError`.

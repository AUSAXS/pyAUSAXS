# Example: Advanced model fitting with pyausaxs
# Make sure you've read the basic example first (debye_scattering.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt

# Prepare the structure:
mol = ausaxs.create_molecule("tests/files/2epe.pdb")
mol.hydrate()

# For repeated use, it's better to read the data file once and pass it to the fitter.
data = ausaxs.read_data("tests/files/2epe.dat")

# By default, the Simple excluded volume model is used as described in the AUSAXS paper: https://doi.org/10.1107/S160057672500562X 
fit_simple = mol.fit("tests/files/2epe.dat")

# The other two common models are the Gaussian sphere model as described by Fraser et al.: https://doi.org/10.1107/S0021889878014296
# This is the model also used in CRYSOL, FoXS, Pepsi-SAXS, and most other similar programs.
ausaxs.settings.exv(ausaxs.ExvModel.fraser)
fit_fraser = mol.fit(data)

# Finally, we can also use the grid-based excluded volume model, also from the AUSAXS paper.
ausaxs.settings.exv(ausaxs.ExvModel.grid)
fit_grid = mol.fit(data)

plt.figure(figsize=(10, 6))
plt.errorbar(data.q(), data.I(), yerr=data.Ierr(), fmt='k.', markersize=4, capsize=2)
plt.plot(data.q(), fit_simple.fitted_curve(), label=f'Simple, χ²={fit_simple.chi2():.2f}')
plt.plot(data.q(), fit_fraser.fitted_curve(), label=f'Fraser, χ²={fit_fraser.chi2():.2f}')
plt.plot(data.q(), fit_grid.fitted_curve(), label=f'Grid, χ²={fit_grid.chi2():.2f}')
plt.xlabel("q")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
plt.show()

# The Fraser model seems broken. That's because it _must_ be used with excluded volume fitting enabled. Let's fix that:
ausaxs.settings.fit(fit_excluded_volume=True)
ausaxs.settings.exv(ausaxs.ExvModel.fraser)
fit_fraser_exv = mol.fit(data)

ausaxs.settings.exv(ausaxs.ExvModel.grid)
fit_grid_exv = mol.fit(data)

plt.figure(figsize=(10, 6))
plt.errorbar(data.q(), data.I(), yerr=data.Ierr(), fmt='k.', markersize=4, capsize=2)
plt.plot(data.q(), fit_simple.fitted_curve(), label=f'Simple, χ²={fit_simple.chi2():.2f}')
plt.plot(data.q(), fit_fraser_exv.fitted_curve(), label=f'Fraser (fitted exv), χ²={fit_fraser_exv.chi2():.2f}')
plt.plot(data.q(), fit_grid_exv.fitted_curve(), label=f'Grid (fitted exv), χ²={fit_grid_exv.chi2():.2f}')
plt.xlabel("q")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
plt.show()

### Summary ###
# In this example, we demonstrated how to perform advanced model fitting using different excluded volume models in pyausaxs.
# We showed how the Fraser model should be used with care, since it requires excluded volume fitting to be enabled for 
# meaningful results. The grid-based model usually does not require this additional fit parameter, and is therefore expected
# to be the most robust choice for general use. However, due to its high computational cost, the default model is instead 
# the Simple model, which provides a good balance between accuracy and performance for most applications.
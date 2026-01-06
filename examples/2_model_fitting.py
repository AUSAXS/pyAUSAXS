# Example: Model fitting with pyausaxs
# Make sure you've read the basic example first (debye_scattering.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt

# Prepare the structure:
mol = ausaxs.create_molecule("tests/files/2epe.pdb")
mol.hydrate()

# Now it's time to change some settings. Everything is available through the settings object.
ausaxs.settings.fit(
    fit_hydration=True,         # The hydration shell density should always be fitted.
    fit_excluded_volume=True,   # Enable fitting of excluded volume too.
    fit_solvent_density=False   # Do not fit solvent density. This is generally not recommended to avoid overfitting.
)

fit = mol.fit("tests/files/2epe.dat")
print(f"Chi-squared of the fit: {fit.chi2()}")

q_data, I_data, I_data_err = fit.data_curve()
I_fit = fit.fitted_curve()
residuals = fit.residuals()

# Plotting the results:
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

plt.sca(ax[0])
plt.errorbar(q_data, I_data, yerr=I_data_err, fmt='k.', markersize=4, capsize=2)
plt.plot(q_data, I_fit, 'r-', label='Fit')
plt.ylabel("I(q)")
plt.loglog()

plt.sca(ax[1])
plt.plot(q_data, residuals, 'k.', markersize=4)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("q")
plt.ylabel("Residuals")
plt.xscale('log')
plt.show()
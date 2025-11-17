# Example: Crystal scattering calculations with pyausaxs

import pyausaxs as ausaxs
import matplotlib.pyplot as plt
import numpy as np

# First read in the structure file and create a Molecule object.
# Currently, only limited support for crystal structures in PDB, CIF, & XYZ formats are available. 
# If you encounter issues, please open an issue on GitHub: https://github.com/AUSAXS/pyAUSAXS/issues
mol = ausaxs.create_molecule("tests/files/Ag_crystal.cif")

# Say we are interested in the wide-angle scattering from this crystal.
# AUSAXS only supports q-values <= 1Å⁻¹, since the excluded volume and hydration shell models become too
# imprecise at higher angles. However, to support crystal scattering where such effects are unimportant, 
# the `mol.debye` methods support custom q-ranges beyond this limit. 
q = np.linspace(0.01, 2, 500)

# First, we calculate the exact Debye intensity for comparison.
_, I_exact = ausaxs.unoptimized.debye_exact(mol, q)

# Next, we evaluate the scattering using the much faster binning approximation normally used in AUSAXS. 
_, I_binned = mol.debye_raw(q)

# Finally, we can plot the results.
plt.figure(figsize=(8,5))
plt.plot(q, I_exact, label="Exact Debye")
plt.plot(q, I_binned, label="Binning approximation")
plt.xlabel("q")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
# plt.show()

# We get excellent agreement! This is due to AUSAXS relying on _weighted_ bins, where the bin centers are 
# shifted to better represent their contents. Compare this to using simple uniform bins, which most other
# binning schemes use:
ausaxs.settings.histogram(weighted_bins=False)  # disable weighted bins
_, I_binned_uniform = mol.debye_raw(q)
plt.figure(figsize=(8,5))
plt.plot(q, I_exact, label="Exact Debye")
plt.plot(q, I_binned_uniform, label="Binning approximation (uniform bins)")
plt.xlabel("q")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
# plt.show()

# The deviations are _much_ larger here, since the highly ordered nature of the crystal structure means that
# small inaccuracies in the bin positions lead to large phase errors in the calculated intensities. However, 
# for larger crystals with more distances competing for space in each bin, even the weighted bins may become
# overwhelmed. In such cases, we can reduce the bin width to improve accuracy:
ausaxs.settings.histogram(
    weighted_bins=True,     # re-enable weighted bins
    bin_width=0.1,          # smaller bin width
    bin_count=4000          # remember to also set the bin count high enough to cover the distance range
)
_, I_binned_fine = mol.debye_raw(q)

ausaxs.settings.histogram(bin_width=0.5)
_, I_binned_coarse = mol.debye_raw(q)

# We make a deviation plot to better visualize the improvement:
r_binned = I_binned / I_exact
r_binned_uniform = I_binned_uniform / I_exact
r_binned_fine = I_binned_fine / I_exact
r_binned_coarse = I_binned_coarse / I_exact

plt.figure(figsize=(8,5))
plt.plot(q, r_binned_uniform, label="Uniform, width = 0.25", lw=2)
plt.plot(q, r_binned_coarse, label="Weighted, width = 0.5", lw=2)
plt.plot(q, r_binned, label="Weighted, width = 0.25", lw=2)
plt.plot(q, r_binned_fine, label="Weighted, width = 0.1", lw=2)
plt.xlabel("q")
plt.ylabel("Deviation ratio")
plt.loglog()
plt.axhline(1, color='k', ls='--', lw=0.8)
plt.ylim(0.5, 2)
plt.yticks([0.5, 1, 2], ["0.5", "1", "2"])
plt.gca().minorticks_off()
plt.legend()
plt.show()
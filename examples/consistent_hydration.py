# Example: Advanced Debye scattering calculations with pyausaxs
# Make sure you've read the basic example first (debye_scattering.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt
import numpy as np

# Prepare the structure:
mol = ausaxs.create_molecule("tests/files/2epe.pdb")

# The hydration shell is randomly generated each time, which can lead to small variations in the scattering.
# This is especially true at higher q-values. To get a more consistent result, we should average over multiple configurations.
ausaxs.settings.set_hist_settings(qmax=1)

num_configs = 100
Iq_list = []
for _ in range(num_configs):
    mol.hydrate()
    q, Iq = mol.debye()
    Iq_list.append(Iq)

Iq_avg = np.mean(Iq_list, axis=0)

# Plotting the results:
plt.figure(figsize=(10, 6))
for Iq in Iq_list:
    plt.plot(q, Iq, "gray", alpha=0.5)
plt.plot(q, Iq_avg, "r-")
plt.xlabel("q (1/Angstrom)")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
plt.show()

# In this case, the variation is not too bad. For some structures, however, the differences can be significant.
# This is especially true for larger molecules with internal cavities. 
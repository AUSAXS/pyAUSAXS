# Example: Basic Debye scattering calculations with pyausaxs

import pyausaxs as ausaxs
import matplotlib.pyplot as plt

# First read in the structure file and create a Molecule object.
# Here, we use a PDB file as an example. PDBx/mmCIF files are also supported.
mol = ausaxs.create_molecule("tests/files/2epe.pdb")

# Next, we have to decide what type of calculation we want to do. 
# The simplest is the pure debye calculation, which computes the scattering without atomic form factors or excluded volume contributions. 
q_raw, Iq_raw = mol.debye_raw()

# For more realistic scattering, we can instead use the Simple excluded volume model described in the AUSAXS paper: https://doi.org/10.1107/S160057672500562X
# This calculation will also include atomic form factors for common elements (H, C, N, O, S). All heavier elements are treated as argon. 
q_simple, Iq_simple = mol.debye()

# In solution, molecules are surrounded by solvent. We can easily add a hydration shell to our molecule:
mol.hydrate()
q_hydrated, Iq_hydrated = mol.debye()

# Plotting the results:
plt.figure(figsize=(10, 6))
plt.plot(q_raw, Iq_raw, label="Raw Debye")
plt.plot(q_simple, Iq_simple, label="Simple Excluded Volume")
plt.plot(q_hydrated, Iq_hydrated, label="With Hydration Shell")
plt.xlabel("q")
plt.ylabel("I(q)")
plt.loglog()
plt.legend()
plt.show()

# And that's it! For more advanced usage, refer to the other examples. 
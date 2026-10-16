# Example: Scattering from homogeneous shapes
# Make sure you've read the basic example first (debye_scattering.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt
import numpy as np

# Any homogeneous body can be modelled by filling it with the cells of a cubic lattice, each with a Gaussian form factor.
# This is the same representation used by the Grid excluded volume model. We start with a sphere, which has an exact solution.
R = 30
q = np.linspace(1e-3, 0.5, 500)
cells = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= R**2, extent=R, spacing=1)  # (N, 3) array of cell centres
_, Iq = ausaxs.shape.debye(cells, spacing=1, q_vals=q)

# A finer lattice gives a better approximation of the body. The pair distances are counted with a fast Fourier transform,
# so even millions of cells are fast.
cells_fine = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= R**2, extent=R, spacing=0.25)
_, Iq_fine = ausaxs.shape.debye(cells_fine, spacing=0.25, q_vals=q)

x = q*R
Iq_exact = (4*np.pi*R**3*(np.sin(x) - x*np.cos(x))/x**3)**2

plt.figure(figsize=(10, 6))
plt.plot(q, Iq_exact, "k", label="Exact sphere")
plt.plot(q, Iq, "--", label=f"1 Å lattice ({len(cells):,} cells)")
plt.plot(q, Iq_fine, ":", label=f"0.25 Å lattice ({len(cells_fine):,} cells)")
plt.xlabel("q [Å⁻¹]")
plt.ylabel("I(q)")
plt.semilogy()
plt.legend()
plt.show()

# Any shape works the same way. Here we compare a barbell, two spheres joined by a thin rod, against the two spheres alone.
def ends(x, y, z, R=10, L=20):
    return (x**2 + y**2 + (z - L)**2 <= R**2) | (x**2 + y**2 + (z + L)**2 <= R**2)

def barbell(x, y, z, r=4, L=20):
    return ends(x, y, z) | ((x**2 + y**2 <= r**2) & (np.abs(z) <= L))

cells_barbell = ausaxs.shape.lattice(barbell, extent=(10, 10, 30), spacing=0.25)
cells_ends = ausaxs.shape.lattice(ends, extent=(10, 10, 30), spacing=0.25)
_, Iq_barbell = ausaxs.shape.debye(cells_barbell, spacing=0.25, q_vals=q)
_, Iq_ends = ausaxs.shape.debye(cells_ends, spacing=0.25, q_vals=q)

plt.figure(figsize=(10, 6))
plt.plot(q, Iq_barbell, label=f"Barbell ({len(cells_barbell):,} cells)")
plt.plot(q, Iq_ends, label=f"Two spheres ({len(cells_ends):,} cells)")
plt.xlabel("q [Å⁻¹]")
plt.ylabel("I(q)")
plt.semilogy()
plt.legend()
plt.show()

# The oscillations at low q are the interference between the two spheres, 40 Å apart. The minimum at q = 0.45 is the first
# zero of the spheres themselves (qR = 4.49), which the rod partially fills in.

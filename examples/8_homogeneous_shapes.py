# Example: Scattering from homogeneous shapes with the lattice transform

import pyausaxs as ausaxs
import matplotlib.pyplot as plt
import numpy as np

# The Grid excluded volume model represents the volume a molecule displaces as a set of cubic lattice cells, each with
# a Gaussian form factor. The same machinery can model any homogeneous body: fill the shape with cells and calculate its
# scattering. Since the cells sit on a lattice, AUSAXS counts their pair distances with a fast Fourier transform rather
# than a pair loop, so even bodies with millions of cells take well under a second.

# A homogeneous sphere is the one shape with a simple exact answer, so we start there.
R = 30
cells = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= R**2, extent=R, spacing=1)
q = np.linspace(1e-3, 0.5, 500)
_, I_grid = ausaxs.shape.debye(cells, spacing=1, q_vals=q)

def sphere(q, R):
    V, x = 4/3*np.pi*R**3, q*R
    return (V*3*(np.sin(x) - x*np.cos(x))/x**3)**2

# The cells fill a slightly different volume than the ideal sphere, so compare against the sphere of the same volume.
R_eff = (3*len(cells)/(4*np.pi))**(1/3)
I_exact = sphere(q, R_eff)

# Each cell is a Gaussian rather than a sharp cube, which softens the surface. This multiplies the intensity by a known
# envelope, exp(-2 V^(2/3) q^2/(4 pi)) with V = spacing^3: about 4% at q = 0.5, and 15% at q = 1 for a 1 Å lattice.
# Dividing it out leaves the point-lattice sum, which matches the continuous sphere almost exactly.
envelope = np.exp(-2*q**2/(4*np.pi))

plt.figure(figsize=(8, 5))
plt.plot(q, I_exact, "k", label="Exact sphere")
plt.plot(q, I_grid, "--", label="Gaussian cells, 1 Å")
plt.plot(q, I_grid/envelope, ":", label="Gaussian cells, envelope removed")
plt.xlabel("q [Å⁻¹]")
plt.ylabel("I(q)")
plt.semilogy()
plt.legend()
# plt.show()

# The residual shrinks with the lattice spacing, since the envelope does. A 0.5 Å lattice has 8x the cells, but the
# transform barely notices:
cells_fine = ausaxs.shape.lattice(lambda x, y, z: x**2 + y**2 + z**2 <= R**2, extent=R, spacing=0.5)
_, I_fine = ausaxs.shape.debye(cells_fine, spacing=0.5, q_vals=q)
I_exact_fine = sphere(q, (3*len(cells_fine)*0.5**3/(4*np.pi))**(1/3))

# Near the minima of the sphere the relative error is meaningless, so we measure the error relative to the local
# height of the curve instead.
def error(I, I_ref):
    height = np.maximum.accumulate(I_ref[::-1])[::-1]
    return np.abs(I - I_ref)/height

plt.figure(figsize=(8, 5))
plt.plot(q, error(I_grid, I_exact), label="1 Å")
plt.plot(q, error(I_fine, I_exact_fine), label="0.5 Å")
plt.plot(q, error(I_grid/envelope, I_exact), label="1 Å, envelope removed")
plt.xlabel("q [Å⁻¹]")
plt.ylabel("error relative to the local curve height")
plt.legend()
# plt.show()

# The remaining ~0.5% is mostly the distance histogram, not the lattice: its 0.25 Å bins are coarse for a homogeneous
# body, whose intensity at high q is the small remainder of a near-perfect cancellation. Narrower bins remove it:
# ausaxs.settings.histogram(bin_width=0.05) takes the 0.5 Å sphere above from a 1.6% to a 0.04% binning error.

# Any shape works the same way. Here is a barbell: two spheres joined by a thin rod, next to the same two spheres alone.
def ends(x, y, z, R=10, L=20):
    return (x**2 + y**2 + (z - L)**2 <= R**2) | (x**2 + y**2 + (z + L)**2 <= R**2)

def barbell(x, y, z, r=4, L=20):
    return ends(x, y, z) | ((x**2 + y**2 <= r**2) & (np.abs(z) <= L))

_, I_barbell = ausaxs.shape.debye(ausaxs.shape.lattice(barbell, extent=(10, 10, 30)), spacing=1, q_vals=q)
_, I_ends = ausaxs.shape.debye(ausaxs.shape.lattice(ends, extent=(10, 10, 30)), spacing=1, q_vals=q)

plt.figure(figsize=(8, 5))
plt.plot(q, I_barbell, label="Barbell")
plt.plot(q, I_ends, label="The two spheres alone")
plt.xlabel("q [Å⁻¹]")
plt.ylabel("I(q)")
plt.semilogy()
plt.legend()
plt.show()

# The shoulders at low q are the interference between the two spheres, 40 Å apart. The deep minimum at q = 0.45 is the
# spheres' own first zero (qR = 4.49); the rod is only ~12% of the volume, but it is enough to fill that zero in.

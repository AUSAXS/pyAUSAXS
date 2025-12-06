# Example: Obtaining distance histograms with pyausaxs
# Make sure you've read the basic example first (debye_scattering.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt

mol = ausaxs.create_molecule("tests/files/2epe.pdb")
mol.hydrate()                   # generate new hydration shell
dist = mol.distance_histogram() # get the distance histogram (triggers calculation)

# We now have access to the full range of distance histogram data. 
# However, since most of the bins representing larger distances are unused, we would like to trim the data
# before visualizing it. We therefore find the last non-zero bin and only plot up to that point:
last_nonzero = dist.counts_total().nonzero()[0][-1] + 1
bins = dist.bins()[:last_nonzero]
p_aa = dist.counts_aa()[:last_nonzero]       # atom-atom distances
p_aw = dist.counts_aw()[:last_nonzero]       # atom-water distances
p_ww = dist.counts_ww()[:last_nonzero]       # water-water distances
p_total = dist.counts_total()[:last_nonzero] # total distances

# Make the plot:
plt.figure(figsize=(10, 6))
plt.plot(bins, p_aa, label="atom-atom")
plt.plot(bins, p_aw, label="atom-water")
plt.plot(bins, p_ww, label="water-water")
plt.plot(bins, p_total, label="total")
plt.xlabel("Distance (Å)")
plt.ylabel("Weighted counts")
plt.title("Distance Histogram")
plt.legend()
plt.show()

# Summary: We have calculated and visualized the partial and total distance histograms for a hydrated molecule.
# The `counts_aa` method gives the distance histogram for all atom-atom pairs, `counts_aw` for atom-water pairs, 
# `counts_ww` for water-water pairs, and `counts_total` for the total histogram.
# Note that the hydration counts have an arbitrary weight. To convert it to something meaningful, multiply it by
# the hydration shell scattering power obtained from a fit. 
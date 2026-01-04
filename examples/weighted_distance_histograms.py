# Example: Examining the weighted bins used in AUSAXS.
# Make sure you've read the distance histogram example first (distance_histograms.py)

import pyausaxs as ausaxs
import matplotlib.pyplot as plt
import numpy as np

#! Need a way to query this value through the Histogram interface
bw = 0.1
ausaxs.settings.histogram(bin_width=bw, bin_count=10000)  # Set the bin width for the distance histogram

# This time, we will use a crystal structure to highlight the weighted bins used in AUSAXS. 
mol = ausaxs.create_molecule("tests/files/Ag_crystal.cif")
dist = mol.distance_histogram() # get the distance histogram (triggers calculation)
dist.truncate()
bins = dist.bins()              # weighted distance axis
p_total = dist.counts_total()   # total distances

# To better visualize the distance peaks, we use a bar plot:
plt.figure(figsize=(10, 6))
plt.bar(bins, p_total, width=0.008, alpha=0.5, label="total (bar)")
plt.xlabel("Distance (Å)")
plt.ylabel("Weighted counts")
plt.title("Distance Histogram")
plt.legend()
plt.show()

# To clearly illustrate how the bin centers shifts, we recalculate the distance histogram without 
# using weighted bins to compare with:
#! we need persistent settings; they shouldn't reset on each invocation
#! instead, we should define a ausaxs.settings.reset() method which delegates
#! to a backend-call (since we need the same functionality in AUSAXS).
ausaxs.settings.histogram(weighted_bins=False, bin_width=bw, bin_count=10000)
mol = ausaxs.create_molecule("tests/files/Ag_crystal.cif")  # recreate molecule to refresh the histogram calculator
dist = mol.distance_histogram()
dist.truncate()
unweighted_bins = dist.bins()               # unweighted distance axis
unweighted_p_total = dist.counts_total()    # matching total distances

# We then plot the bin edges, along with the actual bin centers used in both calculations. 
# This clearly illustrates the deviations within each bin due to the weighted bin approach. 
ind = int(5/bw)
plt.figure(figsize=(14, 4))
bin_edges = np.arange(bw/2, len(bins)*bw, bw)[ind:ind+50]
plt.eventplot(bin_edges, orientation='horizontal', colors='k')
plt.plot(unweighted_bins[ind+1:ind+50], np.ones_like(unweighted_bins[ind+1:ind+50]), 'k.', alpha=0.7)
plt.plot(bins[ind+1:ind+50], np.ones_like(bins[ind+1:ind+50]), 'r.')
plt.gca().get_yaxis().set_visible(False)                        # hide y-axis
plt.gca().spines[['left', 'top', 'right']].set_visible(False)   # hide all but the bottom frame
plt.xlabel("Distance (Å)")
plt.tight_layout()
plt.show()
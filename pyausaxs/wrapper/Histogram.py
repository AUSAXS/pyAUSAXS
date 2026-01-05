import numpy as np
from pyausaxs.wrapper.settings import settings

class Histogram:
    __slots__ = ['_bins', '_aa', '_aw', '_ww']
    def __init__(self, bins, aa, aw, ww):
        self._bins = bins
        self._aa = aa
        self._aw = aw
        self._ww = ww
    
    def truncate(self) -> None:
        """
        Truncate the data to remove all zero-values following the last nonzero index. 
        The resulting data will still have the same size. 
        """
        i = len(self._bins)
        for i in range(len(self._bins)-1, 1, -1):
            if self._aa[i] + self._aw[i] + self._ww[i] != 0:
                break
        self._bins = self._bins[:i]
        self._aa = self._aa[:i]
        self._aw = self._aw[:i]
        self._ww = self._ww[:i]

    def counts_aa(self) -> np.ndarray:
        """The atomic-atomic distance counts."""
        return self._aa

    def counts_ww(self) -> np.ndarray:
        """The water-water (hydration shell) distance counts."""
        return self._ww

    def counts_aw(self) -> np.ndarray:
        """The atomic-water (hydration shell) distance counts."""
        return self._aw

    def counts_total(self) -> np.ndarray:
        """The total distance counts."""
        return self._aa + self._ww + self._aw

    def counts(self) -> np.ndarray:
        """The total distance counts."""
        return self.counts_total()

    def bins(self) -> np.ndarray:
        return self._bins
    
    @staticmethod
    def get_bin_width() -> float:
        """Get the current histogram bin width setting."""
        return float(settings._get("bin_width")) #! remove redundant conversion once AUSAXS setting API has been refactored
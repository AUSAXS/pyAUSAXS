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
        Remove the trailing zero-valued bins, so the data ends at the last bin with a nonzero count.
        """
        nonzero = np.flatnonzero(np.asarray(self._aa) + np.asarray(self._aw) + np.asarray(self._ww))
        n = nonzero[-1] + 1 if nonzero.size else 0
        self._bins = self._bins[:n]
        self._aa = self._aa[:n]
        self._aw = self._aw[:n]
        self._ww = self._ww[:n]

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
        return float(settings.get("bin_width")) #! remove redundant conversion once AUSAXS setting API has been refactored
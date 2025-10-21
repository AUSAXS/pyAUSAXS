import numpy as np

class Histogram:
    __slots__ = ['_bins', '_aa', '_aw', '_ww']
    def __init__(self, bins, aa, aw, ww):
        self._bins = bins
        self._aa = aa
        self._aw = aw
        self._ww = ww

    def counts_aa(self) -> np.ndarray:
        return self._aa

    def counts_ww(self) -> np.ndarray:
        return self._ww

    def counts_aw(self) -> np.ndarray:
        return self._aw

    def counts_total(self) -> np.ndarray:
        return self._aa + self._ww + self._aw

    def counts(self) -> np.ndarray:
        return self.counts_total()

    def bins(self) -> np.ndarray:
        return self._bins
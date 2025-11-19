import math
import numpy as np

class simple_cube:
    @staticmethod
    def points():
        corners = [
            (-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1),
            (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1),
            (0, 0, 0)
        ]
        w = np.ones(len(corners), dtype=float)
        xs = np.array([p[0] for p in corners], dtype=float)
        ys = np.array([p[1] for p in corners], dtype=float)
        zs = np.array([p[2] for p in corners], dtype=float)
        return xs, ys, zs, w

    @staticmethod
    def hist():
        return [
            [0.0, math.sqrt(3.0), 2.0, math.sqrt(8.0), math.sqrt(12.0)],
            [9, 16, 24, 24, 8]
        ]

    @staticmethod
    def debye(q):
        I_expected = np.zeros_like(q, dtype=float)
        dist, mult = simple_cube.hist()
        for m, r in zip(mult, dist):
            if r == 0.0:
                I_expected += m
            else:
                qr = q * r
                term = np.empty_like(qr)
                mask = qr == 0
                term[mask] = 1.0
                term[~mask] = np.sin(qr[~mask]) / qr[~mask]
                I_expected += m * term
        return I_expected

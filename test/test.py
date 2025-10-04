import sys
import math
import numpy as np
from pyausaxs import ausaxs

def test_cube_debye_ci():
    q = np.linspace(0.0, 1.0, 100)

    # Build coordinates for 8 cube corners and center (0,0,0) -> total 9 points
    corners = [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1),
               (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]
    pts = corners + [(0.0, 0.0, 0.0)]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    zs = np.array([p[2] for p in pts], dtype=float)
    ws = np.ones(len(pts), dtype=float)

    # exact distances (unique) and multiplicities
    d_vals = [0.0, math.sqrt(3.0), 2.0, math.sqrt(8.0), math.sqrt(12.0)]
    multiplicities = [9, 16, 24, 24, 8]

    # compute expected I(q) from multiplicities and distances (weights=1)
    I_expected = np.zeros_like(q, dtype=float)
    for m, r in zip(multiplicities, d_vals):
        if r == 0.0:
            I_expected += m
        else:
            qr = q * r
            term = np.empty_like(qr)
            mask = qr == 0
            term[mask] = 1.0
            term[~mask] = np.sin(qr[~mask]) / qr[~mask]
            I_expected += m * term

    # Instantiate native wrapper and compute
    try:
        a = ausaxs()
    except Exception as e:
        print("ERROR: Failed to initialize AUSAXS native library:", e, file=sys.stderr)
        sys.exit(1)

    try:
        I_native = a.debye(q, xs, ys, zs, ws)
    except Exception as e:
        print("ERROR: AUSAXS native debye call failed:", e, file=sys.stderr)
        sys.exit(1)

    rtol = 1e-5
    atol = 1e-8
    if not np.allclose(I_native, I_expected, rtol=rtol, atol=atol):
        diff = np.abs(I_native - I_expected)
        max_idx = int(np.argmax(diff))
        print("FAIL: native result differs from expected")
        print(f"Max diff at q={q[max_idx]:.6f}: native={I_native[max_idx]:.12e}, expected={I_expected[max_idx]:.12e}, diff={diff[max_idx]:.12e}")
        for i in range(min(10, len(q))):
            print(f"q={q[i]:.6f}  native={I_native[i]:.12e}  expected={I_expected[i]:.12e}  diff={I_native[i]-I_expected[i]:.12e}")
        sys.exit(1)
    print("PASS: native AUSAXS debye matches expected cube result")

if __name__ == '__main__':
    test_cube_debye_ci()
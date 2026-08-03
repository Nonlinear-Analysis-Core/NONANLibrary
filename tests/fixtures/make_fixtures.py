#!/usr/bin/env python3
"""Write the shared cross-language reference series.

    python3 tests/fixtures/make_fixtures.py

The CSVs it produces are COMMITTED. Both the MATLAB and the Python suites read
these exact samples, which is what makes 1e-12 agreement a meaningful assertion
rather than a statistical hand-wave. Regenerate only deliberately: changing a
fixture changes every cross-language tolerance that depends on it.

Written with numpy's explicit Generator and a pinned seed so it reproduces
across numpy versions and platforms. Values are written at full float64
precision (%.17g) -- %.6f would silently cap agreement at 1e-6 and make the
tests unable to see a real port divergence.
"""

import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260727


def fgn(n, H, seed):
    """Fractional Gaussian noise by conjugate-symmetric spectral synthesis."""
    m = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.arange(1, m // 2 + 1) / m
    amp = f ** (-(2 * H - 1) / 2)
    rng = np.random.default_rng(seed)
    half = amp * np.exp(2j * np.pi * rng.random(m // 2))
    spec = np.concatenate([[0], half, np.conj(half[:-1][::-1])])
    z = np.real(np.fft.ifft(spec))[:n]
    return (z - z.mean()) / z.std()


def lorenz(n):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    dt, skip = 0.003, 10
    f = lambda v: np.array([s * (v[1] - v[0]),
                            v[0] * (r - v[2]) - v[1],
                            v[0] * v[1] - b * v[2]])

    def rk4(v):
        k1 = f(v); k2 = f(v + dt / 2 * k1)
        k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    v = np.array([1.0, 1.0, 1.0])
    for _ in range(5000):
        v = rk4(v)
    out = np.empty(n)
    for i in range(n):
        for _ in range(skip):
            v = rk4(v)
        out[i] = v[0]
    return out


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(4096)

    series = {
        # DFA known answers
        "white_4096": rng.standard_normal(4096),
        "fgn_H30_4096": fgn(4096, 0.30, SEED + 1),
        "fgn_H50_4096": fgn(4096, 0.50, SEED + 2),
        "fgn_H85_4096": fgn(4096, 0.85, SEED + 3),
        # surrogate contract inputs
        "ar1_phi70_512": None,          # filled below
        "henon_512": None,
        # EMD regression inputs
        "emd_with_zero_512": None,
        "emd_no_zero_512": None,
        # short pseudo-periodic series for the rho search
        "pseudoperiodic_300": (np.sin(2 * np.pi * t[:300] / 25)
                               + 0.35 * np.sin(2 * np.pi * t[:300] / 8)
                               + 0.05 * rng.standard_normal(300)),
    }

    e = rng.standard_normal(512 + 500)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.70 * ar[i - 1] + e[i]
    series["ar1_phi70_512"] = ar[500:]

    v = np.array([0.1, 0.1])
    for _ in range(1000):
        v = np.array([1 - 1.4 * v[0] ** 2 + v[1], 0.3 * v[0]])
    h = np.empty(512)
    for i in range(512):
        v = np.array([1 - 1.4 * v[0] ** 2 + v[1], 0.3 * v[0]])
        h[i] = v[0]
    series["henon_512"] = h

    # The EMD pair differs ONLY in whether an exact 0.0 is present. Everything
    # else -- length, spectrum, amplitude -- is identical, so any behavioural
    # difference between them is attributable to the zero and nothing else.
    tt = np.arange(512)
    base = np.sin(2 * np.pi * tt / 32) + 0.3 * np.sin(2 * np.pi * tt / 7)
    with_zero = base.copy()
    with_zero[0] = 0.0
    no_zero = base.copy()
    no_zero[0] = 1e-9          # same series, no exact zero
    series["emd_with_zero_512"] = with_zero
    series["emd_no_zero_512"] = no_zero

    series["lorenz_2048"] = lorenz(2048)

    for name, y in series.items():
        path = os.path.join(OUT, name + ".csv")
        np.savetxt(path, np.asarray(y, dtype=float), fmt="%.17g", delimiter=",")
        print(f"  wrote {name}.csv  n={len(y)}")


if __name__ == "__main__":
    main()

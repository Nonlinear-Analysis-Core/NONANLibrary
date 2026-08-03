"""Shared helpers for the NONAN Python test suite.

Design note -- why fixtures on disk instead of generating signals in each
language:

MATLAB and NumPy cannot draw the same random numbers. Any cross-language test
that generates its own series is comparing two different realisations and can
only ever check loose statistical agreement, which is far too weak to catch a
port that has drifted. So the reference series are written to CSV once and BOTH
languages read the identical samples. That isolates estimator disagreement from
RNG disagreement, and lets the Python tests assert agreement to ~1e-12 rather
than "close enough".
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")
PYDIR = os.path.join(REPO, "python")


def fixture(name):
    """Load a reference series by name, e.g. fixture('fgn_H85')."""
    path = os.path.join(FIXTURES, name + ".csv")
    return np.loadtxt(path, delimiter=",")


def matlab_reference():
    """Values produced by the MATLAB implementations on the same fixtures.

    Regenerate with:  matlab -batch "addpath('tests/matlab'); make_reference"
    Committed so the Python suite runs in CI without a MATLAB licence.
    """
    with open(os.path.join(FIXTURES, "matlab_reference.json")) as fh:
        return json.load(fh)


def pearson(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def spectral_error(x, z):
    """Relative error between the power spectra of x and z."""
    p = lambda v: np.abs(np.fft.fft(np.asarray(v, float) - np.mean(v))) ** 2
    p0 = p(x)
    return float(np.linalg.norm(p(z) - p0) / np.linalg.norm(p0))

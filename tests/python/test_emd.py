"""Contract tests for python/emd.py.

EMD's contract is short and completely checkable:

  1. It DECOMPOSES. The first IMF is not the input signal.
  2. It RECONSTRUCTS. sum(imfs) == input, to machine precision.
  3. Each IMF is oscillatory -- roughly equal numbers of extrema and zero
     crossings, mean near zero.
  4. It either succeeds or raises. It must never return a wrong answer quietly.

(4) is the one that matters here. emd.py currently violates it: on any series
containing an exact 0.0 it returns the INPUT UNCHANGED, with no warning, no
traceback, and a plausible-looking array shape. A caller cannot tell the
difference between "decomposed" and "did nothing" from the return value alone.
That is worse than a crash, because a crash gets noticed.

There is no emd.m in the MATLAB library, so there is nothing to cross-check
against inside this repo -- the Python EMD is load-bearing and unvalidated.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import nonantest  # noqa: E402

sys.path.insert(0, nonantest.PYDIR)


def run_emd_raw(x):
    import emd as emd_mod
    return np.asarray(emd_mod.emd(np.asarray(x, dtype=float)))


def run_emd(x):
    """Modes as a 2-D (n_modes, n_samples) array.

    atleast_2d is needed only because the failing path returns a 1-D array --
    see TestEmdReturnShape. Normalising here keeps the other tests measuring
    the numerics rather than tripping over the shape first.
    """
    return np.atleast_2d(run_emd_raw(x))


class TestEmdDecomposes(unittest.TestCase):

    def test_returns_something_other_than_the_input(self):
        """The defining property. IMF 1 must not BE the signal."""
        for name in ("emd_no_zero_512", "emd_with_zero_512"):
            with self.subTest(series=name):
                x = nonantest.fixture(name)
                imfs = run_emd(x)
                residual = float(np.max(np.abs(imfs[0] - x)))
                self.assertGreater(
                    residual, 1e-12,
                    f"[{name}] emd returned IMF 1 identical to the input "
                    f"(max|IMF1 - x| = {residual}). No decomposition happened, "
                    f"and nothing was raised or warned. Any downstream "
                    f"correlation against IMF 1 is corr(x, .) in disguise.")

    def test_reconstructs_the_input(self):
        for name in ("emd_no_zero_512", "emd_with_zero_512"):
            with self.subTest(series=name):
                x = nonantest.fixture(name)
                imfs = run_emd(x)
                err = float(np.max(np.abs(imfs.sum(axis=0) - x)))
                self.assertLess(err, 1e-9,
                                f"[{name}] modes do not sum back to the input "
                                f"(max error {err:.3e})")

    def test_produces_more_than_one_mode(self):
        for name in ("emd_no_zero_512", "emd_with_zero_512"):
            with self.subTest(series=name):
                x = nonantest.fixture(name)
                imfs = run_emd(x)
                self.assertGreater(
                    imfs.shape[0], 1,
                    f"[{name}] emd returned {imfs.shape[0]} mode(s). This "
                    f"series is a sum of two sinusoids and must yield several.")


class TestEmdReturnShape(unittest.TestCase):
    """The return RANK must not depend on the data.

    Measured: emd returns (2, 512) on the clean series and (512,) on the same
    series with one sample set to exactly 0.0. A caller writing the obvious
    `for mode in imfs:` iterates 2 modes in one case and 512 scalars in the
    other. Nothing raises. This is a separate harm from the wrong values --
    it breaks the function's type contract, not just its numerics.
    """

    def test_always_returns_two_dimensional_modes(self):
        for name in ("emd_no_zero_512", "emd_with_zero_512"):
            with self.subTest(series=name):
                r = run_emd_raw(nonantest.fixture(name))
                self.assertEqual(
                    r.ndim, 2,
                    f"[{name}] emd returned a {r.ndim}-D array of shape "
                    f"{r.shape}; the documented return is (n_modes, n_samples).")


class TestEmdExactZeroRegression(unittest.TestCase):
    """Isolates the trigger to a single sample.

    The two fixtures are the same series except that one has x[0] = 0.0 and the
    other x[0] = 1e-9. If behaviour differs between them, the exact zero is the
    cause -- there is no other difference to blame.
    """

    def test_an_exact_zero_does_not_change_the_outcome(self):
        z = nonantest.fixture("emd_with_zero_512")
        nz = nonantest.fixture("emd_no_zero_512")
        self.assertEqual(np.count_nonzero(z == 0.0), 1)
        self.assertEqual(np.count_nonzero(nz == 0.0), 0)

        n_z = run_emd(z).shape[0]
        n_nz = run_emd(nz).shape[0]
        self.assertEqual(
            n_z, n_nz,
            f"changing one sample from 1e-9 to exactly 0.0 changed the mode "
            f"count from {n_nz} to {n_z}. extr() does "
            f"`iz = np.where(x==0)` (a tuple) then `any(np.diff(iz)==1)`, "
            f"which raises; the bare `except:` in stop_sifting swallows it and "
            f"sets stop=1, accepting the unsifted signal as IMF 1.")

    def test_extr_handles_a_series_containing_zero(self):
        """Direct test of the failing helper, so the diagnosis is not inferred."""
        import emd as emd_mod
        x = nonantest.fixture("emd_with_zero_512")
        try:
            emd_mod.extr(x, nargout=3)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"extr() raised {type(exc).__name__} on a series "
                      f"containing an exact 0.0: {exc}")

    def test_zeros_are_common_in_real_data(self):
        """Blast radius: how conditional is this?

        Exact zeros are not exotic. Differenced integer-valued data, quantised
        sensor output, centred data, rectified signals and anything with a
        clipped baseline all produce them routinely.
        """
        rng = np.random.default_rng(1)
        hits = 0
        trials = 200
        for _ in range(trials):
            raw = np.round(rng.standard_normal(500) * 4)   # quantised sensor
            if np.any(np.diff(raw) == 0.0):
                hits += 1
        rate = hits / trials
        self.assertGreater(rate, 0.5)
        print(f"\n    [blast radius] quantised series containing an exact "
              f"repeat (-> 0.0 after differencing): {rate:.0%}")


class TestEmdDoesNotSwallowErrors(unittest.TestCase):

    def test_no_bare_except_in_the_sifting_path(self):
        """A bare `except:` is what converts a bug into a wrong answer.

        It catches KeyboardInterrupt and SystemExit too, and here it discards
        the traceback that would have identified the real fault in one minute.
        """
        src = open(os.path.join(nonantest.PYDIR, "emd.py")).read().splitlines()
        bare = [(i + 1, ln.strip()) for i, ln in enumerate(src)
                if ln.strip() == "except:"]
        self.assertEqual(
            bare, [],
            "bare `except:` in emd.py at lines "
            + ", ".join(str(n) for n, _ in bare)
            + ". Catch the specific exception, or at minimum re-raise after "
              "logging. As written, an exception in the sifting loop is "
              "reported to the caller as a successful decomposition.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Cross-language equivalence: python/ must agree with matlab/.

The Python directory is a PORT. Two implementations of one documented method
are two chances to be wrong, and a port that has drifted is the worst case
because both sides look maintained and neither is obviously the reference.

Method: both languages read the SAME samples from tests/fixtures/*.csv, and
MATLAB's answers are frozen in matlab_reference.json (regenerated only by
`matlab -batch "addpath('tests/matlab'); make_reference"`). Comparing
estimators over identical input isolates estimator disagreement from RNG
disagreement -- MATLAB and NumPy cannot draw the same random numbers, so any
test that generates its own series in each language can only check loose
statistical agreement and would miss a real divergence.

Deterministic estimators are compared VALUE BY VALUE to ~1e-10.
Stochastic ones (surrogates) cannot be, and are compared by CONTRACT and by
source-level correspondence instead.
"""

import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nonantest  # noqa: E402

sys.path.insert(0, nonantest.PYDIR)

REF = nonantest.matlab_reference()


class TestDfaPortMatchesMatlab(unittest.TestCase):

    def test_alpha_matches_to_ten_decimals(self):
        import dfa as dfa_mod
        for name, expect in REF["dfa"].items():
            with self.subTest(series=name):
                y = nonantest.fixture(name)
                scales = np.asarray(expect["scales"], dtype=int)
                out = dfa_mod.dfa(y, scales, 1, False)
                alpha = float(np.asarray(out[2]).squeeze())
                self.assertAlmostEqual(
                    alpha, expect["alpha"], places=10,
                    msg=(f"[{name}] dfa.py alpha {alpha!r} vs dfa.m "
                         f"{expect['alpha']!r} on identical samples "
                         f"(difference {abs(alpha - expect['alpha']):.3e})"))

    def test_fluctuation_curve_matches(self):
        """The alpha can agree while the curve behind it does not."""
        import dfa as dfa_mod
        for name, expect in REF["dfa"].items():
            with self.subTest(series=name):
                y = nonantest.fixture(name)
                scales = np.asarray(expect["scales"], dtype=int)
                out = dfa_mod.dfa(y, scales, 1, False)
                got = np.asarray(out[1], dtype=float).ravel()
                want = np.asarray(expect["fluctuation"], dtype=float).ravel()
                self.assertEqual(got.size, want.size,
                                 f"[{name}] {got.size} scales vs {want.size}")
                rel = np.max(np.abs(got - want) / np.abs(want))
                self.assertLess(rel, 1e-10,
                                f"[{name}] fluctuation curves differ by "
                                f"{rel:.3e} relative")


class TestEntSampPortMatchesMatlab(unittest.TestCase):
    """Two confirmed divergences, and they partially cancel.

    Measured on ar1_phi70_512 with m=2, r=0.2:
        Ent_Samp.py  1.7957514904975924
        Ent_Samp.m   1.795557910495278      difference 1.94e-4

    Cause 1 -- the radius. MATLAB's std() uses the N-1 denominator; NumPy's
    np.std() defaults to N (ddof=0). So r differs by sqrt(N/(N-1)) = 1.00098
    and the two implementations are not counting matches at the same radius.

    Cause 2 -- the template count. Ent_Samp.m builds N-m+1 templates of length
    m for B but N-m templates of length m+1 for A. Ent_Samp.py uses N-m for
    both. Richman & Moorman (2000), the reference both files cite, use N-m for
    both precisely so that A and B are comparable; the MATLAB B count is the
    nonstandard one.

    They pull in opposite directions: correcting ONLY the ddof widens the gap
    from 1.94e-4 to 6.36e-4 (measured). So the current near-agreement is
    accidental, and on another series or another m it will not hold. Fixing
    either side alone makes things worse -- both must be settled together, and
    which convention NONAN adopts is a maintainer's decision, not a test's.
    """

    def test_sample_entropy_matches(self):
        import Ent_Samp as es
        y = nonantest.fixture("ar1_phi70_512")
        want = REF["ent_samp"]["ar1_phi70_512"]
        got = float(np.asarray(es.Ent_Samp(y, 2, 0.2)).squeeze())
        self.assertAlmostEqual(
            got, want, places=10,
            msg=(f"Ent_Samp.py {got!r} vs Ent_Samp.m {want!r} "
                 f"(difference {abs(got - want):.3e}). See the class docstring: "
                 f"np.std ddof, and N-m+1 vs N-m templates."))


class TestSurrogatePortsCorrespond(unittest.TestCase):
    """Surrogates are stochastic, so compare structure, not values.

    These are source-level assertions. They are unavoidably a bit literal, but
    the alternative -- eyeballing two files in two languages -- is exactly the
    process that let the differences below survive.
    """

    @staticmethod
    def _src(rel):
        base = nonantest.PYDIR if rel.endswith(".py") else os.path.join(
            os.path.dirname(nonantest.PYDIR), "matlab")
        return open(os.path.join(base, rel)).read()

    def test_findrho_search_bounds_agree(self):
        m = self._src("Surr_findrho.m")
        p = self._src("Surr_findrho.py")
        m_hi = re.search(r"rhoH\s*=\s*([0-9.]+)", m).group(1)
        p_hi = re.search(r"rhoH\s*=\s*([0-9.]+)", p).group(1)
        self.assertEqual(
            float(m_hi), float(p_hi),
            f"Surr_findrho searches [rhoL, {m_hi}] in MATLAB but "
            f"[rhoL, {p_hi}] in Python. The two ports search different "
            f"intervals, so they answer different questions and cannot be "
            f"expected to return comparable rho.")

    def test_findrho_run_length_criterion_agrees(self):
        """The objective function itself differs between the ports."""
        m = self._src("Surr_findrho.m")
        p = self._src("Surr_findrho.py")
        self.assertIn("diff(yi)~=1", m.replace(" ", ""))
        self.assertIn(
            "diff(yi,axis=0)==1", p.replace(" ", ""),
            "port structure changed; re-check this comparison by hand")
        self.fail(
            "findrho_di counts runs using `diff(yi) ~= 1` in MATLAB and "
            "`diff(yi, axis=0) == 1` in Python. Negated condition: MATLAB "
            "locates the BREAKS between consecutive runs, Python locates the "
            "CONTINUATIONS. The two maximise different objectives, so the "
            "'optimal' rho they return is not the same quantity.")

    def test_theiler_function_name_matches_the_file(self):
        p = self._src("Surr_Theiler.py")
        names = re.findall(r"^def\s+(\w+)", p, re.M)
        self.assertIn(
            "Surr_Theiler", names,
            f"python/Surr_Theiler.py defines {names[0]!r}, not "
            f"'Surr_Theiler'. `import Surr_Theiler; Surr_Theiler.Surr_Theiler"
            f"(...)` -- the call the MATLAB user expects -- raises "
            f"AttributeError. The dated name also pins the file to a revision "
            f"that no longer means anything.")


class TestPortCoverage(unittest.TestCase):

    def test_report_which_functions_exist_in_only_one_language(self):
        """Not an assertion -- an inventory, printed for the audit report."""
        mdir = os.path.join(os.path.dirname(nonantest.PYDIR), "matlab")
        mats = {os.path.splitext(f)[0] for f in os.listdir(mdir)
                if f.endswith(".m")}
        pys = {os.path.splitext(f)[0] for f in os.listdir(nonantest.PYDIR)
               if f.endswith(".py")}
        print(f"\n    MATLAB only ({len(mats - pys)}): "
              f"{', '.join(sorted(mats - pys))}")
        print(f"    Python only ({len(pys - mats)}): "
              f"{', '.join(sorted(pys - mats))}")
        print(f"    both ({len(mats & pys)}): {', '.join(sorted(mats & pys))}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

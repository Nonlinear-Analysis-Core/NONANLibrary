# NONAN test suite

Headless. No GUI, no toolboxes beyond base MATLAB, no pytest.

```bash
# MATLAB  (exit 0 = all passed)
matlab -batch "addpath('tests/matlab'); run_tests"
matlab -batch "addpath('tests/matlab'); run_tests('Surr')"   # name filter

# Python
python3 tests/python/run_tests.py
python3 tests/python/run_tests.py emd                        # name filter
```

JUnit XML lands in `tests/artifacts/results.xml`.

## The four kinds of test here

**Contract.** Write down what a function *claims* to preserve or return, then
assert it. `nonantest.surrogateContract` measures whether the spectrum, the
distribution, and the variance survived a surrogate generator; the caller
decides which of those the algorithm actually promised. Getting that
distinction right matters more than the measurement: Algorithm 1 owes you an
exact spectrum, Algorithm 2 owes you an exact distribution and only an
*approximate* spectrum. Holding AAFT to exactness would be filing its design as
a bug.

**Known-answer recovery.** Feed a signal whose answer is known analytically and
check the estimator returns it — white noise → DFA α = 0.5, Brownian → 1.5,
fGn at H → α = H. The generators in `nonantest.signals` are written from
scratch and are deliberately independent of the library: if `fgn_sim` were used
to test `dfa`, a matched pair of errors would cancel and the test would pass.

**Structural.** `testHeadless` scans the shipped source for things that make
the library unusable in batch — `dbstop`, `waitbar`, file/function name
mismatches, CR-only line endings. These are invisible to any test that calls
the function inside `try/catch`, which is why they need their own pass.

**Cross-language.** Both languages read the *same samples* from
`tests/fixtures/*.csv`, and MATLAB's answers are frozen in
`matlab_reference.json`. MATLAB and NumPy cannot draw the same random numbers,
so a test that generates its own series in each language can only check loose
statistical agreement and would miss a real divergence. Identical input makes
1e-10 a meaningful assertion.

## Rules the harness follows

- **Base MATLAB only.** `corr` is Statistics Toolbox, so the suite uses
  `nonantest.pearson`. A test needing a toolbox must skip, not error.
- **No figures.** `nonantest.sideEffects` counts figures opened and fails the
  test if any survive.
- **`dbclear all` before and after every test.** Ten shipped functions execute
  `dbstop if error`, which is global session state. Under `matlab -batch` an
  uncaught error then hangs the process forever instead of failing it — this
  was measured, not assumed. The runner cannot let that leak between tests.
- **Fixtures and references are inputs, never outputs.** `make_fixtures.py` and
  `make_reference.m` are run by hand. A suite that can rewrite its own expected
  values will happily ratify a regression.

## Two traps that bit this harness during construction

Recorded because both produced *passing* tests over real defects:

- **MATLAB `regexp` has no `\b`.** It means backspace. `'^\s*dbstop\b'` matches
  nothing, so the `dbstop` scan passed while ten violations sat in the tree.
  MATLAB spells word boundaries `\<` and `\>`. `testGrepItselfWorks` now
  asserts the scanner can find something known-present, so a broken pattern
  shows up as a failure rather than a clean bill of health.
- **`strsplit` collapses consecutive delimiters by default**, silently dropping
  blank lines and shifting every reported line number. Pass
  `'CollapseDelimiters', false`.

Also: `functiontests(localfunctions)` collects every local function whose name
*ends* with "test", so a helper called `localSurrogateTest` is loaded as a test
case, fails to accept one argument, and takes the **whole file** out of the
suite with no warning. Helpers here are named `local*` and never `*Test`.

## Adding a function

1. Add a known-answer signal to `nonantest.signals` if the estimator has an
   analytic answer.
2. Write down the contract in the test file's header comment before writing any
   assertion. If you cannot state what the function promises, that is the
   finding.
3. Add a `nonantest.sideEffects` check — errors, figures, `dbstop`, runtime.
4. If the function exists in both languages, add a fixture and a line to
   `make_reference.m`.

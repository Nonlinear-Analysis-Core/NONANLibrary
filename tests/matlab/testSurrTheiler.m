function tests = testSurrTheiler
%TESTSURRTHEILER Contract tests for Surr_Theiler.
%
%   Each surrogate algorithm exists to hold one property of the data FIXED
%   while destroying everything else. The test is therefore not "does it run"
%   but "did the property it promised survive". The promises differ per
%   algorithm and are asserted separately -- see nonantest.surrogateContract.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
tc.TestData.series = struct( ...
    'fgn',    nonantest.signals('fgn',    512, 0.85), ...
    'ar1',    nonantest.signals('ar1',    512, 0.70), ...
    'henon',  nonantest.signals('henon',  512), ...
    'skewed', nonantest.signals('skewed', 512));
% Mildly skewed AND autocorrelated: the regime AAFT exists for.
tc.TestData.series.aaft = exp(0.2 * nonantest.signals('fgn', 512, 0.85));
end

function teardown(~)
dbclear all
end

% ------------------------------------------------------------------
% Algorithm 1 -- the Fourier-transform surrogate.
% CONTRACT: |FFT(z)| == |FFT(x)| exactly, every realisation. This is the
% definition of an FT surrogate, not a quality target. Variance is a function
% of the spectrum, so std(z)/std(x) == 1 follows and is asserted with it.
% ------------------------------------------------------------------
function testAlg1PreservesSpectrumExactly(tc)
names = fieldnames(tc.TestData.series);
for i = 1:numel(names)
    x = tc.TestData.series.(names{i});
    r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, 1), 19);

    tc.verifyLessThan(r.spectral_error, 1e-10, sprintf( ...
        ['[%s] Algorithm 1 must preserve the power spectrum EXACTLY.\n' ...
         '     measured spectral error %.4f (machine precision expected)\n' ...
         '     An FT surrogate is defined by |FFT| being held fixed. If it is\n' ...
         '     not, the surrogates differ from the data in a property the null\n' ...
         '     is supposed to control, and a rejection cannot be attributed to\n' ...
         '     nonlinearity.'], names{i}, r.spectral_error));

    tc.verifyLessThan(abs(r.sd_ratio - 1), 0.02, sprintf( ...
        ['[%s] Algorithm 1 must preserve variance (it follows from the\n' ...
         '     spectrum). measured sd ratio %.4f.\n' ...
         '     A ratio near 0.707 = 1/sqrt(2) is the specific signature of\n' ...
         '     real(ifft(.)) discarding the imaginary part because the\n' ...
         '     randomised phase spectrum was not conjugate symmetric.'], ...
        names{i}, r.sd_ratio));
end
end

function testAlg1PreservesAutocorrelation(tc)
% The linear null is "the data are a Gaussian linear process". Its whole
% content is the autocorrelation function, so the surrogate must reproduce it.
x = tc.TestData.series.ar1;
r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, 1), 39);
tc.verifyLessThan(abs(r.acf1_surrogate - r.acf1_original), 0.05, sprintf( ...
    ['Algorithm 1 changed the lag-1 autocorrelation: %.4f -> %.4f.\n' ...
     'The linear structure IS the null hypothesis; it must survive.'], ...
    r.acf1_original, r.acf1_surrogate));
end

% ------------------------------------------------------------------
% Algorithm 2 -- AAFT.
% CONTRACT: the value distribution is preserved EXACTLY (final rank remap).
% The spectrum is only APPROXIMATE, by design. Asserting exactness here would
% be reporting the documented, intended weakness of AAFT as a defect -- the
% weakness IAAFT exists to reduce. So the spectrum is bounded, not pinned.
% ------------------------------------------------------------------
function testAlg2PreservesDistributionExactly(tc)
names = fieldnames(tc.TestData.series);
for i = 1:numel(names)
    x = tc.TestData.series.(names{i});
    r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, 2), 19);
    tc.verifyLessThan(r.distribution_error, 1e-12, sprintf( ...
        ['[%s] AAFT must return an exact permutation of the input values.\n' ...
         '     measured distribution error %.3e'], names{i}, r.distribution_error));
end
end

function testAlg2SpectrumIsApproximateNotExact(tc)
% Documents the design boundary so nobody "fixes" AAFT toward 1e-16 and
% silently turns it into Algorithm 1.
%
% The series matters here. An IID lognormal has no spectral structure to
% preserve, so its measured "spectral error" is periodogram sampling noise
% (~0.88) for correct and incorrect implementations alike -- a test on that
% series measures nothing. exp(0.2 * fGn) is mildly skewed AND autocorrelated,
% which is the regime AAFT is designed for and where correct and incorrect
% implementations actually separate. Measured on this exact series and seed:
%     shipped implementation   0.712
%     conjugate-symmetric fix  0.226
% The 0.45 bound sits between them with room on both sides.
x = tc.TestData.series.aaft;
r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, 2), 39);
tc.verifyGreaterThan(r.spectral_error, 1e-6, ...
    'AAFT with an exact spectrum would not be AAFT -- check the algorithm switch.');
tc.verifyLessThan(r.spectral_error, 0.45, sprintf( ...
    ['AAFT spectral error %.3f. A correct AAFT manages ~0.23 on this series.\n' ...
     'NOTE: AAFT is not required to preserve the spectrum exactly -- the rank\n' ...
     'remap at the end necessarily disturbs it, and reducing that disturbance\n' ...
     'is the entire purpose of IAAFT. This test asserts a QUALITY bound, not\n' ...
     'the exactness contract that applies to Algorithm 1.'], r.spectral_error));
end

% ------------------------------------------------------------------
% Algorithm 0 -- shuffle. Distribution exact, spectrum destroyed BY DESIGN.
% ------------------------------------------------------------------
function testAlg0IsAnExactPermutation(tc)
x = tc.TestData.series.skewed;
r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, 0), 19);
tc.verifyLessThan(r.distribution_error, 1e-12, ...
    'Algorithm 0 must be a permutation of the input.');
tc.verifyLessThan(abs(r.acf1_surrogate), 0.15, ...
    'Algorithm 0 should destroy serial correlation -- that is its null.');
end

% ------------------------------------------------------------------
% Properties every algorithm owes the caller regardless of null.
% ------------------------------------------------------------------
function testAllAlgorithmsReturnRealSameLengthDistinct(tc)
x = tc.TestData.series.fgn;
for alg = [0 1 2]
    r = nonantest.surrogateContract(x, @(v) Surr_Theiler(v, alg), 5);
    tc.verifyTrue(r.length_ok,   sprintf('alg %d changed the series length', alg));
    tc.verifyFalse(r.any_complex, sprintf('alg %d returned a complex series', alg));
    tc.verifyTrue(r.distinct,    sprintf('alg %d returned identical surrogates', alg));
end
end

function testUnknownAlgorithmIsRejected(tc)
% Currently the switch has no otherwise branch, so alg 3 silently returns an
% undefined output rather than telling the caller they asked for nothing.
x = tc.TestData.series.fgn;
s = nonantest.sideEffects(@() Surr_Theiler(x, 3));
tc.verifyTrue(s.errored, ...
    'An unsupported algorithm number should raise, not return silently.');
end

% ------------------------------------------------------------------
% BLAST RADIUS. Does the defect actually change anyone's inference?
%
% Measured, not assumed. Two experiments, 19 surrogates, rank test, nominal
% two-sided alpha = 2/20 = 0.10, time-reversal asymmetry as the statistic
% (the standard nonlinear choice for this null):
%
%   Type I  on AR(1), where the linear null is TRUE:
%       shipped 0.110    conjugate-symmetric fix 0.105    nominal 0.10
%   Power   on Henon, where the null is FALSE:
%       shipped 1.000    conjugate-symmetric fix 1.000
%
% So for the canonical use of Algorithm 1 -- rank test, nonlinear statistic --
% the defect changes NEITHER size NOR power. This bounds the claim: it is a
% real correctness failure, but it is not grounds for saying that published
% surrogate tests built on it reached the wrong conclusion. Testing that
% honestly was the point of running the experiment.
%
% Where the damage IS unconditional is asserted separately, above:
% every Algorithm 1 surrogate has 0.708x the variance of the data, and a
% spectral error of ~0.6. That invalidates any use that reports the surrogate
% DISTRIBUTION rather than a rank -- z-scores against the surrogate mean,
% effect sizes, confidence bands, or any figure asserting "the spectrum is
% preserved". Those uses are not exercised here and remain untested.
% ------------------------------------------------------------------
function testTypeIErrorIsNotInflatedForNonlinearStatistic(tc)
[rate, ~] = localRunExperiment(@(v) Surr_Theiler(v, 1), 'ar1', 200);
fprintf('    [blast radius] Alg 1 Type I on AR(1), time-asymmetry: %.3f (nominal 0.10)\n', rate);
% Binomial 99% upper bound around 0.10 at nRep = 200 is ~0.155.
tc.verifyLessThan(rate, 0.155, sprintf( ...
    ['Type I error %.3f against a nominal 0.10 on AR(1) data, where the\n' ...
     'linear null is TRUE. Excess rejections here would mean the test is\n' ...
     'detecting the surrogate generator rather than nonlinearity.'], rate));
end

function testRetainsPowerAgainstDeterministicChaos(tc)
[rate, ~] = localRunExperiment(@(v) Surr_Theiler(v, 1), 'henon', 100);
fprintf('    [blast radius] Alg 1 power on Henon, time-asymmetry:  %.3f\n', rate);
tc.verifyGreaterThan(rate, 0.90, sprintf( ...
    'Power against the Henon map fell to %.3f; it should be ~1.', rate));
end

function [rate, stat0s] = localRunExperiment(gen, kind, nRep)
nSurr = 19;
n = 256;
reject = 0;
stat0s = zeros(nRep, 1);
for rep = 1:nRep
    if strcmp(kind, 'henon')
        x = nonantest.signals('henon', n) + 0.02*nonantest.signals('white', n, [], 9000+rep);
    else
        x = nonantest.signals(kind, n, 0.70, 5000 + rep);
    end
    stat0s(rep) = localTimeAsymmetry(x);
    stats = zeros(nSurr, 1);
    for k = 1:nSurr
        stats(k) = localTimeAsymmetry(gen(x));
    end
    % Random tie-breaking. A correct FT surrogate preserves every linear
    % property EXACTLY, so a linear statistic produces exact ties and a naive
    % `sum(stats < stat0)` scores them all as rejections. That artefact is not
    % a property of the data and must not be counted as one.
    tol = 1e-9 * max(1, abs(stat0s(rep)));
    below = sum(stats < stat0s(rep) - tol);
    tied  = sum(abs(stats - stat0s(rep)) <= tol);
    rank  = 1 + below + sum(rand(tied, 1) < 0.5);
    if rank == 1 || rank == nSurr + 1
        reject = reject + 1;
    end
end
rate = reject / nRep;
end

function s = localTimeAsymmetry(x)
% Time-reversal asymmetry. Zero in expectation for any linear Gaussian
% process, sensitive to nonlinearity. The standard statistic for this null.
%
% A LINEAR statistic (variance, lag-1 ACF, DFA alpha) is the wrong choice
% against an FT null and is not used here: a correct FT surrogate preserves
% those exactly by construction, so the test is degenerate whether or not the
% generator is buggy. Measured on the shipped code, a lag-1-ACF test has both
% Type I error 0.000 and power 0.000 -- it never rejects anything at all.
x = double(x(:));
x = (x - mean(x)) / std(x);
d = x(2:end) - x(1:end-1);
s = mean(d.^3);
end

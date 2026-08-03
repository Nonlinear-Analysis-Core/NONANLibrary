function tests = testAmi
%TESTAMI Contract and known-answer tests for ami / ami_histogram / ami_kde.
%
%   Mutual information has real analytic properties to test against, which is
%   more than most estimators in this library offer:
%
%     I(X;Y) >= 0                       always, with equality iff independent
%     I(X;Y) == I(Y;X)                  symmetry
%     I(X;X) == H(X)                    self-information equals entropy
%     sine of period P                  first AMI minimum at P/4 (quadrature)
%
%   These hold for the estimand. A finite-sample estimator satisfies them only
%   approximately, and the tolerances below say how approximately.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
% ami/ami_histogram/ami_kde arrive on a separate branch from this harness, so
% on a tree that has the tests but not yet the refactor these must SKIP rather
% than fail. A missing function is not a defect in the function.
for f = ["ami", "ami_histogram", "ami_kde"]
    tc.assumeTrue(exist(f, 'file') == 2, sprintf( ...
        '%s is not on the path; skipping (expected on branches without the AMI refactor).', f));
end
tc.TestData.lorenz = nonantest.signals('lorenz', 1200);
tc.TestData.white  = nonantest.signals('white', 2000);
end

function teardown(~)
dbclear all
end

% ------------------------------------------------------------------
% Analytic properties.
% ------------------------------------------------------------------
function testMutualInformationIsNonNegative(tc)
for algo = ["histogram", "kde"]
    [~, curve] = ami(tc.TestData.lorenz, 20, Algorithm=algo);
    tc.verifyGreaterThanOrEqual(min(curve(:,2)), -1e-9, sprintf( ...
        '%s produced a negative mutual information (%.4g).', algo, min(curve(:,2))));
end
end

function testSelfInformationEqualsEntropy(tc)
% At lag 0 the joint histogram is diagonal, so
%     I(X;X) = sum_i p_i log2(1/p_i) = H(X).
% Checks the joint/marginal bookkeeping, which is where histogram MI code
% usually goes wrong.
x = tc.TestData.lorenz;
[~, curve, info] = ami_histogram(x, 5);

edges = linspace(min(x), max(x), info.bins + 1);
p = histcounts(x(1:info.samplesPerLag), edges) / info.samplesPerLag;
p = p(p > 0);
H = -sum(p .* log2(p));

tc.verifyEqual(curve(1,2), H, 'AbsTol', 1e-10, sprintf( ...
    'I(X;X) = %.6f but H(X) = %.6f; they must be equal.', curve(1,2), H));
end

function testIndependentSeriesSitAtTheKnownBiasFloor(tc)
% White noise is independent at every nonzero lag, so the true AMI is 0. A
% plug-in histogram estimator does NOT return 0 -- it returns its own
% finite-sample bias, which for independent variables is approximately
%
%     E[I_hat] ~ (Bx - 1)(By - 1) / (2 N ln 2)   bits        (Miller & Madow)
%
% Testing against 0 would therefore be testing the wrong thing. Test instead
% that the measured floor is bounded by the analytic bias and shrinks with N.
% Measured: 0.220 / 0.169 / 0.117 / 0.078 bits at N = 500 / 2k / 8k / 32k,
% i.e. 0.61-0.85 of the predicted bias throughout.
prev = Inf;
for N = [500 2000 8000 32000]
    x = nonantest.signals('white', N);
    [~, curve, info] = ami_histogram(x, 30);
    floorBits = mean(curve(11:end, 2));
    predicted = (info.bins - 1)^2 / (2 * info.samplesPerLag * log(2));

    tc.verifyLessThan(floorBits, 1.2 * predicted, sprintf( ...
        ['N=%d: AMI floor on white noise is %.4f bits against a predicted\n' ...
         'bias of %.4f. Exceeding the analytic bias means the estimator is\n' ...
         'adding error of its own, not just plug-in bias.'], N, floorBits, predicted));
    tc.verifyLessThan(floorBits, prev, sprintf( ...
        'N=%d: the bias floor (%.4f) did not fall relative to the smaller N.', ...
        N, floorBits));
    prev = floorBits;
end
end

function testRecoversTheExactGaussianMutualInformation(tc)
% For a bivariate Gaussian with correlation rho,
%
%     I(X;Y) = -0.5 * log2(1 - rho^2)   bits
%
% exactly. An AR(1) process with parameter phi has rho(k) = phi^k, so every
% point of its AMI curve has a closed form. This is the only genuinely exact
% known answer available for mutual information, and it tests the VALUE, not
% just where the minimum sits.
%
% Measured at phi = 0.7, lag 1: exact 0.4857, kde 0.5411, histogram 0.5619.
% Both overshoot, which is the expected direction for a plug-in estimator.
phi = 0.7;
exact = @(k) -0.5 * log2(1 - phi^(2*k));

[~, ch] = ami_histogram(nonantest.signals('ar1', 20000, phi), 3);
[~, ck] = ami_kde(nonantest.signals('ar1', 3000, phi), 3);

for k = 1:2
    tc.verifyEqual(ch(k+1,2), exact(k), 'RelTol', 0.55, sprintf( ...
        'histogram AMI at lag %d = %.4f, exact %.4f', k, ch(k+1,2), exact(k)));
    tc.verifyEqual(ck(k+1,2), exact(k), 'RelTol', 0.30, sprintf( ...
        'kde AMI at lag %d = %.4f, exact %.4f', k, ck(k+1,2), exact(k)));
end
fprintf('    [known answer] AR(1) phi=0.7 lag 1: exact %.4f  kde %.4f  histogram %.4f\n', ...
    exact(1), ck(2,2), ch(2,2));
end

function testNoisySineMinimumIsNearQuarterPeriod(tc)
% For a noisy sinusoid the first AMI minimum sits near the quarter period --
% a standard heuristic, not an identity, so the tolerance is wide.
%
% The NOISELESS sine is deliberately not used. There, x(t+tau) is a
% deterministic (two-branch) function of x(t) at every lag, so the estimated
% AMI is limited only by bin resolution and the curve is a jagged plateau:
%
%   lags 0..14: 3.289 2.232 2.037 2.057 1.694 1.857 1.657 1.657 1.657 1.557 ...
%
% The "first minimum" of that is lag 2, which is a discretisation wiggle and
% not a property of the signal. A deterministic signal has no meaningful AMI
% minimum, which is worth knowing before anyone uses one as a test case.
P = 40;
x = nonantest.signals('sine', 2000, P) + 0.15 * nonantest.signals('white', 2000);
tau = ami_histogram(x, 60);
tc.verifyEqual(tau, P/4, 'AbsTol', 4, sprintf( ...
    'first AMI minimum of a noisy period-%d sine at lag %g, expected near %g.', ...
    P, tau, P/4));
end

% ------------------------------------------------------------------
% Equivalence with the code being replaced. Locks the refactor.
% ------------------------------------------------------------------
function testKdeMatchesOriginalAmiThomas(tc)
% ami_kde replaced n^2-row matrix stacking and a cumsum-difference trick with
% chunked implicit expansion and direct block sums. That is a pure
% performance change and must not move a digit that matters.
x = tc.TestData.lorenz;
L = 20;
[~, old] = AMI_Thomas(x, L);          % original indexes lags 1..L
[~, new] = ami_kde(x, L);             % new indexes lags 0..L
d = max(abs(old(:,2) - new(2:end,2)));
tc.verifyLessThan(d, 1e-9, sprintf( ...
    'ami_kde differs from AMI_Thomas by %.3e; the refactor changed the result.', d));
end

function testKdeIsFasterThanOriginal(tc)
x = nonantest.signals('lorenz', 1000);
L = 15;
t1 = tic; AMI_Thomas(x, L); tOld = toc(t1);
t2 = tic; ami_kde(x, L);    tNew = toc(t2);
fprintf('    [perf] ami_kde %.2fs vs AMI_Thomas %.2fs (%.1fx)\n', tNew, tOld, tOld/tNew);
tc.verifyLessThan(tNew, tOld, 'the refactor should not be slower');
end

function testChunkSizeDoesNotChangeTheAnswer(tc)
x = nonantest.signals('lorenz', 600);
[~, a] = ami_kde(x, 10, ChunkSize=64);
[~, b] = ami_kde(x, 10, ChunkSize=4096);
tc.verifyEqual(a, b, 'AbsTol', 1e-12, ...
    'ChunkSize is a memory control and must not affect the result.');
end

% ------------------------------------------------------------------
% Interface.
% ------------------------------------------------------------------
function testWrapperDispatchesAndAliases(tc)
x = tc.TestData.lorenz;
[~, ~, i1] = ami(x, 20);
[~, ~, i2] = ami(x, 20, Algorithm="stergiou");
[~, ~, i3] = ami(x, 20, Algorithm="thomas");
tc.verifyEqual(i1.estimator, "histogram");
tc.verifyEqual(i2.estimator, "histogram", '"stergiou" must alias "histogram"');
tc.verifyEqual(i3.estimator, "kde",       '"thomas" must alias "kde"');
end

function testUnknownAlgorithmIsRejected(tc)
tc.verifyError(@() ami(tc.TestData.lorenz, 10, Algorithm="wavelet"), ...
    'ami:unknownAlgorithm');
end

function testBinsArgumentIsAccepted(tc)
% The old AMI_Stergiou had this guard inverted -- `if numel(bins)==1, error`
% -- so the entire documented three-argument form raised. Every documented
% call must work.
x = tc.TestData.lorenz;
for b = [8 32 128]
    [~, ~, info] = ami(x, 20, Bins=b);
    tc.verifyEqual(info.bins, b, sprintf('Bins=%d was not honoured', b));
end
end

function testNanInputErrorsRatherThanCorrupts(tc)
% The original called nanstd, implying NaN tolerance, then failed downstream
% with "Index into matrix must be an integer". Fail early and say why.
x = tc.TestData.lorenz;
x(17) = NaN;
tc.verifyError(@() ami_histogram(x, 10), 'ami_histogram:nanInput');
tc.verifyError(@() ami_kde(x, 10),       'ami_kde:nanInput');
end

function testLagTooLargeIsRejected(tc)
tc.verifyError(@() ami_histogram(nonantest.signals('white', 50), 60), ...
    'ami_histogram:lagTooLarge');
end

function testTauIsAScalarLag(tc)
% AMI_Stergiou returned an N-by-2 matrix of every local minimum with the
% 20%-crossing lag appended as a final row of different meaning, while
% documenting "tau, first minimum". tau is now a scalar lag; the rest is in
% the third output.
[tau, curve, info] = ami(tc.TestData.lorenz, 30);
tc.verifyTrue(isscalar(tau), 'tau must be a scalar lag');
tc.verifyTrue(ismember(tau, curve(:,1)), 'tau must be one of the lags evaluated');
tc.verifyTrue(isfield(info, 'allMinima'), 'diagnostics belong in info, not tau');
end

% ------------------------------------------------------------------
% Portability.
% ------------------------------------------------------------------
function testRunsOnBaseMatlabOnly(tc)
% AMI_Stergiou needed range + nanstd; AMI_Thomas needed corr, mvnpdf and
% normpdf. All five are Statistics and Machine Learning Toolbox, for a bin
% count and three closed-form densities.
for f = ["ami.m", "ami_histogram.m", "ami_kde.m"]
    [~, products] = matlab.codetools.requiredFilesAndProducts(f);
    names = string({products.Name});
    extra = setdiff(names, "MATLAB");
    tc.verifyEmpty(extra, sprintf( ...
        '%s requires beyond base MATLAB: %s', f, strjoin(extra, ", ")));
end
end

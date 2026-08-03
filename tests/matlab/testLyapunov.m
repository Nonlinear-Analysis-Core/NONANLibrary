function tests = testLyapunov
%TESTLYAPUNOV Contract tests for the lyapunov wrapper.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
tc.TestData.lorenz = nonantest.signals('lorenz', 3000);
tc.TestData.fs = 1/0.03;
end

function teardown(~)
dbclear all
end

% ------------------------------------------------------------------
% Phase space as an input.
% ------------------------------------------------------------------
function testPhaseSpaceInputMatchesSeriesInput(tc)
% The point of accepting a phase space is that it is the same computation.
% Given the same reconstruction and the same Theiler window, both routes must
% agree exactly -- the wrapper must not silently re-embed or re-parameterise.
x = tc.TestData.lorenz;
fs = tc.TestData.fs;
tau = 10; dim = 5;

fromSeries = lyapunov(x, fs, Tau=tau, Dim=dim);
Y = psr(x, tau, dim);
fromSpace  = lyapunov(Y, fs, Tau=tau);

tc.verifyEqual(fromSpace, fromSeries, 'RelTol', 1e-12, sprintf( ...
    ['Passing a phase space gave %.6f against %.6f from the series it was\n' ...
     'built from. Same data, same embedding, same window: these must agree.'], ...
    fromSpace, fromSeries));
end

function testSuppliedPhaseSpaceIsUsedVerbatim(tc)
x = tc.TestData.lorenz;
Y = psr(x, 10, 5);
[~, extra] = lyapunov(Y, tc.TestData.fs, Tau=10);
tc.verifyEqual(extra.phaseSpace, Y, ...
    'the supplied phase space must be used unchanged');
tc.verifyEqual(extra.dim, 5);
end

function testWarnsWhenTheilerWindowCannotBeDerived(tc)
% A supplied phase space carries no record of the delay that built it, so the
% Theiler default cannot be derived. Guessing changes the answer -- 0.8022 at
% a window of 8 against 0.7833 at the default 1 on one Lorenz reconstruction
% -- so the wrapper must say so rather than pick silently.
Y = psr(tc.TestData.lorenz, 10, 5);
tc.verifyWarning(@() lyapunov(Y, tc.TestData.fs), ...
    'lyapunov:theilerFromDefaultTau');

% Silent once the caller has been explicit, either way.
tc.verifyWarningFree(@() lyapunov(Y, tc.TestData.fs, TheilerWindow=8));
tc.verifyWarningFree(@() lyapunov(Y, tc.TestData.fs, Tau=10));
end

function testBothMethodsCanShareOneReconstruction(tc)
Y = psr(tc.TestData.lorenz, 10, 5);
[~, er] = lyapunov(Y, tc.TestData.fs, Algorithm="rosenstein", Tau=10);
[~, ew] = lyapunov(Y, tc.TestData.fs, Algorithm="wolf", Tau=10);
tc.verifyEqual(er.phaseSpace, ew.phaseSpace, ...
    'both estimators must run on the identical phase space');
end

% ------------------------------------------------------------------
% Known answers. Both methods, one interface, nats throughout.
% ------------------------------------------------------------------
function testRecoversExactMapExponents(tc)
% lambda is a theorem for these, so both methods can be held to it.
cases = {'skewtent', 0.4; 'skewtent', 0.3; 'skewtent', 0.2; 'logistic', []};
for k = 1:size(cases,1)
    if isempty(cases{k,2})
        y = nonantest.signals(cases{k,1}, 4000);
        ref = nonantest.lambdaReference(cases{k,1});
    else
        y = nonantest.signals(cases{k,1}, 4000, cases{k,2});
        ref = nonantest.lambdaReference(cases{k,1}, cases{k,2});
    end
    for algo = ["rosenstein", "wolf"]
        got = lyapunov(y, 1, Algorithm=algo, Tau=1, Dim=3, Evolve=5);
        tc.verifyEqual(got, ref.nats, 'RelTol', 0.30, sprintf( ...
            '%s on %s: %.4f nats against an EXACT %.4f', ...
            algo, cases{k,1}, got, ref.nats));
    end
end
end

function testBothMethodsReturnNats(tc)
% lye_w natively returns bits; the wrapper must convert, so that the two
% methods are directly comparable and match published values.
y = nonantest.signals('logistic', 4000);
[lam, extra] = lyapunov(y, 1, Algorithm="wolf", Tau=1, Dim=3, Evolve=5);
tc.verifyEqual(lam, log(2), 'RelTol', 0.25, sprintf( ...
    'wolf via the wrapper gave %.4f; ln 2 = %.4f nats is exact here.', ...
    lam, log(2)));
tc.verifyEqual(lam, extra.bits*log(2), 'RelTol', 1e-12, ...
    'the reported nats must be the underlying bits times ln 2');
tc.verifyEqual(extra.units, "nats per unit time");
end

function testChaoticExceedsPeriodicThroughTheWrapper(tc)
chaotic  = lyapunov(nonantest.signals('skewtent', 3000, 0.3), 1, Tau=1, Dim=3);
periodic = lyapunov(nonantest.signals('sine', 3000, 50), 1, Tau=12, Dim=3);
tc.verifyGreaterThan(chaotic, periodic, sprintf( ...
    'chaotic %.4f did not exceed periodic %.4f', chaotic, periodic));
end

% ------------------------------------------------------------------
% Diagnostics and interface.
% ------------------------------------------------------------------
function testScalingRegionIsReturnedForInspection(tc)
% The fitted window dominates a Rosenstein estimate, so it must be auditable
% rather than hidden.
[~, extra] = lyapunov(tc.TestData.lorenz, tc.TestData.fs, Tau=10, Dim=5);
tc.verifyTrue(isfield(extra, 'scalingRegion'));
tc.verifyNotEmpty(extra.scalingRegion);
tc.verifyTrue(isfield(extra, 'divergence'));
tc.verifyGreaterThan(extra.fitR2, 0.9, sprintf( ...
    'the fitted scaling region has R2 = %.4f; that is not a linear region', ...
    extra.fitR2));
tc.verifyLessThanOrEqual(max(extra.scalingRegion), numel(extra.divergence));
end

function testAlgorithmAliases(tc)
x = tc.TestData.lorenz; fs = tc.TestData.fs;
a = lyapunov(x, fs, Algorithm="rosenstein", Tau=10, Dim=5);
b = lyapunov(x, fs, Algorithm="r", Tau=10, Dim=5);
tc.verifyEqual(b, a, 'RelTol', 1e-12, '"r" must alias "rosenstein"');
[~, ew] = lyapunov(x, fs, Algorithm="w", Tau=10, Dim=5);
tc.verifyEqual(ew.estimator, "wolf", '"w" must alias "wolf"');
end

function testInvalidInputsAreRejected(tc)
x = tc.TestData.lorenz;
tc.verifyError(@() lyapunov(x, tc.TestData.fs, Algorithm="wavelet"), ...
    'lyapunov:unknownAlgorithm');
tc.verifyError(@() lyapunov([1;2;3], 1, Dim=9, Tau=5), ...
    'lyapunov:tooShort');
xn = x; xn(5) = NaN;
tc.verifyError(@() lyapunov(xn, tc.TestData.fs), 'lyapunov:nanInput');
end

function testRunsHeadless(tc)
s = nonantest.sideEffects(@() lyapunov(tc.TestData.lorenz, tc.TestData.fs, ...
                                       Tau=10, Dim=5));
tc.verifyFalse(s.errored, 'lyapunov errored');
tc.verifyFalse(s.dbstop, 'lyapunov armed the debugger');
tc.verifyEqual(s.figures, 0, 'lyapunov opened a figure');
end

function tests = testDfa
%TESTDFA Known-answer recovery tests for dfa.
%
%   DFA has exact analytic answers on standard processes. Feed a series whose
%   scaling exponent is known by construction and check the estimator returns
%   it. This is the only kind of test that can distinguish "runs without
%   error" from "computes the right number".
%
%     white noise            alpha = 0.5
%     Brownian motion        alpha = 1.5
%     fGn at H               alpha = H
%
%   Tolerances are set from the estimator's own sampling variability at these
%   lengths, not chosen to make the test pass.
tests = functiontests(localfunctions);
end

function teardown(~)
dbclear all
end

function testWhiteNoiseGivesHalf(tc)
x = nonantest.signals('white', 4096);
a = localAlpha(x);
tc.verifyEqual(a, 0.5, 'AbsTol', 0.05, sprintf( ...
    'DFA on white noise returned alpha = %.4f, expected 0.5.', a));
end

function testBrownianGivesThreeHalves(tc)
x = nonantest.signals('brown', 4096);
a = localAlpha(x);
tc.verifyEqual(a, 1.5, 'AbsTol', 0.07, sprintf( ...
    'DFA on Brownian motion returned alpha = %.4f, expected 1.5.', a));
end

function testTracksHacrossFgnLadder(tc)
% The ladder matters more than any single point: an estimator can be biased
% and still monotone, or unbiased at one H and broken elsewhere.
Hs = [0.3 0.5 0.7 0.9];
got = zeros(size(Hs));
for k = 1:numel(Hs)
    got(k) = localAlpha(nonantest.signals('fgn', 4096, Hs(k)));
end
fprintf('    [known answer] fGn ladder  H = %s\n                   alpha = %s\n', ...
    mat2str(Hs), mat2str(round(got, 3)));
for k = 1:numel(Hs)
    tc.verifyEqual(got(k), Hs(k), 'AbsTol', 0.08, sprintf( ...
        'fGn H = %.2f -> DFA alpha = %.4f', Hs(k), got(k)));
end
tc.verifyTrue(all(diff(got) > 0), ...
    'DFA alpha must increase monotonically with H.');
end

function testIsScaleAndOffsetInvariant(tc)
% DFA removes the mean and a local trend, so a constant offset and a positive
% scale factor cannot change alpha. A failure here means the detrending or
% the normalisation is wrong.
x = nonantest.signals('fgn', 2048, 0.75);
a1 = localAlpha(x);
a2 = localAlpha(3.7 * x + 100);
tc.verifyEqual(a2, a1, 'AbsTol', 1e-8, sprintf( ...
    'alpha changed under an affine rescaling: %.10f -> %.10f', a1, a2));
end

function testAcceptsRowAndColumnVectors(tc)
x = nonantest.signals('fgn', 2048, 0.75);
tc.verifyEqual(localAlpha(x(:)'), localAlpha(x(:)), 'AbsTol', 1e-10, ...
    'dfa gave different answers for a row vector and a column vector.');
end

function testRunsHeadlessWithoutOpeningFigures(tc)
x = nonantest.signals('white', 1024);
sc = localScales(1024);
s = nonantest.sideEffects(@() dfa(x, sc, 1, false));
tc.verifyFalse(s.errored, 'dfa errored with plotting disabled.');
tc.verifyEqual(s.figures, 0, sprintf( ...
    'dfa opened %d figure(s) with the plot flag false.', s.figures));
end

function testPlotFlagDefaultsToNoPlot(tc)
% Calling with three arguments must not open a window. The docstring says the
% plot flag "default = true", which is the wrong default for a library
% function: a batch job that omits the argument gets a figure it cannot close.
x = nonantest.signals('white', 1024);
sc = localScales(1024);
s = nonantest.sideEffects(@() dfa(x, sc, 1));
tc.verifyFalse(s.errored, sprintf('3-argument call to dfa errored: %s', ...
    localMsg(s)));
tc.verifyEqual(s.figures, 0, ...
    'dfa opened a figure when the plot argument was omitted.');
end

% ------------------------------------------------------------------ helpers

function a = localAlpha(x)
n = numel(x);
[~, ~, a] = dfa(x, localScales(n), 1, false);
end

function sc = localScales(n)
% Decade-spaced scales from 16 to n/8. Below ~10 the polynomial fit has too
% few points per window; above n/8 there are too few windows to average.
sc = unique(round(logspace(log10(16), log10(floor(n/8)), 18)));
end

function m = localMsg(s)
if isempty(s.err), m = ''; else, m = s.err.message; end
end

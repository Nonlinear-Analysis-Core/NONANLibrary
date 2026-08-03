function tests = testLyE
%TESTLYE Contract and known-answer tests for LyE_R and LyE_W.
%
%   BENCHMARK DESIGN. Lyapunov estimators are tested here in three layers,
%   weakest evidence last, because no single reference is sufficient.
%
%   Layer 1 -- INVARIANCES. Properties that hold for every series with no
%   reference value at all: lambda is unchanged by rescaling the signal,
%   chaotic beats periodic, the output shape does not depend on the data.
%   These cannot be wrong about a benchmark because they use none, so they
%   carry the most weight. Scale invariance is what caught the 1e5 defect.
%
%   Layer 2 -- EXACT lambda, from maps where it is a THEOREM. The skew tent
%   map has lambda = -p ln p - (1-p) ln(1-p) exactly, and varying p gives a
%   LADDER of exact values, which detects proportional bias in a way a single
%   number cannot. No integrator, step size or initial condition enters.
%
%   Layer 3 -- NUMERICAL reference for flows, from Sprott (2003) Appendix A
%   (62 systems catalogued). Legitimate provided the standard control
%   parameters are used and the value is labelled for what it is. It is not
%   circular: Sprott's table includes cases with analytic answers and
%   reproduces them -- his logistic entry is 0.693147181 against
%   ln 2 = 0.6931471806, nine significant figures. A method that recovers the
%   exact answer where one exists is evidence about the method.
%
%   WHAT NONE OF THESE ARE. All three describe the SYSTEM. A finite-sample
%   delay-embedding estimator has a different estimand, depending on sampling
%   rate, tau, dim, N and scaling region. Measured: LyE_W on Lorenz spans
%   ratio 0.69 to 1.43 across ordinary parameter choices, reaching 1.01 at the
%   best of them. Layer-3 tolerances are therefore loose and parameter
%   conditional; a tight assertion there tests the parameters, not the code.
%
%   UNITS. LyE_W accumulates log2 and returns BITS per unit time, faithful to
%   Wolf (1985). Published values are in NATS. The ratio is ln 2 = 0.693, so
%   comparing LyE_W directly against 0.9056 makes a correct answer look like a
%   44% overshoot. Neither the docstring nor the output name says which.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
tc.TestData.lorenz = nonantest.signals('lorenz', 2000);
tc.TestData.sine   = nonantest.signals('sine', 2000, 50);
end

function teardown(~)
dbclear all
end

% =================================================================
% LAYER 1 -- invariances. No reference value required.
% =================================================================

% THE DEFECT. lambda is invariant under x -> c*x: every distance scales by c,
% and ln(c*d) = ln c + ln d leaves the slope alone. LyE_R is not invariant,
% because it excludes temporally-near neighbours with a hard-coded sentinel:
%
%     Ydisti(range_exclude) = 1e5;                        % LyE_R.m
%
% 1e5 is an absolute constant compared against data-scaled distances. Once
% real distances exceed it the sentinel stops being large, the "excluded"
% points become the MINIMUM, and every point pairs with its own immediate
% temporal neighbour -- exactly what the exclusion existed to prevent.
%
% Measured on Lorenz, fitting samples 5-100:
%     scale 1e3   slope 0.9872
%     scale 1e5   slope 0.9795
%     scale 3e5   slope 0.7097
%     scale 1e6   slope 0.0831      <-- 92% collapse
% and at scale 1e6 the median |i-j| of matched pairs falls from 1026 to 4,
% with ~90% of pairs inside the exclusion window.
%
% It fails silently and in the dangerous direction: a spuriously small
% exponent reads as "more stable", which in gait and balance work is a
% substantive claim rather than an obvious error.
function testLyE_R_IsScaleInvariant(tc)
x = tc.TestData.lorenz;
Fs = 1/0.03;
base = localRosensteinSlope(x, Fs);
for c = [1e1 1e3 1e5 1e6]
    got = localRosensteinSlope(c * x, Fs);
    tc.verifyEqual(got, base, 'RelTol', 0.02, sprintf( ...
        ['Scaling the series by %g changed the exponent from %.4f to %.4f\n' ...
         '(%.0f%% of the unscaled value). lambda is invariant under\n' ...
         'multiplication by a positive constant. LyE_R.m excludes near-in-time\n' ...
         'neighbours by assigning the magic distance 1e5, which stops being\n' ...
         'large once the data are. Scale the sentinel to the data (or use Inf)\n' ...
         'and expose it as an input rather than hard-coding it.'], ...
        c, base, got, 100*got/base));
end
end

function testLyE_R_NeighboursAreNotTemporallyAdjacent(tc)
% The mechanism behind the test above, asserted separately so a regression is
% diagnosed rather than merely detected.
for c = [1 1e6]
    out = LyE_R(c * tc.TestData.lorenz, 1/0.03, 5, 3);
    lag = abs(out(:,1) - out(:,2));
    frac = mean(lag <= 4);            % 4 = round(tau*0.8), the exclusion width
    tc.verifyLessThan(frac, 0.05, sprintf( ...
        ['At scale %g, %.1f%% of matched pairs are inside the exclusion window\n' ...
         '(median separation %g samples). Rosenstein requires the neighbour to\n' ...
         'be on a DIFFERENT orbit; pairing a point with its own successor\n' ...
         'measures interpolation, not divergence.'], c, 100*frac, median(lag)));
end
end

function testLyE_R_ReturnsThreeColumnsRegardlessOfData(tc)
% out is preallocated as [(1:M)' IND2] -- two columns -- and column 3 is only
% written inside `if sum(distanceM) ~= 0`. If that never fires the caller
% receives a 2-column matrix and out(:,3) throws somewhere else entirely.
% Same species as the emd.py shape bug: the return TYPE depends on the data.
cases = {'skewtent', 0.3, 1, 2; 'logistic', [], 1, 2; ...
         'lorenz', [], 5, 3; 'rossler', [], 8, 3};
for k = 1:size(cases,1)
    if isempty(cases{k,2})
        y = nonantest.signals(cases{k,1}, 2000);
    else
        y = nonantest.signals(cases{k,1}, 2000, cases{k,2});
    end
    out = LyE_R(y, 1, cases{k,3}, cases{k,4});
    tc.verifyEqual(size(out,2), 3, sprintf( ...
        ['LyE_R returned %d columns on the %s series. The divergence curve is\n' ...
         'column 3, so a 2-column return makes every documented downstream use\n' ...
         'fail with an unrelated indexing error.'], size(out,2), cases{k,1}));
end
end

function testChaoticExceedsPeriodic(tc)
% Ordering needs no reference value and is the minimum any estimator owes.
[~, chaotic]  = LyE_W(nonantest.signals('skewtent', 2000, 0.3), 1, 1, 2, 5);
[~, periodic] = LyE_W(tc.TestData.sine, 1, 12, 3, 10);
tc.verifyGreaterThan(chaotic, periodic + 0.1, sprintf( ...
    ['A chaotic map scored %.4f and a pure sine %.4f. An estimator that\n' ...
     'cannot order these cannot distinguish chaos from a cycle.'], ...
    chaotic, periodic));
tc.verifyLessThan(abs(periodic), 0.25, sprintf( ...
    'LyE_W gave %.4f on a pure sine, where lambda = 0.', periodic));
end

% =================================================================
% LAYER 2 -- exact lambda. Theorems, not computations.
% =================================================================

function testRecoversExactSkewTentLadder(tc)
% lambda = -p ln p - (1-p) ln(1-p) exactly. The ladder matters more than any
% single point: an estimator can be biased and still track, or be right at one
% p and wrong elsewhere. Measured ratios 0.931 / 0.904 / 0.905 / 0.902 at
% p = 0.4 / 0.3 / 0.2 / 0.1 -- a consistent ~10% undershoot, not scatter.
ps = [0.4 0.3 0.2 0.1];
ratios = zeros(size(ps));
for k = 1:numel(ps)
    y = nonantest.signals('skewtent', 2000, ps(k));
    ref = nonantest.lambdaReference('skewtent', ps(k));
    [~, L] = LyE_W(y, 1, 1, 2, 5);
    ratios(k) = L / ref.bits;
    tc.verifyEqual(L, ref.bits, 'RelTol', 0.20, sprintf( ...
        'skew tent p=%.1f: LyE_W %.4f bits against an EXACT %.4f bits (ratio %.3f)', ...
        ps(k), L, ref.bits, ratios(k)));
end
fprintf('    [exact] skew tent ladder ratios: %s (mean %.3f, sd %.4f)\n', ...
    mat2str(round(ratios,3)), mean(ratios), std(ratios));

% Bias should be roughly proportional across the ladder, not erratic.
tc.verifyLessThan(std(ratios), 0.05, sprintf( ...
    ['The ratio to the exact value varies by sd %.4f across the ladder.\n' ...
     'A consistent proportional bias is a property of the method; scatter\n' ...
     'this large suggests the estimate depends on something it should not.'], ...
    std(ratios)));
end

function testRecoversExactLogisticExponent(tc)
% lambda = ln 2 exactly at r = 4, by conjugacy to the tent map.
y = nonantest.signals('logistic', 2000);
ref = nonantest.lambdaReference('logistic');
[~, L] = LyE_W(y, 1, 1, 2, 5);
tc.verifyEqual(L, ref.bits, 'RelTol', 0.15, sprintf( ...
    'logistic r=4: LyE_W %.4f bits against an EXACT %.4f bits (= ln 2 nats)', ...
    L, ref.bits));
end

function testDyadicTentIsRejected(tc)
% p = 0.5 is exactly a binary shift, so in floating point the orbit exhausts
% the mantissa and collapses to exactly 0 after ~50 iterations -- 1 distinct
% value in 2000. LyE_W returns NaN and LyE_R returns a 2-column matrix on it.
% Anyone building a "known answer" test on the standard tent map at slope 2
% gets a silently dead series, so the generator refuses rather than obliging.
tc.verifyError(@() nonantest.signals('skewtent', 100, 0.5), ...
    'nonantest:signals:dyadicTent');
end

% =================================================================
% LAYER 3 -- numerical reference (Sprott 2003). Loose, and labelled.
% =================================================================

function testOrderOfMagnitudeAgainstPublishedFlowValues(tc)
% Sanity, not accuracy. Parameters are stated per system because the estimate
% depends on them: measured ratios 0.97 (Henon), 0.99 (Rossler), 1.01
% (Lorenz) at these settings. Sweeping evolve, tau and dim moves Lorenz from
% 0.69 to 1.43, so the agreement here is a property of the parameters as much
% as of the code -- which is exactly why the tolerance is wide.
cases = { ...
    'henon',   1,      1,  2,  5; ...
    'rossler', 1/0.1,  8,  3, 10; ...
    'lorenz',  1/0.03, 10, 5, 10};
for k = 1:size(cases,1)
    name = cases{k,1};
    y = nonantest.signals(name, 2000);
    ref = nonantest.lambdaReference(name);
    [~, L] = LyE_W(y, cases{k,2}, cases{k,3}, cases{k,4}, cases{k,5});
    nats = L * log(2);
    tc.verifyGreaterThan(nats, 0, sprintf('%s: lambda must be positive', name));
    tc.verifyEqual(nats, ref.nats, 'RelTol', 0.60, sprintf( ...
        ['%s: LyE_W %.4f nats against %s reference %.4f (%s).\n' ...
         'Tolerance is deliberately wide -- an order-of-magnitude and sign\n' ...
         'check against a value for the SYSTEM, not for this estimator.'], ...
        name, nats, ref.tier, ref.nats, ref.source));
    fprintf('    [%s] %-8s %.4f nats vs ref %.4f (ratio %.2f)\n', ...
        ref.tier, name, nats, ref.nats, nats/ref.nats);
end
end

function testLyE_W_ReturnsBitsNotNats(tc)
% Pins the convention against the exact logistic value, where bits and nats
% differ by a factor of 1.44 and cannot be confused.
y = nonantest.signals('logistic', 2000);
[~, L] = LyE_W(y, 1, 1, 2, 5);
tc.verifyEqual(L, 1.0, 'AbsTol', 0.15, sprintf( ...
    ['LyE_W returned %.4f on the logistic map at r=4. Expected ~1.0 BITS per\n' ...
     'iteration (lambda = ln 2 nats). If this now reads ~0.69 the function has\n' ...
     'been switched to nats, which is a breaking change to every published\n' ...
     'NONAN value and must be released as one.'], L));
end

% =================================================================
% Interface and headless behaviour.
% =================================================================

function testLyE_W_DocumentedArgumentListWorks(tc)
% The header must describe a call that can actually be made. It previously
% documented a nine-argument form including SCALEMN, but the body tests
% nargin == 8 and reads varargin{1:3} as SCALEMX, ANGLMX, ZMULT -- SCALEMN
% having been removed from the code, per the function's own changelog,
% without the signature line being updated.
s = nonantest.sideEffects(@() LyE_W(tc.TestData.lorenz, 1/0.03, 5, 3, 10, ...
                                    1, 30*pi/180, 1));
tc.verifyFalse(s.errored, sprintf( ...
    'The documented extended form of LyE_W fails: "%s".', localMsg(s)));

% And the form that is no longer supported must not silently appear to work.
s9 = nonantest.sideEffects(@() LyE_W(tc.TestData.lorenz, 1/0.03, 5, 3, 10, ...
                                     1, 0.1, 30*pi/180, 1));
tc.verifyTrue(s9.errored, ...
    'A nine-argument call should raise, since SCALEMN is not implemented.');
end

function testLyE_W_HeaderDoesNotDocumentUnimplementedArguments(tc)
here = fileparts(mfilename('fullpath'));
src = fileread(fullfile(fileparts(fileparts(here)), 'matlab', 'LyE_W.m'));
src = regexprep(src, '\r\n?', newline);
head = extractBefore(src, min(strlength(src), 4000));
tc.verifyEmpty(regexp(head, 'SCALEMX,SCALEMN', 'once'), ...
    ['The header documents a signature containing SCALEMN, which the body ' ...
     'does not accept. Either implement it or remove it from the header.']);
end

function testLyE_W_DocumentsItsUnits(tc)
% LyE_W accumulates log2 and returns bits; every published reference value is
% in nats. The factor is ln 2 = 0.693, large enough that a correct answer
% looks like a 44% error if the convention is not stated.
here = fileparts(mfilename('fullpath'));
src = fileread(fullfile(fileparts(fileparts(here)), 'matlab', 'LyE_W.m'));
head = extractBefore(src, min(strlength(src), 4000));
tc.verifyNotEmpty(regexp(head, 'BITS|bits per', 'once'), ...
    ['The header does not state the units of LyE. It returns bits per unit ' ...
     'time; the literature quotes nats.']);
end

function testLyE_R_DoesNotArmDebuggerOrOpenFigures(tc)
s = nonantest.sideEffects(@() LyE_R(tc.TestData.lorenz, 1/0.03, 5, 3));
tc.verifyFalse(s.errored, sprintf('LyE_R errored: %s', localMsg(s)));
tc.verifyFalse(s.dbstop, 'LyE_R executed `dbstop if error`.');
tc.verifyEqual(s.figures, 0, 'LyE_R opened a figure in its default form.');
end

function testLyE_R_MemoryIsNotQuadratic(tc)
% Only per-propagation-step means are consumed, so the (M-1)-by-(M-1)
% distance matrix need never be materialised. Storing it costs 8*(M-1)^2
% bytes: 0.9 GB at N=10800, 3.2 GB at N=20000. That is a ceiling on series
% length, not a slowdown.
%
% Asserted against the source rather than by measuring peak RSS, which
% MATLAB does not expose portably. The functional test below confirms the
% accumulator path still produces a usable curve.
here = fileparts(mfilename('fullpath'));
src = fileread(fullfile(fileparts(fileparts(here)), 'matlab', 'LyE_R.m'));
src = regexprep(src, '\r\n?', newline);
lines = strsplit(src, newline, 'CollapseDelimiters', false);

bad = {};
for k = 1:numel(lines)
    stripped = regexprep(lines{k}, '%.*$', '');
    if ~isempty(regexp(stripped, 'zeros\s*\(\s*M\s*-\s*1\s*,\s*M\s*-\s*1\s*\)', 'once'))
        bad{end+1} = sprintf('  line %d: %s', k, strtrim(lines{k})); %#ok<AGROW>
    end
end
tc.verifyEmpty(bad, sprintf( ...
    ['LyE_R allocates a dense (M-1)-by-(M-1) matrix:\n%s\n' ...
     'Only per-row means are used. Accumulate a running sum and count per\n' ...
     'propagation step instead.'], strjoin(bad, newline)));
end

function testLyE_R_DivergenceCurveIsUsableAtLength(tc)
% Guards the accumulator path: the curve must still be finite, populated and
% monotone-ish over its scaling region at a length where the old dense
% allocation was already 0.3 GB.
N = 6000;
y = nonantest.signals('lorenz', N);
out = LyE_R(y, 1/0.03, 5, 3);
tc.verifyEqual(size(out,2), 3, 'LyE_R must return three columns');
d = out(:,3);
nz = d(d ~= 0);
tc.verifyGreaterThan(numel(nz), 100, ...
    'divergence curve is almost entirely empty');
tc.verifyTrue(all(isfinite(nz)), ...
    'divergence curve contains non-finite values');
[slope, ~, info] = nonantest.scaling_region(d, 1/0.03);
tc.verifyTrue(info.ok, 'no clean scaling region found at N = 6000');
tc.verifyGreaterThan(slope, 0, ...
    sprintf('Lorenz slope came out %.4f; it must be positive', slope));
end

% ---------------------------------------------------------------- helpers

function s = localRosensteinSlope(x, Fs)
% LyE_R's 4-argument form returns the divergence CURVE, not an exponent, so
% the scaling region is part of the measurement and is held fixed at samples
% 5-100 for every comparison. Choosing a good window generally is an open
% problem, tracked separately.
out = LyE_R(x, Fs, 5, 3);
d = out(:,3);
idx = (5:100)';
p = polyfit(idx/Fs, d(idx), 1);
s = p(1);
end

function m = localMsg(s)
if isempty(s.err), m = ''; else, m = s.err.message; end
end

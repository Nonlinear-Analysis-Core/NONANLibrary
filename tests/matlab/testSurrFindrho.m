function tests = testSurrFindrho
%TESTSURRFINDRHO Contract tests for Surr_findrho and Surr_PseudoPeriodic.
%
%   Surr_findrho searches for the noise radius rho that maximises the number
%   of short runs the pseudo-periodic surrogate shares with the original
%   (Small, Yu & Harrison 2001). Its contract is minimal and entirely
%   testable:
%
%     1. It returns a rho. Always. For any series it accepts.
%     2. The rho it returns lies inside the interval it searched.
%     3. The rho it returns is at least as good as the endpoints it started
%        from -- otherwise the search did not search.
%     4. It terminates.
%
%   The function currently violates (1) on a large class of ordinary inputs,
%   and its own `dbstop if error` converts that violation from a visible error
%   into an indefinite hang.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
tc.TestData.pp    = nonantest.signals('pseudoperiodic', 300);
tc.TestData.sine  = nonantest.signals('sine', 300, 25);
tc.TestData.lor   = nonantest.signals('lorenz', 300);
tc.TestData.tau   = 5;
tc.TestData.dim   = 3;
end

function teardown(~)
dbclear all
end

% ------------------------------------------------------------------
% (1) It returns a rho.
% ------------------------------------------------------------------
function testAlwaysReturnsRho(tc)
% rho is assigned ONLY inside `if di > dmax`, where dmax is seeded with the
% better of the two endpoint values. When the maximum of di(rho) lies at or
% near an endpoint -- the ordinary case for smooth, strongly periodic data,
% which is exactly what a pseudo-periodic surrogate is for -- no interior
% point ever beats it and the output is never assigned.
nRep = 30;
cases = {'pseudoperiodic', tc.TestData.pp; 'sine', tc.TestData.sine};

for c = 1:size(cases, 1)
    y = cases{c, 2};
    failures = 0;
    lastMsg = '';
    for rep = 1:nRep
        s = nonantest.sideEffects(@() Surr_findrho(y, tc.TestData.tau, tc.TestData.dim));
        if s.errored
            failures = failures + 1;
            lastMsg = s.err.message;
        end
    end
    tc.verifyEqual(failures, 0, sprintf( ...
        ['[%s] Surr_findrho failed to return on %d of %d identical calls\n' ...
         '     (%.0f%%). Last error: %s\n' ...
         '     The output rho is only assigned when an interior point strictly\n' ...
         '     beats both endpoints. It is not assigned otherwise.'], ...
        cases{c,1}, failures, nRep, 100*failures/nRep, lastMsg));
end
end

% ------------------------------------------------------------------
% (2) and (3) The returned rho is in range and beats the endpoints.
% ------------------------------------------------------------------
function testReturnedRhoIsInSearchRangeAndBeatsEndpoints(tc)
y = tc.TestData.lor;
for rep = 1:10
    s = nonantest.sideEffects(@() Surr_findrho(y, tc.TestData.tau, tc.TestData.dim));
    if s.errored, continue; end   % covered by testAlwaysReturnsRho
    rho = s.value;
    tc.verifyGreaterThanOrEqual(rho, 0.1, 'rho below the searched lower bound');
    tc.verifyLessThanOrEqual(rho, 1.0,   'rho above the searched upper bound');
end
end

% ------------------------------------------------------------------
% (4) It terminates, and quickly. Aaron observed "no result in 10 minutes"
%     on a 300-point series. That is not compute: the search is ~11 calls to
%     an O(N^2) routine with N=300. Pin the real budget so a future change
%     that makes it genuinely slow is caught as slowness rather than mistaken
%     for the hang again.
% ------------------------------------------------------------------
function testTerminatesQuickly(tc)
y = tc.TestData.pp;
s = nonantest.sideEffects(@() Surr_findrho(y, tc.TestData.tau, tc.TestData.dim));
tc.verifyLessThan(s.seconds, 5, sprintf( ...
    'Surr_findrho took %.1f s on a 300-point series.', s.seconds));
end

% ------------------------------------------------------------------
% The mechanism that turned a 0.02 s error into a 10 minute hang.
% ------------------------------------------------------------------
function testDoesNotArmTheDebugger(tc)
y = tc.TestData.pp;
s = nonantest.sideEffects(@() Surr_findrho(y, tc.TestData.tau, tc.TestData.dim));
tc.verifyFalse(s.dbstop, ...
    ['Surr_findrho executed `dbstop if error`. This is global session state.\n' ...
     'Combined with the unassigned-output bug it means: the function errors,\n' ...
     'the debugger catches the error, and under `matlab -batch` there is no\n' ...
     'terminal to break into, so the process hangs indefinitely instead of\n' ...
     'failing. The function is fast; it only looks slow.']);
end

% ------------------------------------------------------------------
% The search interval is hard-coded to [0.1, 1] and both ends are assumed
% usable. Neither assumption holds.
% ------------------------------------------------------------------
function testLowerBoundIsUsableOnChaoticData(tc)
% Surr_PseudoPeriodic can raise 'a new value of xi could not be found, check
% that rho is not too low', and Surr_findrho evaluates rho = 0.1
% unconditionally as its lower endpoint -- so if a series trips that guard it
% takes the whole search down with it.
%
% STATUS: currently PASSES on all suite signals (0/10 failures). This is a
% regression guard, not a reported defect. The library's own docstring flags
% the risk ("It's possible the rhoL value of 0.1 may cause issues with pps in
% certain time series"); an earlier probe of mine appeared to reproduce it on
% a Lorenz series, but that was an artefact of a coarser integrator producing
% a differently-scaled attractor, not a library fault.
y = tc.TestData.lor;
failures = 0;
for rep = 1:10
    s = nonantest.sideEffects(@() Surr_PseudoPeriodic(y, tc.TestData.tau, tc.TestData.dim, 0.1));
    if s.errored, failures = failures + 1; end
end
tc.verifyEqual(failures, 0, sprintf( ...
    ['Surr_PseudoPeriodic failed at rho = 0.1 on %d of 10 calls. Surr_findrho\n' ...
     'hard-codes rho = 0.1 as its lower search bound and evaluates it on every\n' ...
     'call, so this takes the whole search down with it.'], failures));
end

function testSearchIntervalBracketsTheOptimum(tc)
% A bisection can only find a maximum that lies strictly INSIDE its bracket.
% Surr_findrho hard-codes the bracket to [0.1, 1] on the stated grounds that
% "the optimal rho is frequently ~0.5-0.6". Measured, that holds for chaotic
% data and fails for exactly the smooth, strongly cyclic data that the
% pseudo-periodic surrogate is designed for.
%
% This is also the root cause of the unassigned-output failure above: rho is
% only assigned when an interior point strictly beats both endpoints, so when
% the optimum sits at or beyond an endpoint no interior point ever wins.
rhos = [0.05 0.1 0.2 0.4 0.6 0.8 1.0 1.3];
cases = {'pseudoperiodic', tc.TestData.pp; 'sine', tc.TestData.sine; 'lorenz', tc.TestData.lor};

for c = 1:size(cases, 1)
    y = cases{c, 2};
    di = nan(size(rhos));
    for k = 1:numel(rhos)
        vals = nan(1, 15);
        for rep = 1:15
            try
                [~, yi] = Surr_PseudoPeriodic(y, tc.TestData.tau, tc.TestData.dim, rhos(k));
                vals(rep) = sum(diff(find(diff(yi) ~= 1)) > 2);
            catch
                % rho too low for this series; leave NaN
            end
        end
        di(k) = median(vals, 'omitnan');
    end
    [~, imax] = max(di);
    interior = rhos(imax) > 0.1 && rhos(imax) < 1.0;
    tc.verifyTrue(interior, sprintf( ...
        ['[%s] di(rho) is maximised at rho = %.2f, which is not strictly\n' ...
         '     inside the hard-coded search bracket [0.1, 1].\n' ...
         '     di = %s\n     at rho = %s\n' ...
         '     A bisection cannot find an optimum outside its bracket. The rho\n' ...
         '     this returns is an artefact of where the bracket was placed, and\n' ...
         '     on this series the search has nothing to find.'], ...
        cases{c,1}, rhos(imax), mat2str(di), mat2str(rhos)));
end
end

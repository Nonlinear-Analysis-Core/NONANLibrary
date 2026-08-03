function [slope, idx, info] = scaling_region(d, fs, opts)
%SCALING_REGION Automatic linear scaling region of a divergence curve.
%
%   [slope, idx, info] = nonantest.scaling_region(d, fs)
%
%   d is the average line divergence curve (LyE_R column 3), fs the sample
%   rate. Returns the fitted slope in natural log units per unit time, the
%   indices used, and diagnostics.
%
%   WHY THIS EXISTS. LyE_R does not return an exponent -- it returns a curve,
%   and the caller picks a window and fits a slope. That choice dominates the
%   answer. Measured on one Lorenz series, fitting samples 2-30 gives 1.67
%   and samples 10-150 gives 0.74, from identical data.
%
%   Worse, a single fixed window cannot serve both maps and flows. A map's
%   divergence curve saturates within a few iterations: for the logistic map
%   it is flat by sample 13, so a 5-50 window sits entirely on the plateau
%   and returns a slope near zero. That is not an estimator failure, it is a
%   window failure -- and it looks exactly like an estimator failure in a
%   results table, which is how a benchmark ends up libelling a method.
%
%   METHOD. The scaling region is the initial, approximately linear rise
%   before saturation. This searches windows anchored near the start of the
%   curve and picks the one that is longest subject to staying linear:
%
%     1. Locate saturation: the first index where the curve stops rising,
%        taken as the first point that fails to exceed the running maximum by
%        a meaningful margin over a short lookahead.
%     2. Among candidate windows [a, b] with a small and b <= saturation,
%        each at least MinLen points, score by R^2 of a straight-line fit.
%     3. Prefer the longest window whose R^2 is within Tol of the best R^2
%        seen. Longest-among-near-best avoids picking a 3-point window that
%        is trivially straight.
%
%   This is deliberately simple and inspectable. It is not the last word on
%   scaling-region selection -- that is an open problem and a tracked task --
%   but it is uniform, automatic, and stated, which is what a benchmark
%   needs. Callers get idx and info back so the choice can be audited or
%   plotted rather than trusted.

arguments
    d  (:,1) double
    fs (1,1) double = 1
    opts.MinLen (1,1) double = 4
    opts.MaxLen (1,1) double = 200
    opts.Tol    (1,1) double = 0.02
end

info = struct('saturation', NaN, 'r2', NaN, 'nCandidates', 0, 'ok', false);

d = d(isfinite(d));
n = numel(d);
if n < opts.MinLen + 2
    slope = NaN; idx = []; return
end

sat = findSaturation(d);
info.saturation = sat;

hi = min(sat, opts.MaxLen);
if hi < opts.MinLen + 1
    hi = min(n, max(opts.MinLen + 1, 8));
end

best = struct('r2', -Inf, 'a', 1, 'b', min(hi, opts.MinLen + 1), 'len', 0);
cands = {};
% Anchor near the start: the scaling region begins where the neighbour pair
% separation is still small. Allow a few starting points to skip any initial
% transient in the curve itself.
for a = 1:min(4, max(1, hi - opts.MinLen))
    for b = (a + opts.MinLen - 1):hi
        seg = d(a:b);
        r2 = lineR2((a:b)', seg);
        cands{end+1} = struct('r2', r2, 'a', a, 'b', b, 'len', b - a + 1); %#ok<AGROW>
        if r2 > best.r2
            best = cands{end};
        end
    end
end
info.nCandidates = numel(cands);
if isempty(cands)
    slope = NaN; idx = []; return
end

% Longest window whose R^2 is within Tol of the best.
pick = best;
for k = 1:numel(cands)
    c = cands{k};
    if c.r2 >= best.r2 - opts.Tol && c.len > pick.len
        pick = c;
    end
end

idx = (pick.a:pick.b)';
p = polyfit(idx / fs, d(idx), 1);
slope = p(1);
info.r2 = pick.r2;
info.ok = pick.len >= opts.MinLen && isfinite(slope);
end

% ---------------------------------------------------------------- helpers

function s = findSaturation(d)
%FINDSATURATION First index at which the curve stops climbing.
n = numel(d);
span = max(d) - min(d);
if span <= 0
    s = n; return
end
tol = 0.01 * span;                 % "meaningful" rise over the lookahead
look = max(3, round(0.02 * n));
s = n;
runmax = d(1);
for i = 2:(n - look)
    runmax = max(runmax, d(i));
    ahead = max(d(i+1 : i+look));
    if ahead <= runmax + tol
        s = i;
        return
    end
end
end

function r2 = lineR2(x, y)
x = x(:); y = y(:);
if numel(x) < 2 || all(y == y(1))
    r2 = -Inf; return
end
p = polyfit(x, y, 1);
res = y - polyval(p, x);
ss = sum((y - mean(y)).^2);
if ss <= 0
    r2 = -Inf; return
end
r2 = 1 - sum(res.^2) / ss;
end

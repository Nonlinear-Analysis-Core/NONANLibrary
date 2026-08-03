function [y, info] = sprott_series(sys, n, opts)
%SPROTT_SERIES Scalar observable from a Sprott Appendix A system.
%
%   [y, info] = nonantest.sprott_series(sys, n)
%   sys is one element of nonantest.sprott_catalog().
%
%   Returns an n-by-1 scalar observable and a struct of what was done:
%     info.fs        samples per unit time (1 for maps)
%     info.decim     integrator steps per output sample (flows)
%     info.period    estimated dominant period, in samples
%     info.degenerate  true if the orbit collapsed or diverged
%
%   Protocol, uniform across all systems and not tuned per system. Flows are
%   integrated with RK4 at the catalogue's dt, a 20000-step transient is
%   discarded, and decimation is chosen so the dominant spectral period lands
%   near TargetPeriod samples. Maps are iterated directly after a 1000-step
%   transient. The systems' natural timescales span three orders of
%   magnitude, so a single fixed rate would score estimators mostly on that
%   mismatch.
%
%   Degeneracy checks. An orbit can collapse to a fixed point, diverge,
%   exhaust the mantissa, or drift without bound, each producing a series
%   that looks valid to a caller. These are flagged in info.degenerate rather
%   than returned silently.

arguments
    sys (1,1) struct
    n   (1,1) double {mustBePositive, mustBeInteger}
    opts.TargetPeriod (1,1) double = 40
    opts.Transient    (1,1) double = 20000
end

info = struct('fs', 1, 'decim', 1, 'period', NaN, 'degenerate', false, ...
              'reason', "");

if ~sys.usable
    y = [];
    info.degenerate = true;
    info.reason = sys.note;
    return
end

if sys.kind == "map"
    y = iterateMap(sys, n, 1000);
    info.fs = 1;
else
    % Pilot run to find the dominant period, then decimate to hit the target.
    pilot = integrateFlow(sys, 4096, 1, opts.Transient);
    pk = dominantPeriod(pilot);                 % in integrator steps
    if ~isfinite(pk) || pk <= 0
        pk = 200;
    end
    info.decim = max(1, round(pk / opts.TargetPeriod));
    y = integrateFlow(sys, n, info.decim, opts.Transient);
    info.fs = 1 / (sys.dt * info.decim);
end

[bad, why] = isDegenerate(y);
info.degenerate = bad;
info.reason = why;
if ~bad
    info.period = dominantPeriod(y);
end
end

% ---------------------------------------------------------------- internals

function y = iterateMap(sys, n, transient)
v = sys.x0;
for i = 1:transient
    v = sys.f(v);
    if ~all(isfinite(v)), break; end
end
y = zeros(n, 1);
for i = 1:n
    v = sys.f(v);
    if ~all(isfinite(v))
        y(i:end) = NaN;
        return
    end
    y(i) = v(sys.obs);
end
end

function y = integrateFlow(sys, n, decim, transient)
dt = sys.dt;
f = sys.f;
v = sys.x0;
t = 0;
for i = 1:transient
    [v, t] = rk4(f, v, t, dt);
    if ~all(isfinite(v)), break; end
end
y = zeros(n, 1);
for i = 1:n
    for k = 1:decim
        [v, t] = rk4(f, v, t, dt);
    end
    if ~all(isfinite(v))
        y(i:end) = NaN;
        return
    end
    y(i) = v(sys.obs);
end
end

function [v, t] = rk4(f, v, t, dt)
k1 = f(t, v);
k2 = f(t + dt/2, v + dt/2*k1);
k3 = f(t + dt/2, v + dt/2*k2);
k4 = f(t + dt,   v + dt*k3);
v = v + dt/6*(k1 + 2*k2 + 2*k3 + k4);
t = t + dt;
end

function p = dominantPeriod(y)
%DOMINANTPERIOD Period, in samples, of the largest non-DC spectral peak.
y = y(isfinite(y));
if numel(y) < 16
    p = NaN;
    return
end
y = y - mean(y);
if std(y) == 0
    p = NaN;
    return
end
m = 2^nextpow2(numel(y));
P = abs(fft(y, m)).^2;
half = P(2:floor(m/2));
[~, k] = max(half);
p = m / k;
end

function [bad, why] = isDegenerate(y)
bad = false;
why = "";
if isempty(y)
    bad = true; why = "empty series"; return
end
if any(~isfinite(y))
    bad = true; why = "orbit diverged or produced non-finite values"; return
end
if std(y) == 0
    bad = true; why = "orbit collapsed to a constant"; return
end
u = numel(unique(y));
if u < numel(y) / 20
    bad = true;
    why = sprintf("orbit collapsed to %d distinct values out of %d " + ...
                  "(periodic, or floating-point mantissa exhausted)", u, numel(y));
    return
end
% A near-constant tail catches slow collapse onto a fixed point.
tail = y(max(1, end-99):end);
if std(tail) < 1e-10 * max(1, std(y))
    bad = true; why = "orbit settled to a fixed point"; return
end

% Unbounded observable. Some systems are chaotic on a bounded attractor in
% one coordinate but diffuse without bound in another: a pendulum angle in a
% running solution, or labyrinth chaos, which random-walks through space.
% Delay embedding assumes the observable is bounded and recurrent -- points
% close in phase space must be revisits, not merely nearby in time. A
% drifting observable has no recurrences at all, so every "nearest neighbour"
% is a temporal neighbour and both estimators return ~0.
% Measured drift/range: 0.70 (damped pendulum), 1.10 (driven pendulum),
% 0.92 (labyrinth) against 0.04 (Lorenz) and 0.00 (Rossler).
p = polyfit((1:numel(y))', y, 1);
drift = abs(p(1)) * numel(y);
rangeY = max(y) - min(y);
if rangeY > 0 && drift > 0.5 * rangeY
    bad = true;
    why = sprintf("observable is unbounded (linear drift is %.0f%% of its " + ...
                  "range); delay embedding requires a bounded, recurrent " + ...
                  "observable. Wrapping or differencing recovers only part " + ...
                  "of the exponent (measured ratio 0.12 to 0.28), so this " + ...
                  "system needs a different observable, not a transform.", ...
                  100*drift/rangeY);
    return
end
end

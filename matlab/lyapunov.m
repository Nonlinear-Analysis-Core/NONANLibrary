function [lambda, extra] = lyapunov(x, fs, opts)
%LYAPUNOV Largest Lyapunov exponent of a time series or phase space.
%   LAMBDA = LYAPUNOV(X,FS) estimates the largest Lyapunov exponent of X
%   sampled at FS, in NATS per unit time.
%
%   X may be either a column vector, in which case a phase space is
%   reconstructed using Tau and Dim, or an N-by-D matrix that is already a
%   phase space, in which case Tau and Dim are ignored. Passing a phase
%   space directly lets the reconstruction be chosen, inspected or shared
%   between estimators rather than being redone inside each one.
%
%   [LAMBDA,EXTRA] = LYAPUNOV(...) returns a struct of method-specific
%   diagnostics: the divergence curve and fitted scaling region for
%   "rosenstein", the per-iteration table for "wolf", plus the phase space
%   used, the algorithm, and the units.
%
%   ___ = LYAPUNOV(X,FS,Algorithm=ALG) selects the method:
%      "rosenstein"  (default) mean log divergence of nearest neighbours,
%                    slope fitted over an automatically selected scaling
%                    region. Aliases: "r".
%      "wolf"        renormalising nearest-neighbour tracking. Aliases: "w".
%
%   ___ = LYAPUNOV(X,FS,Tau=T,Dim=D) sets the delay and embedding dimension
%   used when X is a column vector. Defaults are Tau = 1 and Dim = 3. Choose
%   Tau with AMI and Dim with FNN.
%
%   ___ = LYAPUNOV(X,FS,TheilerWindow=W) excludes candidate neighbours within
%   +/-W samples in time ("rosenstein" only). Default round(Tau*0.8).
%
%   When X is already a phase space it carries no record of the delay that
%   built it, so the default cannot be derived and LYAPUNOV warns if neither
%   TheilerWindow nor Tau is given. The choice matters: on one Lorenz
%   reconstruction at Tau=10, the exponent is 0.8022 with a window of 8 and
%   0.7833 with the default window of 1.
%
%   ___ = LYAPUNOV(X,FS,Evolve=E) sets the propagation length in samples
%   ("wolf" only). Default 10.
%
%   Input Arguments
%      X   time series (column vector) or phase space (N-by-D matrix)
%      FS  sampling frequency, Hz. Use 1 for maps.
%
%   Notes
%   LAMBDA is returned in NATS per unit time for both methods, so they are
%   directly comparable. LYE_W natively accumulates log2 and returns bits;
%   this wrapper converts. Published reference values, including Sprott
%   (2003) Appendix A, are in nats.
%
%   The two methods disagree, and the disagreement is real rather than a bug
%   in either. Over the 55 usable systems of Sprott's Appendix A the median
%   ratio to the reference is 0.96 for "wolf" and 0.85 for "rosenstein". Both
%   are weakest on conservative systems, where lambda is small and the noise
%   floor dominates. Report which method produced a published exponent.
%
%   "rosenstein" fits a slope to a divergence curve, and the fitted window
%   dominates the answer: on one Lorenz series, samples 2-30 give 1.67 and
%   samples 10-150 give 0.74 from identical data. The window is selected
%   automatically and returned in EXTRA.scalingRegion so the choice can be
%   inspected rather than trusted.
%
%   Examples
%      % From a time series, delay from AMI and dimension from FNN
%      tau = ami(x, 50);
%      lam = lyapunov(x, 100, Tau=tau, Dim=5);
%
%      % Same reconstruction, both methods, no repeated embedding
%      Y  = psr(x, tau, 5);
%      lr = lyapunov(Y, 100, Algorithm="rosenstein");
%      lw = lyapunov(Y, 100, Algorithm="wolf");
%
%      % Inspect the fitted scaling region
%      [lam, extra] = lyapunov(x, 100, Tau=tau, Dim=5);
%      plot(extra.divergence); hold on
%      plot(extra.scalingRegion, extra.divergence(extra.scalingRegion), 'r')
%
%   References
%      Rosenstein, M. T., Collins, J. J. and De Luca, C. J. (1993). A
%      practical method for calculating largest Lyapunov exponents from small
%      data sets. Physica D, 65(1-2), 117-134.
%
%      Wolf, A., Swift, J. B., Swinney, H. L. and Vastano, J. A. (1985).
%      Determining Lyapunov exponents from a time series. Physica D, 16(3),
%      285-317.
%
%      Sprott, J. C. (2003). Chaos and Time-Series Analysis. Oxford
%      University Press, Appendix A.
%
%   See also LYE_R, LYE_W, PSR, AMI, FNN.

arguments
    x  double {mustBeNonempty}
    fs (1,1) double {mustBePositive}
    opts.Algorithm     (1,1) string = "rosenstein"
    opts.Tau           (1,1) double {mustBePositive, mustBeInteger} = 1
    opts.Dim           (1,1) double {mustBePositive, mustBeInteger} = 3
    opts.TheilerWindow double {mustBeScalarOrEmpty} = []
    opts.Evolve        (1,1) double {mustBePositive, mustBeInteger} = 10
end

if anynan(x)
    error('lyapunov:nanInput', ...
        'x contains NaN. Remove or impute before estimating an exponent.');
end

% A column vector is a scalar observable and gets embedded; anything wider is
% taken to be a phase space already.
if isvector(x)
    x = x(:);
    if numel(x) <= (opts.Dim-1)*opts.Tau + 2
        error('lyapunov:tooShort', ...
            ['Series of %d samples is too short for Dim=%d at Tau=%d; ' ...
             'the reconstruction would have %d points.'], ...
            numel(x), opts.Dim, opts.Tau, numel(x)-(opts.Dim-1)*opts.Tau);
    end
    Y = psr(x, opts.Tau, opts.Dim);
    tau = opts.Tau;
else
    Y = x;
    tau = opts.Tau;
    % A supplied phase space carries no record of the delay that built it, so
    % the Theiler window cannot be defaulted from Tau the way it can for a
    % raw series. Guessing changes the answer: on a Lorenz reconstruction at
    % Tau=10 the exponent is 0.8022 with the matching window of 8 and 0.7833
    % with the Tau=1 default. Say so rather than pick silently.
    if isempty(opts.TheilerWindow)
        if opts.Tau == 1
            warning('lyapunov:theilerFromDefaultTau', ...
                ['A phase space was supplied without TheilerWindow or Tau, so ' ...
                 'the temporal exclusion defaults to 1 sample. That is almost ' ...
                 'certainly too small for an embedded series and will bias the ' ...
                 'exponent. Pass TheilerWindow (or Tau) to match the delay used ' ...
                 'to build the phase space.']);
        end
    end
end

theiler = opts.TheilerWindow;
if isempty(theiler)
    theiler = round(tau*0.8);
end
extraTheiler = theiler;

extra = struct('phaseSpace', Y, 'algorithm', lower(opts.Algorithm), ...
               'units', "nats per unit time", 'fs', fs, ...
               'theilerWindow', extraTheiler, 'tau', tau, 'dim', size(Y,2));

switch lower(opts.Algorithm)

    case {"rosenstein", "r"}
        out = lye_r(Y, fs, tau, size(Y,2), 'TheilerWindow', theiler);
        d = out(:,3);
        [lambda, idx, info] = local_scaling_region(d, fs);
        extra.divergence    = d;
        extra.matchedPairs  = out(:,1:2);
        extra.scalingRegion = idx;
        extra.fitR2         = info.r2;
        extra.estimator     = "rosenstein";

    case {"wolf", "w"}
        [out, L] = lye_w(Y, fs, tau, size(Y,2), opts.Evolve);
        lambda = L * log(2);                  % bits -> nats
        extra.iterations = out;
        extra.bits       = L;
        extra.estimator  = "wolf";

    otherwise
        error('lyapunov:unknownAlgorithm', ...
            ['Unknown Algorithm "%s". Supported: "rosenstein" (alias "r"), ' ...
             '"wolf" (alias "w").'], opts.Algorithm);
end
end

% ---------------------------------------------------------------- helpers

function [slope, idx, info] = local_scaling_region(d, fs)
%LOCAL_SCALING_REGION Longest near-linear window before saturation.
%   The divergence curve rises approximately linearly then saturates. A fixed
%   window cannot serve both maps and flows: a map's curve is flat by roughly
%   sample 13, so a window running to 50 sits on the plateau and returns a
%   slope near zero.
d = d(isfinite(d));
n = numel(d);
info = struct('r2', NaN, 'saturation', NaN);
if n < 6
    slope = NaN; idx = []; return
end

sat = local_saturation(d);
info.saturation = sat;
hi = max(6, min(sat, 200));

best = struct('r2', -Inf, 'a', 1, 'b', min(hi,5), 'len', 0);
cand = {};
for a = 1:min(4, max(1, hi-4))
    for b = (a+3):hi
        r2 = local_line_r2((a:b)', d(a:b));
        cand{end+1} = struct('r2', r2, 'a', a, 'b', b, 'len', b-a+1); %#ok<AGROW>
        if r2 > best.r2, best = cand{end}; end
    end
end
if isempty(cand), slope = NaN; idx = []; return, end

pick = best;
for k = 1:numel(cand)
    if cand{k}.r2 >= best.r2 - 0.02 && cand{k}.len > pick.len
        pick = cand{k};
    end
end
idx = (pick.a:pick.b)';
p = polyfit(idx/fs, d(idx), 1);
slope = p(1);
info.r2 = pick.r2;
end

function s = local_saturation(d)
n = numel(d);
span = max(d) - min(d);
if span <= 0, s = n; return, end
tol = 0.01*span;
look = max(3, round(0.02*n));
s = n; runmax = d(1);
for i = 2:(n-look)
    runmax = max(runmax, d(i));
    if max(d(i+1:i+look)) <= runmax + tol, s = i; return, end
end
end

function r2 = local_line_r2(xv, yv)
if numel(xv) < 2 || all(yv == yv(1)), r2 = -Inf; return, end
p = polyfit(xv, yv, 1);
ss = sum((yv - mean(yv)).^2);
if ss <= 0, r2 = -Inf; return, end
r2 = 1 - sum((yv - polyval(p, xv)).^2)/ss;
end

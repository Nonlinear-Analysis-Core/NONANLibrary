function [tau, curve, info] = ami_histogram(x, L, opts)
%AMI_HISTOGRAM Average mutual information vs lag, equal-width histogram estimator.
%
%   [tau, curve] = AMI_HISTOGRAM(x, L) returns the lag of the first local
%   minimum of the average mutual information of x with itself, over lags
%   0:L, together with the full AMI curve.
%
%   [tau, curve, info] = AMI_HISTOGRAM(...) also returns diagnostics.
%
%   Name-value arguments
%     Bins      number of bins per axis. Default [] selects Scott's rule.
%     Fraction  fallback threshold, as a fraction of AMI at lag 0, used when
%               no local minimum exists. Default 0.2 (Abarbanel et al. 1993).
%
%   WHY THE FIRST MINIMUM
%   For delay embedding you want coordinates that are neither redundant
%   (tau too small, x(t) ~ x(t+tau)) nor causally disconnected (tau too
%   large). Mutual information measures general dependence rather than only
%   linear correlation, so its first minimum is the standard choice.
%   Fraser & Swinney (1986) is the origin of that criterion.
%
%   ESTIMATOR, STATED HONESTLY
%   This is an equal-width (fixed-bin) histogram estimator. Fraser & Swinney's
%   own algorithm is a RECURSIVE ADAPTIVE PARTITION, which is not what this
%   computes -- naming this function after them would be a misattribution of
%   the same kind the old name made in the other direction. The first-minimum
%   criterion is theirs; the estimator is a plain histogram.
%
%   Histogram AMI is biased upward at small bin counts and noisy at large
%   ones, and Scott's rule is a UNIVARIATE density heuristic being applied to
%   a bivariate histogram. It is adequate for locating a minimum, which only
%   needs the shape of the curve, and unreliable as an absolute nats/bits
%   value. Use ami(..., Algorithm="kde") or a future estimator if you need
%   the value itself.
%
%   TODO -- estimators worth adding, roughly in order of value here:
%     * adaptive partition  -- Fraser & Swinney's actual 1986 algorithm.
%     * knn / KSG           -- Kraskov, Stogbauer & Grassberger (2004).
%                              Bin-free, much lower bias, the modern default.
%     * copula              -- rank-transform to uniform marginals, estimate
%                              the copula density. Invariant to monotone
%                              transforms of the signal, which is attractive
%                              for biomechanical data with arbitrary units.
%     * kde variants        -- adaptive/variable bandwidth rather than the
%                              fixed Silverman-type rule used by "kde".
%
%   Constant sample size across lags: every lag uses N-L pairs, not N-lag.
%   Otherwise the curve confounds a change in dependence with a change in
%   sample size, and histogram AMI is strongly N-dependent.
%
%   Base MATLAB only -- no Statistics Toolbox.
%
%   References
%     Fraser, A. M. & Swinney, H. L. (1986). Independent coordinates for
%       strange attractors from mutual information. Physical Review A,
%       33(2), 1134-1140.
%     Scott, D. W. (1979). On optimal and data-based histograms.
%       Biometrika, 66(3), 605-610.
%     Abarbanel, H. D. I., Brown, R., Sidorowich, J. J. & Tsimring, L. S.
%       (1993). The analysis of observed chaotic data in physical systems.
%       Reviews of Modern Physics, 65(4), 1331-1392.
%
%   See also AMI, AMI_KDE.

arguments
    x  (:,1) double {mustBeNonempty}
    L  (1,1) double {mustBePositive, mustBeInteger}
    opts.Bins     double {mustBeScalarOrEmpty, mustBePositive, mustBeInteger} = []
    opts.Fraction (1,1) double {mustBePositive} = 0.2
end

if anynan(x)
    error('ami_histogram:nanInput', ...
          'x contains NaN. Mutual information is undefined for missing data; remove or impute first.');
end
N = numel(x);
if L >= N
    error('ami_histogram:lagTooLarge', ...
          'L (%d) must be smaller than the number of samples (%d).', L, N);
end

nBins = opts.Bins;
if isempty(nBins)
    nBins = scottBins(x);
end

% Constant overlap across lags -- see note above.
m = N - L;

edges = binEdges(x, nBins);
curve = zeros(L + 1, 2);
for lag = 0:L
    a = x(1:m);
    b = x((1 + lag):(m + lag));
    curve(lag + 1, :) = [lag, mutualInformationBits(a, b, edges)];
end

[tau, info] = firstMinimum(curve, opts.Fraction);
info.bins = nBins;
info.samplesPerLag = m;
info.estimator = "histogram";
end

% ---------------------------------------------------------------- helpers

function nBins = scottBins(x)
%SCOTTBINS Scott's rule: bin width h = 3.49 * sigma * N^(-1/3).
s = std(x);
if s == 0
    nBins = 1;
    return
end
h = 3.49 * s * numel(x)^(-1/3);
nBins = max(2, ceil((max(x) - min(x)) / h));
end

function edges = binEdges(x, nBins)
%BINEDGES Equal-width edges spanning the data, closed at the top.
%
% The old implementation scaled with `1 + floor(v / (max/(bins-eps)))`, where
% `eps` is 2.2e-16 and eps(bins) is ~3.6e-15 for realistic bin counts -- so
% `bins - eps == bins` exactly and the subtraction did nothing. The maximum
% sample therefore landed in bin bins+1, giving one extra bin holding exactly
% one point. linspace edges plus histcounts' closed final bin is the correct
% and obvious way to say what was intended.
lo = min(x);
hi = max(x);
if hi == lo
    hi = lo + 1;
end
edges = linspace(lo, hi, nBins + 1);
end

function I = mutualInformationBits(a, b, edges)
%MUTUALINFORMATIONBITS I(A;B) in bits from a joint equal-width histogram.
n = numel(a);
pAB = histcounts2(a, b, edges, edges) / n;

nz = pAB > 0;                 % 0*log0 == 0; skip rather than nudge with eps
if ~any(nz, 'all')
    I = 0;
    return
end
pA = sum(pAB, 2);
pB = sum(pAB, 1);

outer = pA * pB;              % implicit expansion, no repmat
I = sum(pAB(nz) .* log2(pAB(nz) ./ outer(nz)));
end

function [tau, info] = firstMinimum(curve, fraction)
%FIRSTMINIMUM Lag of the first strict local minimum of the AMI curve.
%
% The old test used >= and <=, so every interior point of a plateau counted
% as a minimum and adjacent noise-level wiggles were both reported. Requiring
% a strict decrease on the left removes plateaus; a run of equal values can
% still only contribute its first element.
v = curve(:, 2);
isMin = [false; v(2:end-1) < v(1:end-2) & v(2:end-1) <= v(3:end); false];
idx = find(isMin);

info = struct();
info.allMinima = curve(idx, :);
info.usedFallback = false;
info.fractionLag = NaN;

% Fallback for weakly periodic series with no clear minimum (Abarbanel 1993).
below = find(v <= fraction * v(1), 1, 'first');
if ~isempty(below)
    info.fractionLag = curve(below, 1);
end

if ~isempty(idx)
    tau = curve(idx(1), 1);
else
    tau = info.fractionLag;              % NaN if that also failed
    info.usedFallback = true;
    if isnan(tau)
        warning('ami_histogram:noMinimum', ...
                ['No local minimum in AMI over lags 0:%d and the curve never fell to ' ...
                 '%.0f%% of its lag-0 value. Increase L, or inspect the curve directly.'], ...
                curve(end, 1), 100 * fraction);
    end
end
end

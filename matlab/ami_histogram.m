function [tau, curve, info] = ami_histogram(x, L, opts)
%AMI_HISTOGRAM Average mutual information versus lag, histogram estimator.
%   TAU = AMI_HISTOGRAM(X,L) returns the lag of the first local minimum of
%   the average mutual information of X with itself over lags 0:L, using an
%   equal-width joint histogram.
%
%   [TAU,CURVE] = AMI_HISTOGRAM(X,L) also returns the AMI curve as an
%   (L+1)-by-2 array of [lag, ami], in bits.
%
%   [TAU,CURVE,INFO] = AMI_HISTOGRAM(X,L) also returns a struct with fields
%   allMinima, usedFallback, fractionLag, bins, samplesPerLag and estimator.
%
%   ___ = AMI_HISTOGRAM(X,L,Bins=B) sets the number of bins per axis.
%   Default selects B by Scott's rule.
%
%   ___ = AMI_HISTOGRAM(X,L,Fraction=F) sets the fallback threshold, as a
%   fraction of AMI at lag 0, used when the curve has no local minimum.
%   Default 0.2.
%
%   Input Arguments
%      X  time series, real column vector, no NaN
%      L  maximum lag, positive integer, less than numel(X)
%
%   Notes
%   Every lag uses the same number of pairs, N-L rather than N-lag, so the
%   curve is not confounded by a changing sample size.
%
%   The plug-in histogram estimate is biased upward. For independent data the
%   bias is approximately (Bx-1)(By-1)/(2*N*ln2) bits, so at N = 2000 with
%   Scott's rule the floor is about 0.17 bits. Where the true AMI is smaller
%   than that, the curve reflects the estimator rather than the signal. Use
%   AMI_KDE when the AMI value itself matters; the histogram is adequate for
%   locating a minimum.
%
%   Runs on base MATLAB. No toolbox required.
%
%   Examples
%      tau = ami_histogram(x, 50);
%      [tau, curve, info] = ami_histogram(x, 50, Bins=32);
%
%   References
%      Fraser, A. M. and Swinney, H. L. (1986). Independent coordinates for
%      strange attractors from mutual information. Physical Review A, 33(2),
%      1134-1140.
%
%      Scott, D. W. (1979). On optimal and data-based histograms.
%      Biometrika, 66(3), 605-610.
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
%BINEDGES Equal-width edges spanning the data. histcounts closes the last bin.
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
% Strict on the left, so a plateau contributes at most its first point.
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

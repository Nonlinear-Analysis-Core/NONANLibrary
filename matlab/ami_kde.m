function [tau, curve, info] = ami_kde(x, L, opts)
%AMI_KDE Average mutual information vs lag, Gaussian kernel density estimator.
%
%   [tau, curve] = AMI_KDE(x, L) returns the lag of the first local minimum
%   of the average mutual information of x with itself over lags 0:L.
%
%   This is the estimator published as AMI_Thomas: marginal densities from a
%   univariate Gaussian kernel, the joint density from a bivariate Gaussian
%   kernel whose covariance carries the sample correlation, and
%
%       I = (1/n) * sum_i log2( p_xy(i) / (p_x(i) * p_y(i)) )
%
%   evaluated at the data points themselves. Bandwidths follow
%   h = std / n^(1/6), the standard bivariate Silverman-type rule.
%
%   Numerically identical to the original to ~1e-12; see tests/matlab.
%
%   WHAT CHANGED, AND WHY IT MATTERS
%   The original spent its time and memory building matrices it did not need:
%
%     * `Extended(v, n)` replicated a vector by LOOPING n times to fill rows
%       of a preallocated matrix. That is repmat, and with implicit expansion
%       it is not needed at all.
%     * `linear_depth` built TWO full n^2-by-2 matrices (`Blocks`, `Bricks`)
%       and added them, purely to enumerate every pairwise difference, before
%       passing the result to mvnpdf as one n^2-row call.
%     * Block sums were then recovered by `cumsum` over all n^2 rows followed
%       by differencing every n-th element. Beyond being indirect, that loses
%       precision: cumsum accumulates over the whole array, so each block sum
%       is a difference of two large and nearly equal partial sums.
%
%   Peak memory for the joint density was therefore ~5 n^2 doubles. Here it is
%   O(ChunkSize * n), bounded by a name-value argument, and the block sums are
%   computed directly.
%
%   It also dropped its Statistics Toolbox dependency: corr, mvnpdf and
%   normpdf are all replaced with closed forms, so the function now runs on
%   base MATLAB.
%
%   Name-value arguments
%     Fraction   fallback threshold as a fraction of AMI at lag 0. Default 0.2.
%     ChunkSize  rows of the pairwise kernel evaluated at once. Default 512.
%                Lower it if memory is tight; it does not change the result.
%
%   COST. This is O(L * n^2) kernel evaluations and is much slower than
%   Algorithm="histogram". It is the better estimator of the AMI VALUE; the
%   histogram is usually enough to locate a minimum.
%
%   References
%     Thomas, R. D., Moses, N. C., Semple, E. A. & Strang, A. J. (2014). An
%       efficient algorithm for the computation of average mutual information.
%     Fraser, A. M. & Swinney, H. L. (1986). Physical Review A, 33, 1134.
%
%   See also AMI, AMI_HISTOGRAM.

arguments
    x  (:,1) double {mustBeNonempty}
    L  (1,1) double {mustBePositive, mustBeInteger}
    opts.Fraction  (1,1) double {mustBePositive} = 0.2
    opts.ChunkSize (1,1) double {mustBePositive, mustBeInteger} = 512
end

if anynan(x)
    error('ami_kde:nanInput', ...
          'x contains NaN. Mutual information is undefined for missing data.');
end
if L >= numel(x)
    error('ami_kde:lagTooLarge', ...
          'L (%d) must be smaller than the number of samples (%d).', L, numel(x));
end

curve = zeros(L + 1, 2);
curve(1, :) = [0, mutualInformationBits(x, x, opts.ChunkSize)];
for lag = 1:L
    a = x(1:end-lag);
    b = x(lag+1:end);
    curve(lag + 1, :) = [lag, mutualInformationBits(a, b, opts.ChunkSize)];
end

[tau, info] = firstMinimum(curve, opts.Fraction);
info.estimator = "kde";
end

% ---------------------------------------------------------------- helpers

function I = mutualInformationBits(X, Y, chunk)
n = numel(X);
hx = std(X) / n^(1/6);
hy = std(Y) / n^(1/6);
if hx == 0 || hy == 0
    I = 0;
    return
end

pX = marginalDensity(X, hx, chunk);
pY = marginalDensity(Y, hy, chunk);
pXY = jointDensity(X, Y, hx, hy, chunk);

ok = pXY > 0 & pX > 0 & pY > 0;
I = sum(log2(pXY(ok) ./ (pX(ok) .* pY(ok)))) / n;
end

function p = marginalDensity(v, h, chunk)
%MARGINALDENSITY Gaussian KDE evaluated at the sample points.
%   normpdf(z) = exp(-z^2/2)/sqrt(2*pi), inlined to avoid the toolbox.
n = numel(v);
p = zeros(n, 1);
c = 1 / (n * h * sqrt(2*pi));
for lo = 1:chunk:n
    hi = min(lo + chunk - 1, n);
    z = (v(lo:hi) - v.') / h;                 % implicit expansion, no repmat
    p(lo:hi) = c * sum(exp(-0.5 * z.^2), 2);
end
end

function p = jointDensity(X, Y, hx, hy, chunk)
%JOINTDENSITY Bivariate Gaussian KDE with correlated kernel.
%
%   Covariance W = [hx^2, r*hx*hy; r*hx*hy, hy^2] where r is the sample
%   correlation of (X, Y). The closed form of the bivariate normal density is
%   used directly rather than calling mvnpdf on an n^2-row argument.
n = numel(X);
r = pearson(X, Y);
r = min(max(r, -0.999999), 0.999999);         % keep the kernel non-singular
omr2 = 1 - r^2;
c = 1 / (n * 2*pi * hx * hy * sqrt(omr2));

p = zeros(n, 1);
for lo = 1:chunk:n
    hi = min(lo + chunk - 1, n);
    u = (X(lo:hi) - X.') / hx;
    v = (Y(lo:hi) - Y.') / hy;
    q = (u.^2 - 2*r*(u.*v) + v.^2) / omr2;
    p(lo:hi) = c * sum(exp(-0.5 * q), 2);
end
end

function r = pearson(a, b)
a = a - mean(a);
b = b - mean(b);
r = (a.' * b) / (norm(a) * norm(b));
end

function [tau, info] = firstMinimum(curve, fraction)
%FIRSTMINIMUM Lag of the first strict local minimum. See ami_histogram.
v = curve(:, 2);
isMin = [false; v(2:end-1) < v(1:end-2) & v(2:end-1) <= v(3:end); false];
idx = find(isMin);

info = struct();
info.allMinima = curve(idx, :);
info.usedFallback = false;
info.fractionLag = NaN;

below = find(v <= fraction * v(1), 1, 'first');
if ~isempty(below)
    info.fractionLag = curve(below, 1);
end

if ~isempty(idx)
    tau = curve(idx(1), 1);
else
    tau = info.fractionLag;
    info.usedFallback = true;
    if isnan(tau)
        warning('ami_kde:noMinimum', ...
                ['No local minimum in AMI over lags 0:%d and the curve never fell to ' ...
                 '%.0f%% of its lag-0 value.'], curve(end, 1), 100 * fraction);
    end
end
end

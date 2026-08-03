function [tau, curve, info] = ami_kde(x, L, opts)
%AMI_KDE Average mutual information versus lag, kernel density estimator.
%   TAU = AMI_KDE(X,L) returns the lag of the first local minimum of the
%   average mutual information of X with itself over lags 0:L, using Gaussian
%   kernel density estimates of the marginal and joint densities.
%
%   [TAU,CURVE] = AMI_KDE(X,L) also returns the AMI curve as an (L+1)-by-2
%   array of [lag, ami], in bits.
%
%   [TAU,CURVE,INFO] = AMI_KDE(X,L) also returns a struct with fields
%   allMinima, usedFallback, fractionLag and estimator.
%
%   ___ = AMI_KDE(X,L,Fraction=F) sets the fallback threshold, as a fraction
%   of AMI at lag 0, used when the curve has no local minimum. Default 0.2.
%
%   ___ = AMI_KDE(X,L,ChunkSize=C) sets how many rows of the pairwise kernel
%   are evaluated at once. Default 512. Lower it to reduce peak memory; it
%   does not change the result.
%
%   Input Arguments
%      X  time series, real column vector, no NaN
%      L  maximum lag, positive integer, less than numel(X)
%
%   Notes
%   Marginals use a univariate Gaussian kernel with bandwidth std(X)/N^(1/6).
%   The joint density uses a bivariate Gaussian kernel whose covariance
%   carries the sample correlation. Densities are evaluated at the data
%   points and AMI is the mean log2 ratio.
%
%   Cost is O(L*N^2) kernel evaluations, substantially slower than
%   AMI_HISTOGRAM, in exchange for lower bias. Peak memory is O(ChunkSize*N).
%
%   Runs on base MATLAB. No toolbox required.
%
%   Examples
%      tau = ami_kde(x, 50);
%      tau = ami_kde(x, 50, ChunkSize=128);   % lower peak memory
%
%   References
%      Thomas, R. D., Moses, N. C., Semple, E. A. and Strang, A. J. (2014).
%      An efficient algorithm for the computation of average mutual
%      information. Behavior Research Methods.
%
%      Fraser, A. M. and Swinney, H. L. (1986). Independent coordinates for
%      strange attractors from mutual information. Physical Review A, 33(2),
%      1134-1140.
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
%MARGINALDENSITY Gaussian KDE at the sample points. normpdf inlined.
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
%JOINTDENSITY Bivariate Gaussian KDE, covariance [hx^2 r*hx*hy; r*hx*hy hy^2].
% Closed form rather than mvnpdf, to avoid the Statistics Toolbox.
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
%FIRSTMINIMUM Lag of the first strict local minimum. See AMI_HISTOGRAM.
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

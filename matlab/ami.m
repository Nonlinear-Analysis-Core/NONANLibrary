function [tau, curve, info] = ami(x, L, opts)
%AMI Average mutual information vs lag, for choosing an embedding delay.
%
%   [tau, curve] = AMI(x, L) returns the lag of the first local minimum of
%   the average mutual information of x with itself over lags 0:L, and the
%   full AMI curve as an (L+1)-by-2 array of [lag, ami].
%
%   [tau, curve, info] = AMI(...) also returns diagnostics: all local minima
%   found, the fallback lag, the bin count actually used, and the estimator.
%
%   AMI(x, L, Algorithm=name) selects the estimator:
%
%     "histogram"  (default)  equal-width joint histogram. Fast, base MATLAB.
%                             Alias: "stergiou" (the old file name).
%     "kde"                   Gaussian kernel density estimate.
%                             Alias: "thomas" (Thomas et al., where the code
%                             was published).
%
%   Estimator-specific options are forwarded:
%     Bins      "histogram" only. Default: Scott's rule.
%     Fraction  fallback threshold as a fraction of AMI at lag 0. Default 0.2.
%
%   WHY A WRAPPER
%   There are many ways to estimate mutual information and they disagree,
%   particularly in absolute value. Which one produced a published tau is a
%   material detail, so it belongs in an argument rather than in a file name.
%   The old layout -- AMI_Stergiou and AMI_Thomas -- named implementations
%   after people rather than after what distinguishes them, so a user could
%   not tell from the call which estimator they had used.
%
%   NOTE ON NAMING. "histogram" is deliberately not called "fraser_swinney".
%   Fraser & Swinney (1986) contributed the first-minimum criterion, which
%   both estimators use, but their algorithm is a recursive adaptive
%   partition that neither implements. Naming the fixed-bin histogram after
%   them would repeat the original mistake in the opposite direction.
%
%   TODO -- estimators to add behind this same switch:
%     "adaptive"  Fraser & Swinney's actual recursive partition (1986)
%     "knn"       Kraskov, Stogbauer & Grassberger (2004); bin-free, low bias
%     "copula"    rank-transform to uniform marginals, estimate copula
%                 density; invariant to monotone rescaling of the signal
%     "kde-adaptive"  variable-bandwidth KDE instead of a fixed rule
%
%   Example
%       x = fgn_sim(4096, 0.8);
%       [tau, curve] = ami(x, 50);
%       [tau2, ~]    = ami(x, 50, Algorithm="kde");
%
%   See also AMI_HISTOGRAM, AMI_KDE, MUTUAL_INFORMATION.

arguments
    x  (:,1) double {mustBeNonempty}
    L  (1,1) double {mustBePositive, mustBeInteger}
    opts.Algorithm (1,1) string = "histogram"
    opts.Bins      double {mustBeScalarOrEmpty, mustBePositive, mustBeInteger} = []
    opts.Fraction  (1,1) double {mustBePositive} = 0.2
end

switch lower(opts.Algorithm)

    case {"histogram", "stergiou"}
        args = {"Fraction", opts.Fraction};
        if ~isempty(opts.Bins)
            args = [args, {"Bins", opts.Bins}];
        end
        [tau, curve, info] = ami_histogram(x, L, args{:});

    case {"kde", "thomas"}
        if ~isempty(opts.Bins)
            warning('ami:binsIgnored', ...
                    'Bins applies to Algorithm="histogram" and is ignored for "kde".');
        end
        [tau, curve, info] = ami_kde(x, L, "Fraction", opts.Fraction);

    otherwise
        error('ami:unknownAlgorithm', ...
              ['Unknown Algorithm "%s". Supported: "histogram" (alias "stergiou"), ' ...
               '"kde" (alias "thomas").'], opts.Algorithm);
end

info.algorithm = lower(opts.Algorithm);
end

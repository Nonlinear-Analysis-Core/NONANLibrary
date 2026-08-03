function [tau, curve, info] = ami(x, L, opts)
%AMI Average mutual information versus lag.
%   TAU = AMI(X,L) returns the lag of the first local minimum of the average
%   mutual information of X with itself, computed over lags 0:L. TAU is the
%   embedding delay for phase space reconstruction.
%
%   [TAU,CURVE] = AMI(X,L) also returns the AMI curve as an (L+1)-by-2 array
%   whose columns are [lag, ami]. AMI is in bits.
%
%   [TAU,CURVE,INFO] = AMI(X,L) also returns a struct with fields:
%      allMinima     all local minima found, as [lag, ami] rows
%      usedFallback  true if no local minimum existed and the fraction
%                    threshold was used instead
%      fractionLag   lag at which AMI first fell below Fraction*AMI(0)
%      bins          number of bins used ("histogram" only)
%      estimator     estimator that ran
%      algorithm     value of Algorithm as supplied
%
%   ___ = AMI(X,L,Algorithm=ALG) selects the estimator:
%      "histogram"  (default) equal-width joint histogram. Fast.
%                   Aliases: "stergiou".
%      "kde"        bivariate Gaussian kernel density estimate. Slower,
%                   O(L*N^2), but less biased. Aliases: "thomas".
%
%   ___ = AMI(X,L,Bins=B) sets the number of histogram bins per axis.
%   Default selects B by Scott's rule. Ignored by "kde".
%
%   ___ = AMI(X,L,Fraction=F) sets the fallback threshold, as a fraction of
%   AMI at lag 0, used when the curve has no local minimum. Default 0.2.
%
%   Input Arguments
%      X  time series, real column vector, no NaN
%      L  maximum lag, positive integer, less than numel(X)
%
%   Notes
%   The two estimators disagree in absolute value; the histogram is biased
%   upward and the bias grows with bin count. For a bivariate Gaussian with
%   correlation r the true AMI is -0.5*log2(1-r^2); on an AR(1) series with
%   phi = 0.7 at lag 1 the exact value is 0.4857 bits, against 0.5619
%   (histogram) and 0.5411 (kde). Report which estimator produced a
%   published tau.
%
%   Examples
%      % Embedding delay for a Lorenz series
%      tau = ami(x, 50);
%
%      % Same series, kernel density estimator
%      tau = ami(x, 50, Algorithm="kde");
%
%      % Fixed bin count, and inspect the curve
%      [tau, curve] = ami(x, 50, Bins=64);
%      plot(curve(:,1), curve(:,2))
%
%   References
%      Fraser, A. M. and Swinney, H. L. (1986). Independent coordinates for
%      strange attractors from mutual information. Physical Review A, 33(2),
%      1134-1140.
%
%      Thomas, R. D., Moses, N. C., Semple, E. A. and Strang, A. J. (2014).
%      An efficient algorithm for the computation of average mutual
%      information. Behavior Research Methods.
%
%      Abarbanel, H. D. I., Brown, R., Sidorowich, J. J. and Tsimring, L. S.
%      (1993). The analysis of observed chaotic data in physical systems.
%      Reviews of Modern Physics, 65(4), 1331-1392.
%
%   See also AMI_HISTOGRAM, AMI_KDE, LYE_R, LYE_W.

% TODO: add Algorithm="adaptive" (Fraser-Swinney recursive partition),
% "knn" (Kraskov-Stogbauer-Grassberger 2004), "copula", "kde-adaptive".

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

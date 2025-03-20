function [sampen_value] = sampen(data, m, r, flag)
%SAMPEN Calculate the sample entropy of a time series.
%
%   sampen_value = sampen(data, m, r, flag)
%
%   Inputs:
%       data - A vector containing the time series data.
%       m    - Embedding dimension.
%       r    - Tolerance threshold. If flag is 'prop', then r is a proportion
%              of the standard deviation of data (e.g., 0.2 means 0.2*std(data)).
%              If flag is 'const', then r is used as the constant threshold.
%       flag - (optional) A string that specifies how to interpret r.
%              Use 'prop' (default) if r is a proportion of std(data),
%              or 'const' if r is a constant.
%
%   Output:
%       sampen_value - The computed sample entropy value.
%
%   The sample entropy is defined as:
%       SampEn = -log( A / B )
%   where:
%       B = number of pairs of vectors of length m that are similar.
%       A = number of pairs of vectors of length m+1 that are similar.
%
%   Reference:
%       Richman, J. S. & Moorman, J. R. (2000), 
%       "Physiological time-series analysis using approximate entropy and sample entropy",
%       American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
%
%   Written by Aaron D. Likens and Seung Kyeom Kim


% Ensure that the data is a column vector
data = data(:);
N = length(data);

% Check if the time series is long enough
if N < m+1
    error('The time series must have at least m+1 data points.');
end

% Set default r and flag if not provided
if nargin < 3 || isempty(r)
    r = 0.2;
end
if nargin < 4 || isempty(flag)
    flag = 'prop';
end

% If r is a proportion of std(data), update r accordingly.
if strcmpi(flag, 'prop')
    r = r * std(data);
end

%% Create embedding vectors of length m
% Each row of X is a vector of m consecutive data points.
X = zeros(N - m + 1, m);
for i = 1:(N - m + 1)
    X(i, :) = data(i:i+m-1);
end

% Count the number of similar pairs for vectors of length m (B)
B = 0;
for i = 1:size(X, 1)
    for j = i+1:size(X, 1)
        % Using Chebyshev distance: maximum absolute difference
        if max(abs(X(i,:) - X(j,:))) <= r
            B = B + 1;
        end
    end
end

%% Create embedding vectors of length m+1
X1 = zeros(N - m, m+1);
for i = 1:(N - m)
    X1(i, :) = data(i:i+m);
end

% Count the number of similar pairs for vectors of length m+1 (A)
A = 0;
for i = 1:size(X1, 1)
    for j = i+1:size(X1, 1)
        if max(abs(X1(i,:) - X1(j,:))) <= r
            A = A + 1;
        end
    end
end

% Normalize counts by the number of comparisons (excluding self-matches)
A = A/(N - m - 1);
B = B/(N - m);

fprintf("ratio is %f\n", A/B);

%% Calculate sample entropy
if B == 0
    sampen_value = Inf;
else
    sampen_value = -log(A / B);
end

end
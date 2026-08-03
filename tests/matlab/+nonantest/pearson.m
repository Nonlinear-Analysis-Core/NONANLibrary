function r = pearson(a, b)
%PEARSON Pearson correlation, base MATLAB only.
%   corr() lives in the Statistics and Machine Learning Toolbox. The harness
%   must not depend on a toolbox to check a base-MATLAB library, so this
%   two-line replacement is used throughout the suite instead.
a = double(a(:)) - mean(double(a(:)));
b = double(b(:)) - mean(double(b(:)));
r = (a' * b) / (norm(a) * norm(b));
end

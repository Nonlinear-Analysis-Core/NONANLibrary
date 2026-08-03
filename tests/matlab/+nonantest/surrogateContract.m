function r = surrogateContract(x, gen, nSurr)
%SURROGATECONTRACT Measure what a surrogate generator actually preserves.
%
%   r = nonantest.surrogateContract(x, gen, nSurr)
%
%   gen is a function handle producing ONE surrogate: @(v) Surr_Theiler(v,1).
%
%   Returns a struct of measured quantities. It asserts nothing -- the caller
%   decides which contract applies, because the contract differs by algorithm
%   and conflating them is the trap:
%
%     Algorithm 0 (shuffle) : distribution EXACT,   spectrum destroyed BY DESIGN
%     Algorithm 1 (FT)      : spectrum EXACT,       distribution Gaussianised
%     Algorithm 2 (AAFT)    : distribution EXACT,   spectrum APPROXIMATE by design
%     IAAFT                 : distribution EXACT,   spectrum close
%
%   "Exact" means machine precision, ~1e-15, not "small". An FT surrogate is
%   DEFINED by |FFT(surrogate)| == |FFT(original)| realisation by realisation.
%   A spectral error of 0.25 is not a tolerance question, it is a different
%   algorithm. Conversely, holding Algorithm 2 to exact spectral preservation
%   would be reporting its design as a defect -- approximating the spectrum is
%   precisely the weakness IAAFT was built to reduce.
%
%   Fields:
%     spectral_error       ||P(z) - P(x)|| / ||P(x)||, mean over surrogates
%     distribution_error   max|sort(z) - sort(x)| / std(x), mean
%     sd_ratio             std(z)/std(x), mean.  ~0.707 is the signature of a
%                          non-conjugate-symmetric ifft losing its imaginary part
%     acf1_original        lag-1 autocorrelation of the input
%     acf1_surrogate       lag-1 autocorrelation of the surrogates, mean
%     length_ok            every surrogate had the same length as the input
%     distinct             surrogates are not identical to each other
%     any_complex          a surrogate came back complex

if nargin < 3 || isempty(nSurr), nSurr = 19; end

x = double(x(:));
n = numel(x);
pow = @(v) abs(fft(v - mean(v))).^2;
p0  = pow(x);
s0  = sort(x);
sd0 = std(x);

specErr = zeros(nSurr,1);
distErr = zeros(nSurr,1);
sdRatio = zeros(nSurr,1);
acf1    = zeros(nSurr,1);
lenOk   = true;
anyCplx = false;
first   = [];
distinct = false;

for k = 1:nSurr
    z = gen(x);
    z = z(:);
    if numel(z) ~= n
        lenOk = false;
        specErr(k) = NaN; distErr(k) = NaN; sdRatio(k) = NaN; acf1(k) = NaN;
        continue
    end
    if ~isreal(z), anyCplx = true; z = real(z); end
    z = double(z);

    specErr(k) = norm(pow(z) - p0) / norm(p0);
    distErr(k) = max(abs(sort(z) - s0)) / sd0;
    sdRatio(k) = std(z) / sd0;
    acf1(k)    = nonantest.pearson(z(1:end-1), z(2:end));

    if isempty(first)
        first = z;
    elseif ~distinct && max(abs(z - first)) > 1e-12
        distinct = true;
    end
end

r = struct( ...
    'spectral_error',     mean(specErr, 'omitnan'), ...
    'distribution_error', mean(distErr, 'omitnan'), ...
    'sd_ratio',           mean(sdRatio, 'omitnan'), ...
    'acf1_original',      nonantest.pearson(x(1:end-1), x(2:end)), ...
    'acf1_surrogate',     mean(acf1, 'omitnan'), ...
    'length_ok',          lenOk, ...
    'distinct',           distinct, ...
    'any_complex',        anyCplx, ...
    'n_surrogates',       nSurr);
end

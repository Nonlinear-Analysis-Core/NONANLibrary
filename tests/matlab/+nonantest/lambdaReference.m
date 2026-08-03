function r = lambdaReference(kind, varargin)
%LAMBDAREFERENCE Reference largest Lyapunov exponent, with its provenance.
%
%   r = nonantest.lambdaReference('skewtent', 0.3)
%   r = nonantest.lambdaReference('lorenz')
%
%   Returns a struct:
%     r.nats     lambda1 in nats per unit time (per iteration for maps)
%     r.bits     the same value in bits, = nats/ln(2). LyE_W accumulates
%                log2 and therefore reports bits.
%     r.tier     "exact" | "numerical"
%     r.source   where the value comes from
%
%   TIER MATTERS AND IS RETURNED DELIBERATELY. A test asserting agreement
%   with an exact value can be tight; one asserting agreement with a
%   well-converged numerical value for a flow cannot, because the estimator's
%   own estimand differs from the system's lambda by an amount that depends
%   on sampling rate, embedding, series length and scaling region. Callers
%   should branch on r.tier rather than hard-coding a tolerance and forgetting
%   which kind of number they are comparing against.

switch lower(kind)

    case 'skewtent'
        % THEOREM. The skew tent map has uniform invariant density on [0,1]
        % and |f'| equal to 1/p below the kink and 1/(1-p) above it, so
        %     lambda = -p ln p - (1-p) ln(1-p)
        % which is the binary entropy of p in nats. Nothing numerical enters.
        p = 0.3;
        if ~isempty(varargin), p = varargin{1}; end
        r.nats = -p*log(p) - (1-p)*log(1-p);
        r.tier = "exact";
        r.source = sprintf('analytic: -p ln p - (1-p) ln(1-p), p = %g', p);

    case 'logistic'
        % THEOREM. At r = 4 the logistic map is conjugate to the symmetric
        % tent map through x = sin^2(pi*y/2). Conjugacy preserves Lyapunov
        % exponents, and the tent map with slope 2 has lambda = ln 2.
        r.nats = log(2);
        r.tier = "exact";
        r.source = 'analytic: conjugate to the tent map, lambda = ln 2';

    case 'lorenz'
        r.nats = 0.9056;
        r.tier = "numerical";
        r.source = 'Sprott (2003) Appendix A, sigma=10 rho=28 beta=8/3';

    case 'henon'
        r.nats = 0.41922;
        r.tier = "numerical";
        r.source = 'Sprott (2003) Appendix A, a=1.4 b=0.3';

    case 'rossler'
        r.nats = 0.0714;
        r.tier = "numerical";
        r.source = 'Sprott (2003) Appendix A, a=0.2 b=0.2 c=5.7';

    otherwise
        error('nonantest:lambdaReference:unknown', ...
              'no reference exponent for "%s"', kind);
end

r.bits = r.nats / log(2);
end

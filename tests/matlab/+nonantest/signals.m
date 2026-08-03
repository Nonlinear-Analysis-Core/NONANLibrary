function y = signals(kind, n, varargin)
%SIGNALS Deterministic test signals with known analytic answers.
%
%   y = nonantest.signals(kind, n, ...) returns an n-by-1 column vector.
%
%   Every generator here is implemented from scratch in base MATLAB and is
%   deliberately INDEPENDENT of the NONAN library. Known-answer tests are
%   worthless if the signal and the estimator share an implementation: if
%   fgn_sim were used to test dfa, a matched pair of errors would cancel and
%   the test would pass. These generators are the reference; NONAN's own
%   generators (fgn_sim) are themselves things under test.
%
%   kind                     known answer
%   ------------------------ ---------------------------------------------
%   'white'                  DFA alpha = 0.5,  Higuchi D = 2.0
%   'brown'                  DFA alpha = 1.5,  Higuchi D = 1.5
%   'fgn',H                  DFA alpha = H
%   'sine',period            Higuchi D -> 1.0, deterministic
%   'ar1',phi                lag-1 ACF = phi, linear Gaussian (null is TRUE)
%   'skewed'                 lognormal marginal, skewness ~ 3.4
%   'pseudoperiodic'         noisy cycle, the use case for Surr_PseudoPeriodic
%   'withzero'               contains an exact 0.0 (EMD regression case)
%
%   LYAPUNOV BENCHMARKS, IN TWO TIERS
%
%   Tier 1 -- lambda is a THEOREM. No integrator, no step size, no initial
%   condition, nothing numerical to be wrong about:
%
%   'skewtent',p             lambda = -p*ln(p) - (1-p)*ln(1-p) nats/iter,
%                            EXACTLY. Uniform invariant density, |f'| is 1/p
%                            or 1/(1-p). Varying p gives a LADDER of exact
%                            values rather than a single point, which tests
%                            for proportional bias rather than one number.
%                            AVOID p = 0.5: the symmetric tent is exactly a
%                            binary shift, so in floating point it exhausts
%                            the mantissa and the orbit collapses to exactly
%                            0 after ~50 iterations. Use p = 0.3, 0.4, ...
%   'logistic'               lambda = ln(2) EXACTLY at r = 4, by the
%                            conjugacy x = sin^2(pi*y/2) to the tent map.
%
%   Tier 2 -- lambda is a well-converged NUMERICAL result at standard
%   parameters, from Sprott (2003) Appendix A, which catalogues 62 systems.
%   Not a theorem, and legitimate to use provided the standard control
%   parameters are used and the value is labelled for what it is:
%
%   'lorenz'                 lambda1 = 0.9056  (sigma 10, rho 28, beta 8/3)
%   'henon'                  lambda1 = 0.41922 (a 1.4, b 0.3)
%   'rossler'                lambda1 = 0.0714  (a 0.2, b 0.2, c 5.7)
%
%   Why tier 2 is not circular: Sprott's own table contains cases whose
%   answer is known analytically, and reproduces them. His logistic entry is
%   0.693147181 against ln(2) = 0.6931471806 -- agreement to nine significant
%   figures. A method that recovers the exact answer where one exists is
%   evidence about the method, not an assumption about it.
%
%   Both tiers still measure a property of the SYSTEM. Neither is the
%   estimand of a finite-sample delay-embedding estimator, which also depends
%   on sampling rate, tau, dim, N, and the scaling region. Treat tier 2 as an
%   order-of-magnitude and sign check, and put the weight on tier 1 and on
%   invariances (scale invariance, ordering) that hold with no reference
%   value at all.
%
%   The RNG is seeded per call from a fixed base so the suite is reproducible
%   across machines and MATLAB versions ('twister' is pinned explicitly).

seed = 20260727;
extra = [];
if ~isempty(varargin), extra = varargin{1}; end
if numel(varargin) >= 2, seed = varargin{2}; end

switch lower(kind)

    case 'white'
        y = localRandn(n, seed);

    case 'brown'
        y = cumsum(localRandn(n, seed));

    case 'fgn'
        % Fractional Gaussian noise by spectral synthesis. Builds a series whose
        % power spectrum follows f^(1-2H) and whose DFA alpha is therefore H.
        % Circulant embedding at 4x length, then truncate, to keep the periodic
        % wrap-around out of the returned segment.
        H = extra;
        m = 2^nextpow2(4 * n);
        f = (1:m/2)' / m;
        amp = f .^ (-(2*H - 1) / 2);
        rng(seed, 'twister');
        ph = 2*pi*rand(m/2, 1);
        half = amp .* exp(1i * ph);
        spec = [0; half; conj(flipud(half(1:end-1)))];   % conjugate symmetric
        z = real(ifft(spec));
        y = z(1:n);
        y = (y - mean(y)) / std(y);

    case 'sine'
        period = 25;
        if ~isempty(extra), period = extra; end
        y = sin(2*pi*(0:n-1)' / period);

    case 'lorenz'
        % Classic parameters, RK4, dt small enough that the discretisation
        % error does not move lambda1 at the second decimal place. Transient
        % of 5000 steps discarded, then decimated to dt_sample = 0.03.
        sigma = 10; rho = 28; beta = 8/3;
        dt = 0.003; skip = 10;
        f = @(v) [sigma*(v(2)-v(1)); v(1)*(rho-v(3))-v(2); v(1)*v(2)-beta*v(3)];
        v = [1; 1; 1];
        for i = 1:5000, v = rk4(f, v, dt); end
        y = zeros(n, 1);
        for i = 1:n
            for j = 1:skip, v = rk4(f, v, dt); end
            y(i) = v(1);
        end

    case 'henon'
        a = 1.4; b = 0.3;
        v = [0.1; 0.1];
        for i = 1:1000, v = [1 - a*v(1)^2 + v(2); b*v(1)]; end
        y = zeros(n, 1);
        for i = 1:n
            v = [1 - a*v(1)^2 + v(2); b*v(1)];
            y(i) = v(1);
        end

    case 'skewtent'
        % lambda = -p*ln(p) - (1-p)*ln(1-p) nats/iteration, exactly.
        p = 0.3;
        if ~isempty(extra), p = extra; end
        if abs(p - 0.5) < 1e-12
            error('nonantest:signals:dyadicTent', ...
                ['p = 0.5 is the symmetric tent map, which is exactly a binary ' ...
                 'shift. In floating point the orbit exhausts the mantissa and ' ...
                 'collapses to exactly 0 after ~50 iterations. Use a ' ...
                 'non-dyadic p such as 0.3.']);
        end
        x0 = 0.3141592653589793;      % irrational-ish, not a dyadic rational
        v = x0;
        for i = 1:500                  % transient onto the invariant measure
            if v < p, v = v/p; else, v = (1-v)/(1-p); end
        end
        y = zeros(n, 1);
        for i = 1:n
            if v < p, v = v/p; else, v = (1-v)/(1-p); end
            y(i) = v;
        end

    case 'logistic'
        % r = 4: lambda = ln(2) exactly, via conjugacy to the tent map.
        v = 0.1234567;
        for i = 1:500, v = 4*v*(1-v); end
        y = zeros(n, 1);
        for i = 1:n
            v = 4*v*(1-v);
            y(i) = v;
        end

    case 'rossler'
        % a = 0.2, b = 0.2, c = 5.7. lambda1 = 0.0714 (Sprott 2003).
        % Much slower than Lorenz, so a longer sample interval is used.
        a = 0.2; b = 0.2; c = 5.7;
        dt = 0.01; skip = 10;          % dt_sample = 0.1
        f = @(v) [-v(2) - v(3); v(1) + a*v(2); b + v(3)*(v(1) - c)];
        v = [1; 1; 1];
        for i = 1:5000, v = rk4(f, v, dt); end
        y = zeros(n, 1);
        for i = 1:n
            for j = 1:skip, v = rk4(f, v, dt); end
            y(i) = v(1);
        end

    case 'ar1'
        phi = 0.7;
        if ~isempty(extra), phi = extra; end
        e = localRandn(n + 500, seed);
        y = filter(1, [1 -phi], e);
        y = y(501:end);

    case 'skewed'
        y = exp(0.9 * localRandn(n, seed));

    case 'pseudoperiodic'
        t = (0:n-1)';
        y = sin(2*pi*t/25) + 0.35*sin(2*pi*t/8) + 0.05*localRandn(n, seed);

    case 'withzero'
        % A sine sampled so that t=0 gives an exact 0.0. This is the class of
        % series that silently breaks python/emd.py's extr().
        y = sin(2*pi*(0:n-1)' / 32) + 0.3*sin(2*pi*(0:n-1)' / 7);
        y(1) = 0.0;

    otherwise
        error('nonantest:signals:unknownKind', 'unknown signal kind "%s"', kind);
end

y = y(:);
end

function x = localRandn(n, seed)
rng(seed, 'twister');
x = randn(n, 1);
end

function v = rk4(f, v, dt)
k1 = f(v);
k2 = f(v + dt/2*k1);
k3 = f(v + dt/2*k2);
k4 = f(v + dt*k3);
v = v + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end

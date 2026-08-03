function R = lye_benchmark(opts)
%LYE_BENCHMARK Evaluate LyE_R and LyE_W over Sprott (2003) Appendix A.
%
%   R = nonantest.lye_benchmark()
%   R = nonantest.lye_benchmark(N=4000, Verbose=true)
%
%   Runs both estimators over every usable system in the catalogue and
%   returns a table with, per system and per estimator:
%
%     expected     Sprott's lambda, NATS per unit time
%     wolf / ros   observed estimate, converted to NATS
%     *Diff        observed - expected, SIGNED, so the direction of the error
%                  is visible rather than averaged away
%     *AbsDiff     |observed - expected|
%     *Ratio       observed / expected, 1.00 = agreement
%
%   Both an absolute difference and a ratio are reported because neither is
%   sufficient alone. The reference exponents span 0.0064 (Henon
%   area-preserving) to 8.87 (linear congruential), a factor of 1400, so a
%   mean difference is dominated by the large-lambda systems and a mean ratio
%   by the small-lambda ones. Where the two disagree for a given estimator,
%   that disagreement is the finding.
%
%   PROTOCOL, IDENTICAL FOR EVERY SYSTEM.
%     - series from nonantest.sprott_series (uniform automatic sampling:
%       flows decimated so the dominant period is ~40 samples)
%     - tau  = first minimum of AMI, capped, via ami() where available and a
%              first-zero-crossing of the autocorrelation as fallback
%     - dim  = 5 for flows, 2 for 1-D maps, 3 for 2-D maps
%     - LyE_W: evolve = 5 (maps) or 10 (flows); output converted bits -> nats
%     - LyE_R: divergence curve fitted over an AUTOMATICALLY selected scaling
%              region (nonantest.scaling_region): longest near-linear window
%              before saturation. A fixed window cannot serve both maps and
%              flows -- a map's curve saturates within ~13 samples, so any
%              window starting at 5 and running to 50 sits on the plateau and
%              returns a slope near zero, which reads in a results table as an
%              estimator failure when it is a window failure.
%
%   Nothing is tuned per system. Hand-tuning would let a weak estimator be
%   rescued case by case, which is what a benchmark exists to prevent. The
%   cost is that a system where the protocol is a poor fit shows up as a
%   failure; that is the intended trade and such cases are reported, not
%   hidden.
%
%   UNITS. LyE_W accumulates log2 and returns bits; the appendix is base-e.
%   Everything below is converted to NATS before comparison.
%
%   LyE_R does not return an exponent at all -- it returns the average line
%   divergence curve, and the slope over a chosen scaling region is the
%   estimate. The window choice dominates the result, so it is fixed and
%   stated here. Treat LyE_R numbers as "this estimator under this window",
%   not as the method's best achievable answer.

arguments
    opts.N       (1,1) double = 4000
    opts.Verbose (1,1) logical = true
end

c = nonantest.sprott_catalog();
c = c([c.usable]);

n = numel(c);
name = strings(n,1); section = strings(n,1); category = strings(n,1);
tier = strings(n,1); expected = zeros(n,1);
wolf = nan(n,1); ros = nan(n,1);
wolfDiff = nan(n,1); rosDiff = nan(n,1);
wolfAbsDiff = nan(n,1); rosAbsDiff = nan(n,1);
wolfRatio = nan(n,1); rosRatio = nan(n,1);
tauUsed = nan(n,1); dimUsed = nan(n,1); note = strings(n,1);

if opts.Verbose
    fprintf('\n%s\n', repmat('=',1,104));
    fprintf('LyE benchmark against Sprott (2003) Appendix A. All values in NATS ' + ...
            "per unit time.\n");
    fprintf(['diff = observed - expected (signed, so the direction of the ' ...
             'error is visible)\n']);
    fprintf('ratio = observed / expected (1.00 = agreement)\n');
    fprintf('%s\n', repmat('=',1,104));
    fprintf('%-24s %-5s %9s | %9s %9s %8s %6s | %9s %9s %8s %6s\n', ...
        'system','tier','expected', ...
        'W obs','W diff','W |diff|','W r', ...
        'R obs','R diff','R |diff|','R r');
    fprintf('%s\n', repmat('-',1,104));
end

for i = 1:n
    s = c(i);
    name(i) = s.name; section(i) = s.section;
    category(i) = s.category; tier(i) = s.tier; expected(i) = s.lambda;

    try
        [y, gi] = nonantest.sprott_series(s, opts.N);
        if gi.degenerate
            note(i) = "series degenerate: " + gi.reason;
            continue
        end

        isMap = s.kind == "map";
        tau = pickTau(y, isMap);
        dim = 5; if isMap, dim = 3; end
        tauUsed(i) = tau; dimUsed(i) = dim;
        fs = gi.fs;

        ev = 10; if isMap, ev = 5; end
        try
            [~, Lw] = LyE_W(y, fs, tau, dim, ev);
            wolf(i) = Lw * log(2);                 % bits -> nats
        catch ME
            note(i) = note(i) + " wolf:" + string(ME.message);
        end

        try
            out = LyE_R(y, fs, tau, dim);
            if size(out,2) < 3
                note(i) = note(i) + " rosen: LyE_R returned 2 columns";
            else
                [sl, ~, si] = nonantest.scaling_region(out(:,3), fs);
                ros(i) = sl;
                if ~si.ok
                    note(i) = note(i) + " rosen: no clean scaling region";
                end
            end
        catch ME
            note(i) = note(i) + " rosen:" + string(ME.message);
        end

    catch ME
        note(i) = "generation failed: " + string(ME.message);
    end

    wolfDiff(i)    = wolf(i) - expected(i);
    rosDiff(i)     = ros(i)  - expected(i);
    wolfAbsDiff(i) = abs(wolfDiff(i));
    rosAbsDiff(i)  = abs(rosDiff(i));
    wolfRatio(i)   = wolf(i) / expected(i);
    rosRatio(i)    = ros(i)  / expected(i);

    if opts.Verbose
        fprintf('%-24s %-5s %9.4f | %9.4f %+9.4f %8.4f %6.2f | %9.4f %+9.4f %8.4f %6.2f', ...
            name(i), tier(i), expected(i), ...
            wolf(i), wolfDiff(i), wolfAbsDiff(i), wolfRatio(i), ...
            ros(i),  rosDiff(i),  rosAbsDiff(i),  rosRatio(i));
        if strlength(note(i)) > 0
            fprintf('  <- %s', extractBefore(note(i) + " ", min(70, strlength(note(i))+1)));
        end
        fprintf('\n');
    end
end

R = table(name, section, category, tier, expected, ...
          wolf, wolfDiff, wolfAbsDiff, wolfRatio, ...
          ros,  rosDiff,  rosAbsDiff,  rosRatio, ...
          tauUsed, dimUsed, note);

if opts.Verbose
    summarise(R);
end
end

% ---------------------------------------------------------------- reporting

function summarise(R)
%SUMMARISE Per-estimator and per-category accuracy.
%
% Both a difference and a ratio are reported because neither alone is
% adequate here. Reference exponents span 0.0064 (Henon area-preserving) to
% 8.87 (linear congruential), a factor of 1400, so a raw difference is
% dominated by the large-lambda systems and a ratio is dominated by the
% small-lambda ones. Median absolute difference says how far off in nats;
% median ratio says how far off proportionally. A method can look good on one
% and bad on the other, and that disagreement is itself informative.

ok = R(strlength(R.note) == 0, :);
fprintf('\n%s\n', repmat('=',1,104));
fprintf('SUMMARY  (%d of %d systems scored; %d excluded by protocol guards)\n', ...
    height(ok), height(R), height(R) - height(ok));
fprintf('%s\n', repmat('=',1,104));

fprintf('\n%-24s %4s | %9s %9s %8s %8s | %9s %9s %8s %8s\n', ...
    'category','n', ...
    'W med|d|','W med r','W<=25%','W<=50%', ...
    'R med|d|','R med r','R<=25%','R<=50%');
fprintf('%s\n', repmat('-',1,104));

cats = unique(ok.category, 'stable');
for k = 1:numel(cats)
    sub = ok(ok.category == cats(k), :);
    printRow(char(cats(k)), sub);
end
fprintf('%s\n', repmat('-',1,104));
printRow('ALL', ok);

ex = ok(ok.tier == "exact", :);
if height(ex) > 0
    fprintf('%s\n', repmat('-',1,104));
    printRow('exact-lambda only', ex);
end
fprintf('\n');
end

function printRow(label, T)
w = T.wolfRatio(isfinite(T.wolfRatio));
r = T.rosRatio(isfinite(T.rosRatio));
wd = T.wolfAbsDiff(isfinite(T.wolfAbsDiff));
rd = T.rosAbsDiff(isfinite(T.rosAbsDiff));
pc = @(v, p) 100 * nnz(abs(v - 1) <= p) / max(1, numel(v));
fprintf('%-24s %4d | %9.4f %9.2f %7.0f%% %7.0f%% | %9.4f %9.2f %7.0f%% %7.0f%%\n', ...
    label, height(T), ...
    median(wd), median(w), pc(w, 0.25), pc(w, 0.50), ...
    median(rd), median(r), pc(r, 0.25), pc(r, 0.50));
end

% ---------------------------------------------------------------- helpers

function tau = pickTau(y, isMap)
%PICKTAU First AMI minimum where available, else first ACF zero crossing.
if isMap
    tau = 1;                        % maps are already maximally decorrelated
    return
end
tau = [];
if exist('ami', 'file') == 2
    try
        tau = ami(y, min(50, floor(numel(y)/10)));
    catch
        tau = [];
    end
end
if isempty(tau) || ~isfinite(tau) || tau < 1
    tau = acfFirstZero(y);
end
tau = max(1, min(round(tau), 25));
end

function k = acfFirstZero(y)
y = y - mean(y);
mx = min(100, floor(numel(y)/4));
for k = 1:mx
    r = (y(1:end-k)' * y(k+1:end)) / (norm(y(1:end-k)) * norm(y(k+1:end)));
    if r <= 0
        return
    end
end
k = max(1, round(mx/4));
end

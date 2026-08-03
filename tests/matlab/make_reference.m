function make_reference()
%MAKE_REFERENCE Record MATLAB's answers on the shared fixtures.
%
%   matlab -batch "addpath('tests/matlab'); make_reference"
%
%   Writes tests/fixtures/matlab_reference.json, which IS COMMITTED. The
%   Python suite compares against this file, so cross-language equivalence is
%   checked in CI without needing a MATLAB licence on the runner.
%
%   Regenerate deliberately, never as part of a test run: if a test could
%   rewrite its own expected values, a port that drifted would simply update
%   the reference and keep passing. Fixtures and reference are inputs to the
%   tests, not outputs of them.
%
%   Only DETERMINISTIC quantities go in here. Anything driven by rand() cannot
%   be compared value-by-value across languages -- MATLAB and NumPy do not
%   share a generator -- and is tested by contract instead (see
%   nonantest.surrogateContract), not by equality.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'matlab'));
addpath(here);
dbclear all

fx = fullfile(repo, 'tests', 'fixtures');
ref = struct();

series = {'white_4096', 'fgn_H30_4096', 'fgn_H50_4096', 'fgn_H85_4096'};
for i = 1:numel(series)
    y = readmatrix(fullfile(fx, [series{i} '.csv']));
    n = numel(y);
    sc = unique(round(logspace(log10(16), log10(floor(n/8)), 18)));
    [~, fl, alpha] = dfa(y, sc, 1, false);
    ref.dfa.(series{i}) = struct( ...
        'scales', sc, 'fluctuation', fl(:)', 'alpha', alpha);
end

y = readmatrix(fullfile(fx, 'ar1_phi70_512.csv'));
ref.ent_samp.ar1_phi70_512 = Ent_Samp(y, 2, 0.2);

y = readmatrix(fullfile(fx, 'lorenz_2048.csv'));
ref.ami_thomas.lorenz_2048 = localFirstMinAmi(y);

ref.generated_by = version;
out = fullfile(fx, 'matlab_reference.json');
fid = fopen(out, 'w');
fprintf(fid, '%s', jsonencode(ref, 'PrettyPrint', true));
fclose(fid);
fprintf('wrote %s\n', out);
end

function v = localFirstMinAmi(y)
try
    a = AMI_Thomas(y, 20);
    v = a(1, 1);
catch ME
    v = NaN;
    fprintf('AMI_Thomas unavailable for reference: %s\n', ME.message);
end
end

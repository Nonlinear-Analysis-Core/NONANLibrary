function run_tests(varargin)
%RUN_TESTS Headless entry point for the NONAN MATLAB test suite.
%
%   From a shell, at the repository root:
%       matlab -batch "addpath('tests/matlab'); run_tests"
%
%   Optional name filter (substring match on test name):
%       matlab -batch "addpath('tests/matlab'); run_tests('Surr')"
%
%   Exits with status 0 if every test passed, 1 otherwise, so it can be wired
%   straight into CI. Writes JUnit XML to tests/artifacts/results.xml.
%
%   DESIGN CONSTRAINTS (these are requirements, not preferences)
%   - Base MATLAB only. No Statistics, Signal Processing, or Image Processing
%     toolbox in the harness itself. A test that needs a toolbox must call
%     nonantest.requireToolbox and be skipped, not error, when it is absent.
%   - No figures. Any test that leaves a figure open fails: the library is used
%     on clusters and in -batch runs where a figure is a hang or a crash.
%   - No interactive debugger. Several NONAN functions execute `dbstop if error`
%     at load, which is global session state and turns any later uncaught error
%     into an indefinite halt under `matlab -batch`. The runner clears it before
%     and after every test.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));

addpath(fullfile(repo, 'matlab'));
addpath(here);

% The library arms the debugger as a side effect of being called. Under -batch
% that converts an error into a hang rather than a failure, so the runner must
% start from a known-clean state.
dbclear all

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.TestReportPlugin
import matlab.unittest.plugins.XMLPlugin

suite = TestSuite.fromFolder(here);
if nargin >= 1 && ~isempty(varargin{1})
    suite = suite.selectIf(matlab.unittest.selectors.HasName( ...
        matlab.unittest.constraints.ContainsSubstring(varargin{1})));
end

artifacts = fullfile(repo, 'tests', 'artifacts');
if ~exist(artifacts, 'dir'), mkdir(artifacts); end

runner = TestRunner.withTextOutput('OutputDetail', 3);
runner.addPlugin(XMLPlugin.producingJUnitFormat(fullfile(artifacts, 'results.xml')));

results = runner.run(suite);

fprintf('\n================ NONAN test summary ================\n');
fprintf('  passed  %d\n', nnz([results.Passed]));
fprintf('  failed  %d\n', nnz([results.Failed]));
fprintf('  skipped %d\n', nnz([results.Incomplete]));
fprintf('  time    %.1f s\n', sum([results.Duration]));
fprintf('====================================================\n');

failed = [results.Failed];
if any(failed)
    fprintf('\nFailing tests:\n');
    names = {results(failed).Name};
    for k = 1:numel(names)
        fprintf('  - %s\n', names{k});
    end
end

dbclear all
exit(double(any(failed)));
end

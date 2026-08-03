function tests = testHeadless
%TESTHEADLESS Structural sweep: constructs that make the library unusable in CI.
%
%   These are source-level tests, not numerical ones. They exist because the
%   failure they catch is invisible to any test that calls a function inside
%   try/catch -- MATLAB does not break into the debugger for errors inside a
%   try block, so `dbstop if error` is undetectable from a passing test suite
%   and lethal to a user script.
%
%   Verified behaviour, not inference: `matlab -batch` DOES honour
%   `dbstop if error`. A script that arms it and then raises an uncaught error
%   hangs forever rather than exiting. This was measured, not assumed.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
here = fileparts(mfilename('fullpath'));
tc.TestData.repo = fileparts(fileparts(here));
tc.TestData.mfiles = localShippedMFiles(tc.TestData.repo);
end

function testGrepItselfWorks(tc)
% A source-scanning test that matches nothing passes silently and looks like
% good news. This one asserts the scanner can find something that is
% definitely there, so a broken pattern or a broken file list shows up as a
% failure instead of a clean bill of health.
tc.verifyNotEmpty(tc.TestData.mfiles, 'no .m files were found to scan');
tc.verifyGreaterThan(numel(tc.TestData.mfiles), 20, ...
    sprintf('only %d .m files found; expected ~26', numel(tc.TestData.mfiles)));

% 'function' appears in every one of them.
sanity = localGrep(tc.TestData.mfiles, '^\s*\<function\>');
tc.verifyGreaterThanOrEqual(numel(sanity), numel(tc.TestData.mfiles), ...
    'localGrep failed to find "function" in every file -- the scanner is broken.');

% And the word-boundary syntax actually behaves. MATLAB regexp reads \b as a
% backspace, not a word boundary; \< and \> are the MATLAB spellings.
tc.verifyNotEmpty(regexp('dbstop if error', '^\s*\<dbstop\>', 'once'), ...
    'word-boundary syntax is not matching -- check for \b vs \< in the patterns.');
end

function testNoDbstopInShippedCode(tc)
% NOTE: MATLAB regexp does NOT support \b as a word boundary -- it is a
% backspace, and '^\s*dbstop\b' silently matches nothing. Use \< and \>.
% The first version of this test passed against eight real violations.
hits = localGrep(tc.TestData.mfiles, '^\s*\<dbstop\>');
tc.verifyEmpty(hits, sprintf( ...
    ['`dbstop if error` found in %d shipped file(s):\n%s\n' ...
     'This is global session state, not a local setting. Once any of these\n' ...
     'runs, every subsequent uncaught error in the session -- including in\n' ...
     'the user''s own code -- halts in the debugger. Under `matlab -batch`,\n' ...
     'on a cluster, or in CI there is nothing to halt into and the process\n' ...
     'hangs until killed. It also cannot be cleared by the caller before the\n' ...
     'fact, because the library re-arms it on every call.'], ...
    numel(hits), localFormat(hits)));
end

function testNoBlockingUiInShippedCode(tc)
hits = localGrep(tc.TestData.mfiles, '\<(waitbar|questdlg|msgbox|inputdlg|uiwait|keyboard)\s*\(');
tc.verifyEmpty(hits, sprintf( ...
    ['Blocking or GUI-only call found in %d location(s):\n%s\n' ...
     'These require a display. On a headless node waitbar either errors or\n' ...
     'silently accumulates handles; the others block forever waiting for a\n' ...
     'user who is not there.'], numel(hits), localFormat(hits)));
end

function testNoUnguardedConsoleWrites(tc)
% display() and a bare disp of progress text make batch logs unreadable and
% cannot be suppressed by the caller. Flag the ones inside search loops.
hits = localGrep(tc.TestData.mfiles, '^\s*display\s*\(');
tc.verifyEmpty(hits, sprintf( ...
    ['`display(...)` used for progress output in %d location(s):\n%s\n' ...
     'It cannot be silenced and is deprecated in favour of disp/fprintf.'], ...
    numel(hits), localFormat(hits)));
end

function testFileNameMatchesFunctionName(tc)
bad = {};
for i = 1:numel(tc.TestData.mfiles)
    f = tc.TestData.mfiles{i};
    [~, base] = fileparts(f);
    name = localFirstFunctionName(f);
    if ~isempty(name) && ~strcmp(name, base)
        bad{end+1} = sprintf('  %s declares function "%s"', ...
            localRel(tc.TestData.repo, f), name); %#ok<AGROW>
    end
end
tc.verifyEmpty(bad, sprintf( ...
    ['File and function name disagree in %d file(s):\n%s\n' ...
     'MATLAB dispatches on the FILE name, so the declared name is dead and\n' ...
     'misleading: `help` shows one name, callers must type another, and any\n' ...
     'editor "go to definition" or refactoring tool is wrong. It also breaks\n' ...
     'the documented public API if the declared name is what users read.'], ...
    numel(bad), strjoin(bad, newline)));
end

function testNoCrOnlyLineEndings(tc)
bad = {};
for i = 1:numel(tc.TestData.mfiles)
    f = tc.TestData.mfiles{i};
    fid = fopen(f, 'r'); raw = fread(fid, Inf, '*uint8'); fclose(fid);
    nCR = nnz(raw == 13);
    nLF = nnz(raw == 10);
    if nCR > 0 && nLF == 0
        bad{end+1} = sprintf('  %s (%d CR, 0 LF)', localRel(tc.TestData.repo, f), nCR); %#ok<AGROW>
    end
end
tc.verifyEmpty(bad, sprintf( ...
    ['Classic-Mac CR-only line endings in %d file(s):\n%s\n' ...
     'Every line-oriented tool sees these as a single line: git diff is\n' ...
     'useless, code review is impossible, grep matches the whole file, and\n' ...
     'the MATLAB editor may reflow them on first save producing a diff that\n' ...
     'touches every line.'], numel(bad), strjoin(bad, newline)));
end

% ------------------------------------------------------------------ helpers

function files = localShippedMFiles(repo)
% Shipped code only. archive/ is explicitly excluded -- it is not on the
% user's path and holding it to the same standard would be noise.
files = {};
for d = {'matlab', '.'}
    p = fullfile(repo, d{1});
    listing = dir(fullfile(p, '*.m'));
    for k = 1:numel(listing)
        if ~listing(k).isdir
            files{end+1} = fullfile(listing(k).folder, listing(k).name); %#ok<AGROW>
        end
    end
end
files = unique(files);
end

function hits = localGrep(files, pattern)
hits = {};
for i = 1:numel(files)
    lines = localReadLines(files{i});
    for k = 1:numel(lines)
        stripped = regexprep(lines{k}, '%.*$', '');   % ignore comments
        if ~isempty(regexp(stripped, pattern, 'once'))
            hits{end+1} = struct('file', files{i}, 'line', k, ...
                'text', strtrim(lines{k})); %#ok<AGROW>
        end
    end
end
if ~isempty(hits), hits = [hits{:}]; end
end

function lines = localReadLines(f)
fid = fopen(f, 'r'); raw = fread(fid, Inf, '*char')'; fclose(fid);
raw = regexprep(raw, '\r\n?', newline);      % normalise CRLF and CR-only
% CollapseDelimiters must be false: the default true silently drops blank
% lines, which shifts every reported line number after the first blank.
lines = strsplit(raw, newline, 'CollapseDelimiters', false);
end

function name = localFirstFunctionName(f)
name = '';
lines = localReadLines(f);
for k = 1:numel(lines)
    tok = regexp(lines{k}, '^\s*function\s+(?:.*?=\s*)?([A-Za-z]\w*)\s*[\(\s]', 'tokens', 'once');
    if ~isempty(tok)
        name = tok{1};
        return
    end
end
end

function s = localFormat(hits)
if isempty(hits), s = ''; return; end
parts = arrayfun(@(h) sprintf('  %s:%d:  %s', h.file, h.line, h.text), ...
    hits, 'UniformOutput', false);
s = strjoin(parts, newline);
end

function r = localRel(repo, f)
r = strrep(f, [repo filesep], '');
end

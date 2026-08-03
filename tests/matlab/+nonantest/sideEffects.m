function s = sideEffects(fn)
%SIDEEFFECTS Run fn and report the environment damage it caused.
%
%   s = nonantest.sideEffects(@() SomeFunction(args))
%
%   A numerically correct function can still be unusable on a cluster. This
%   records the three ways NONAN functions currently break headless execution:
%
%     s.figures    number of figures left open (each one is a resource leak,
%                  and on a display-less node a hard error)
%     s.dbstop     true if the call armed the debugger. `dbstop if error` is
%                  GLOBAL SESSION STATE, not local to the function: once any
%                  NONAN function has run, every later uncaught error in that
%                  session -- including in the caller's own code -- drops into
%                  the debugger. Under `matlab -batch` there is no terminal to
%                  drop into and the process hangs until it is killed.
%     s.errored    true if fn threw; s.err holds the exception
%     s.seconds    wall time
%
%   fn is wrapped in try/catch, which also suppresses any dbstop the callee
%   arms -- MATLAB does not break into the debugger for errors inside a try
%   block. That is exactly why the suite can test these functions at all, and
%   exactly why an ordinary user script cannot.

before = findall(0, 'Type', 'figure');
dbclear all

s = struct('figures', 0, 'dbstop', false, 'errored', false, ...
           'err', [], 'seconds', NaN, 'value', []);

t = tic;
try
    if nargout(fn) == 0
        fn();
    else
        s.value = fn();
    end
catch ME
    s.errored = true;
    s.err = ME;
end
s.seconds = toc(t);

st = dbstatus;
s.dbstop = any(strcmp({st.cond}, 'error'));

after = findall(0, 'Type', 'figure');
opened = setdiff(after, before);
s.figures = numel(opened);

close(opened(ishandle(opened)));
dbclear all
end

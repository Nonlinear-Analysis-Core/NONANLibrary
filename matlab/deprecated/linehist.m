function varargout = linehist(varargin)
%LINEHIST (deprecated) Use line_hist instead.
%   linehist has been renamed to line_hist. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','linehist:deprecated')
%
%   See also LINE_HIST.

warning('linehist:deprecated', ...
    'linehist has been renamed to line_hist. Update your code; this shim will be removed.');
[varargout{1:nargout}] = line_hist(varargin{:});
end

function varargout = LyE_R(varargin)
%LYE_R (deprecated) Use lye_r instead.
%   LyE_R has been renamed to lye_r. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','LyE_R:deprecated')
%
%   See also LYE_R.

warning('LyE_R:deprecated', ...
    'LyE_R has been renamed to lye_r. Update your code; this shim will be removed.');
[varargout{1:nargout}] = lye_r(varargin{:});
end

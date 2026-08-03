function varargout = MDRQA021525(varargin)
%MDRQA021525 (deprecated) Use mdrqa instead.
%   MDRQA021525 has been renamed to mdrqa. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','MDRQA021525:deprecated')
%
%   See also MDRQA.

warning('MDRQA021525:deprecated', ...
    'MDRQA021525 has been renamed to mdrqa. Update your code; this shim will be removed.');
[varargout{1:nargout}] = mdrqa(varargin{:});
end

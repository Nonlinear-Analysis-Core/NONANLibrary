function varargout = CRQA021525(varargin)
%CRQA021525 (deprecated) Use crqa instead.
%   CRQA021525 has been renamed to crqa. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','CRQA021525:deprecated')
%
%   See also CRQA.

warning('CRQA021525:deprecated', ...
    'CRQA021525 has been renamed to crqa. Update your code; this shim will be removed.');
[varargout{1:nargout}] = crqa(varargin{:});
end

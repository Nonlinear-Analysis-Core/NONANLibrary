function varargout = Ent_MS_Plus(varargin)
%ENT_MS_PLUS (deprecated) Use ent_ms_plus instead.
%   Ent_MS_Plus has been renamed to ent_ms_plus. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_MS_Plus:deprecated')
%
%   See also ENT_MS_PLUS.

warning('Ent_MS_Plus:deprecated', ...
    'Ent_MS_Plus has been renamed to ent_ms_plus. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_ms_plus(varargin{:});
end

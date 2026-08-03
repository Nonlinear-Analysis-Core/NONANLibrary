function varargout = Ent_xSamp(varargin)
%ENT_XSAMP (deprecated) Use ent_xsamp instead.
%   Ent_xSamp has been renamed to ent_xsamp. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_xSamp:deprecated')
%
%   See also ENT_XSAMP.

warning('Ent_xSamp:deprecated', ...
    'Ent_xSamp has been renamed to ent_xsamp. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_xsamp(varargin{:});
end

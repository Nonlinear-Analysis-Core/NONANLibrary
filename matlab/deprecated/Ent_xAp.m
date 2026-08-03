function varargout = Ent_xAp(varargin)
%ENT_XAP (deprecated) Use ent_xap instead.
%   Ent_xAp has been renamed to ent_xap. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_xAp:deprecated')
%
%   See also ENT_XAP.

warning('Ent_xAp:deprecated', ...
    'Ent_xAp has been renamed to ent_xap. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_xap(varargin{:});
end

function varargout = Ent_Ap(varargin)
%ENT_AP (deprecated) Use ent_ap instead.
%   Ent_Ap has been renamed to ent_ap. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_Ap:deprecated')
%
%   See also ENT_AP.

warning('Ent_Ap:deprecated', ...
    'Ent_Ap has been renamed to ent_ap. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_ap(varargin{:});
end

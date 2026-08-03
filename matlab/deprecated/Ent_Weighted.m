function varargout = Ent_Weighted(varargin)
%ENT_WEIGHTED (deprecated) Use ent_weighted instead.
%   Ent_Weighted has been renamed to ent_weighted. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_Weighted:deprecated')
%
%   See also ENT_WEIGHTED.

warning('Ent_Weighted:deprecated', ...
    'Ent_Weighted has been renamed to ent_weighted. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_weighted(varargin{:});
end

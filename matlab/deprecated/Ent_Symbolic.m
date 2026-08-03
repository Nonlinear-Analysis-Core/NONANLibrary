function varargout = Ent_Symbolic(varargin)
%ENT_SYMBOLIC (deprecated) Use ent_symbolic instead.
%   Ent_Symbolic has been renamed to ent_symbolic. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Ent_Symbolic:deprecated')
%
%   See also ENT_SYMBOLIC.

warning('Ent_Symbolic:deprecated', ...
    'Ent_Symbolic has been renamed to ent_symbolic. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ent_symbolic(varargin{:});
end

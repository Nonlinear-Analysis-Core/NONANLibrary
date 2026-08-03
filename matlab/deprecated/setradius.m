function varargout = setradius(varargin)
%SETRADIUS (deprecated) Use set_radius instead.
%   setradius has been renamed to set_radius. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','setradius:deprecated')
%
%   See also SET_RADIUS.

warning('setradius:deprecated', ...
    'setradius has been renamed to set_radius. Update your code; this shim will be removed.');
[varargout{1:nargout}] = set_radius(varargin{:});
end

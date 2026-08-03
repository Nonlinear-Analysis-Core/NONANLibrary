function varargout = Surr_findrho(varargin)
%SURR_FINDRHO (deprecated) Use surr_find_rho instead.
%   Surr_findrho has been renamed to surr_find_rho. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Surr_findrho:deprecated')
%
%   See also SURR_FIND_RHO.

warning('Surr_findrho:deprecated', ...
    'Surr_findrho has been renamed to surr_find_rho. Update your code; this shim will be removed.');
[varargout{1:nargout}] = surr_find_rho(varargin{:});
end

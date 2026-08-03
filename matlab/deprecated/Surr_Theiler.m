function varargout = Surr_Theiler(varargin)
%SURR_THEILER (deprecated) Use surr_theiler instead.
%   Surr_Theiler has been renamed to surr_theiler. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Surr_Theiler:deprecated')
%
%   See also SURR_THEILER.

warning('Surr_Theiler:deprecated', ...
    'Surr_Theiler has been renamed to surr_theiler. Update your code; this shim will be removed.');
[varargout{1:nargout}] = surr_theiler(varargin{:});
end

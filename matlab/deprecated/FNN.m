function varargout = FNN(varargin)
%FNN (deprecated) Use fnn instead.
%   FNN has been renamed to fnn. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','FNN:deprecated')
%
%   See also FNN.

warning('FNN:deprecated', ...
    'FNN has been renamed to fnn. Update your code; this shim will be removed.');
[varargout{1:nargout}] = fnn(varargin{:});
end

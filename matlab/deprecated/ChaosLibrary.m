function varargout = ChaosLibrary(varargin)
%CHAOSLIBRARY (deprecated) Use chaos_library instead.
%   ChaosLibrary has been renamed to chaos_library. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','ChaosLibrary:deprecated')
%
%   See also CHAOS_LIBRARY.

warning('ChaosLibrary:deprecated', ...
    'ChaosLibrary has been renamed to chaos_library. Update your code; this shim will be removed.');
[varargout{1:nargout}] = chaos_library(varargin{:});
end

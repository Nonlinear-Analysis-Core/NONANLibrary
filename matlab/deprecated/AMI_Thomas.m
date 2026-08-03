function varargout = AMI_Thomas(varargin)
%AMI_THOMAS (deprecated) Use ami_thomas instead.
%   AMI_Thomas has been renamed to ami_thomas. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','AMI_Thomas:deprecated')
%
%   See also AMI_THOMAS.

warning('AMI_Thomas:deprecated', ...
    'AMI_Thomas has been renamed to ami_thomas. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ami_thomas(varargin{:});
end

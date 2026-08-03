function varargout = AMI_Stergiou(varargin)
%AMI_STERGIOU (deprecated) Use ami_stergiou instead.
%   AMI_Stergiou has been renamed to ami_stergiou. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','AMI_Stergiou:deprecated')
%
%   See also AMI_STERGIOU.

warning('AMI_Stergiou:deprecated', ...
    'AMI_Stergiou has been renamed to ami_stergiou. Update your code; this shim will be removed.');
[varargout{1:nargout}] = ami_stergiou(varargin{:});
end

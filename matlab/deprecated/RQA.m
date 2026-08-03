function varargout = RQA(varargin)
%RQA (deprecated) Use rqa_legacy instead.
%   RQA has been renamed to rqa_legacy. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','RQA:deprecated')
%
%   See also RQA_LEGACY.

warning('RQA:deprecated', ...
    'RQA has been renamed to rqa_legacy. Update your code; this shim will be removed.');
[varargout{1:nargout}] = rqa_legacy(varargin{:});
end

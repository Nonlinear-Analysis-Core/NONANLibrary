function varargout = RQA_plot(varargin)
%RQA_PLOT (deprecated) Use rqa_plot instead.
%   RQA_plot has been renamed to rqa_plot. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','RQA_plot:deprecated')
%
%   See also RQA_PLOT.

warning('RQA_plot:deprecated', ...
    'RQA_plot has been renamed to rqa_plot. Update your code; this shim will be removed.');
[varargout{1:nargout}] = rqa_plot(varargin{:});
end

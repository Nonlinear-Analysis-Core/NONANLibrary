function varargout = Surr_PseudoPeriodic(varargin)
%SURR_PSEUDOPERIODIC (deprecated) Use surr_pseudo_periodic instead.
%   Surr_PseudoPeriodic has been renamed to surr_pseudo_periodic. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','Surr_PseudoPeriodic:deprecated')
%
%   See also SURR_PSEUDO_PERIODIC.

warning('Surr_PseudoPeriodic:deprecated', ...
    'Surr_PseudoPeriodic has been renamed to surr_pseudo_periodic. Update your code; this shim will be removed.');
[varargout{1:nargout}] = surr_pseudo_periodic(varargin{:});
end

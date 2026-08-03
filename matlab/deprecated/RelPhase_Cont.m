function varargout = RelPhase_Cont(varargin)
%RELPHASE_CONT (deprecated) Use rel_phase_cont instead.
%   RelPhase_Cont has been renamed to rel_phase_cont. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','RelPhase_Cont:deprecated')
%
%   See also REL_PHASE_CONT.

warning('RelPhase_Cont:deprecated', ...
    'RelPhase_Cont has been renamed to rel_phase_cont. Update your code; this shim will be removed.');
[varargout{1:nargout}] = rel_phase_cont(varargin{:});
end

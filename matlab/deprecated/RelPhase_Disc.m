function varargout = RelPhase_Disc(varargin)
%RELPHASE_DISC (deprecated) Use rel_phase_disc instead.
%   RelPhase_Disc has been renamed to rel_phase_disc. This shim forwards every argument and
%   output unchanged, so existing scripts keep working, and warns once per
%   session.
%
%   Silence with: warning('off','RelPhase_Disc:deprecated')
%
%   See also REL_PHASE_DISC.

warning('RelPhase_Disc:deprecated', ...
    'RelPhase_Disc has been renamed to rel_phase_disc. Update your code; this shim will be removed.');
[varargout{1:nargout}] = rel_phase_disc(varargin{:});
end

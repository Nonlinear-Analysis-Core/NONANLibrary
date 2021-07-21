function [crp, crpH] = RelPhase_Cont(data1,data2,samprate)
% RELPHASE - calculates the relative phase between two segments, joint
% angles, coordinates, etc.  It is calculated using the atan2 function in
% order to preserve the quadrant in the phase portrait.  The sign of the
% relative phase is determined from segment A - segment B.  If the sign is
% negative, B leads A.  If the sign is positive, A leads B.
%
% Syntax:
%       Calculate relative phase from position and velocity data
%       rp = relphase(pa,va,pb,vb)
%           pa - angle (or position) of segment A
%           va - velocity of segment A
%           pb - angle (or position) of segment B
%           vb - velocity of segment B
%           rp - relative phase in degrees
%
% Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
% Variability, University of Nebraska at Omaha
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its
%    contributors may be used to endorse or promote products derived from
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%% Normalize the data

data1 = data1 - min(data1) - range(data1)/2;
data2 = data2 - min(data2) - range(data2)/2;

% Calculate a basic phase

crp = atan((diff(data1)*samprate.*data2(1:end-1) - diff(data2)*samprate.*data1(1:end-1))./(data1(1:end-1).*data2(1:end-1) - diff(data2)*samprate.*diff(data1)*samprate))*180/pi;

% Calculate the phase angle using a Hilbert tranform
env1H = hilbert(data1);
env2H = hilbert(data2);
crpH = atan((imag(env1H).*data2 - imag(env2H).*data1)./(data1.*data2 + imag(env1H).*imag(env2H)))*180/pi;

return

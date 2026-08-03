function [rho,out]=Surr_findrho(y,tau,dim,varargin)
% rho=Surr_findrho20200626(y,tau,dim)
% inputs  - y, time series
%         - tau, time lag for phase space reconstruction
%         - dim, embedding dimension for phase space reconstruction
% inputs  - rhoL, optional lower search bound. Default 0.1.
%         - rhoH, optional upper search bound. Default 1.
% outputs - rho, noise radius
%         - out, an informational array with results of rho and di from the
%                iterative processes
% Remarks
% - This code finds an optimal value of rho for the pseudo periodic
%   surrogation algorithm. It maximizes the number of short sequences in
%   the surrogate that are identical to the original time series.
% - The method in this code first finds rho at a range of values untill a
%   suspected peak is found. A binary search is then performed around the
%   peak untill the percent difference decreases below a threshold.
% Future Work
% - The value rho tends to maximize with a pulse with very small values
%   around it. It is suspected this may cause issues with some time series
%   but it has not been encountered.
% - It's possible the rhoL value of 0.1 may cause issues with pps in
%   certain time series.
% - It's possible a time series may have an optimal value of rho above 0.1
%   but this has not been encountered.
% References
% - Small, M., Yu, D., & G., H. R. (2001). Surrogate Test for
%   Pseudoperiodic Time Series Data. Physical Revew Letters, 87(18).
%   https://doi.org/10.1063/1.1487534
% Version History
% May 2001 - Created by Michael Small
%          - It is believed the original version of this code was written
%            by Michael Small but the source could not be confirmed.
% Jun 2020 - Modifed by Ben Senderling
%          - Removed input handling, all three are necessary. In this
%            version the initial search for the bounds of the binary search
%            was removed completely. It was noticed the optimal rho is
%            frequently ~0.5-0.6. The bounds were replaced with 0.1 and 1 
%            for rhoL and rhoH. This offered a 35% speed improvement over 
%            the previous version when testing an ankle angle, n=10800, 10 
%            times. Added an out variable to use for troubleshooting and
%            diagnostics.
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
%%


%% Search bounds
% Optional fourth and fifth arguments override the default bracket. The
% defaults were chosen from an ankle-angle signal, for which the optimum is
% near 0.5; smooth strongly cyclic data can have its optimum at or below the
% lower bound, in which case the bisection cannot reach it and a warning is
% issued below.

rhoL=0.1;
rhoH=1;
if numel(varargin)>=1 && ~isempty(varargin{1}), rhoL=varargin{1}; end
if numel(varargin)>=2 && ~isempty(varargin{2}), rhoH=varargin{2}; end
if ~isscalar(rhoL) || ~isscalar(rhoH) || ~(rhoL>0) || ~(rhoH>rhoL)
    error('Surr_findrho:bounds', ...
        'Search bounds must satisfy 0 < rhoL < rhoH.');
end

%% Find upper bound for binary search

[~,yi]=Surr_PseudoPeriodic(y,tau,dim,rhoH);
diH=findrho_di(yi,2);
out(1,1:3)=[1,rhoH,diH];

%% Find lower bound for the binary search

[~,yi]=Surr_PseudoPeriodic(y,tau,dim,rhoL);
diL=findrho_di(yi,2);
out(2,1:3)=[2,rhoL,diL];

%% Find which bound has more short continuous sequences from the original 
% time series.

% Seed both the best value and the rho that produced it. Previously only
% dmax was seeded, and rho was assigned solely inside `if di>dmax` in the
% loop below, so whenever no interior point beat both endpoints the output
% was never assigned and the caller received an unassigned-output error. That
% is the ordinary case for smooth cyclic data, where di(rho) is maximised at
% or beyond an endpoint: measured failure rates were 30% on a sine and 18% on
% a noisy cycle.
if diH>diL
    dmax=diH;
    rho=rhoH;
else
    dmax=diL;
    rho=rhoL;
end

%% Perform binary search

precision=0.02;
ind=3;

while abs(rhoH-rhoL)/rhoL>precision
    rhoi=(rhoH+rhoL)/2;
    [~,yi]=Surr_PseudoPeriodic(y,tau,dim,rhoi);
    di=findrho_di(yi,2);
    out(ind,1:3)=[ind,rhoi,di];
    ind=ind+1;
    
    if di>dmax
        dmax=di;
        rho=rhoi;
    end
    if diL<diH
        diL=di;
        rhoL=rhoi;
    else
        diH=di;
        rhoH=rhoi;
    end
end

% If the best rho sits in the outer part of the bracket, the search was
% pushed against a bound and the optimum is probably outside it, so the
% returned value reflects where the bracket was placed rather than a maximum
% of di(rho). Proximity is judged on a log scale because the bisection is
% effectively multiplicative, and with a margin rather than exact equality
% because di is stochastic: a single noisy interior point can beat both
% endpoints by one count and land just inside the bound.
loBound=out(2,2);
hiBound=out(1,2);
margin=0.15*log(hiBound/loBound);
if log(rho/loBound)<margin || log(hiBound/rho)<margin
    warning('Surr_findrho:optimumAtBound', ...
        ['The best rho found (%g) lies against a search bound [%g %g], so ' ...
         'the optimum is probably not bracketed and this value may not be a ' ...
         'maximum of di(rho). Re-run with wider bounds, e.g. ' ...
         'Surr_findrho(y,tau,dim,%g,%g).'], rho, loBound, hiBound, loBound/10, hiBound*2);
end

end

function di=findrho_di(yi,n)
% This function counts the number of short continuous sequences in the 
% surrogate that are pulled from the original with length of n.

di=diff(find(diff(yi)~=1));
di=sum(di>n);

end





function [ys,yi]=Surr_PseudoPeriodic(y,tau,dim,rho)
% [ys,yi]=Surr_PseudoPeriodic20200623(y,tau,dim,rho)
% inputs  - y, time series
%         - tau, time lag for phase space reconstruction
%         - dim, embedding dimension for phase space reconstruction
%         - rho, noise radius
% outputs - ys, surrogate time series
%         - yi, selected indexes for surrogate from original time series
% Remarks
% - This code produces one pseudo periodic surrogate. It is appropriate to
%   run on period time series to remove the long-term correlations. This is
%   useful when testing for the presense of chaos or testing various
%   nonlinear analysis methods.
% - There may be an optimal value of rho. This can be found by using a
%   different function. Or it can be specified manually.
% - If rho is too low, ~<0.01, the code will not be able to find a
%   neighbor.
% Future Work
% - Previous versions had occationally created surrgates with plataues. It
%   is unknown if these are present in the current version.
% References
% - Small, M., Yu, D., & G., H. R. (2001). Surrogate Test for 
%   Pseudoperiodic Time Series Data. Physical Revew Letters, 87(18). 
%   https://doi.org/10.1063/1.1487534
% Version History
% May 2001 - Created by Michael Small
%            - The original version of this script was converted from
%              Michael Small's C code to MATLAB by Ben Senderling.
% Jun 2020 - Modified by Ben Senderling
%          - The original was heavily modified while referencing Small, 
%            2001. For loops and equations were indexed to save space and 
%            speed up the script. The phase space reconstruction was 
%            changed from a backwards to forwards lag. The initial seed was
%            removed as an input. Added a line to remove self-matches.
%            Added an exception in case a new value of xi could not be
%            found.
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
%% Begin Code
dbstop if error

%% Phase space reconstruction
N=length(y);
Y=zeros(N-(dim-1)*tau,dim);
for i=1:dim
    Y(:,i)=y(1+(i-1)*tau:N-(dim-i)*tau);
end

%% Seeding and initial points
xi=floor(rand(1)*length(Y))+1;
ys=zeros(length(Y),1);
ys(1)=y(xi);
yi=zeros(length(Y),1);
yi(1)=xi;

% This is needed for the probability calculation because the point before
% the most probably is selected. We want to point after so the difference
% is 2.
M=length(Y)-2;

%% Construct the surrogate
for i=2:1:length(Y)
    
    % Calculates the distance from the previous point to all other points.
    % This is the probability calculation in Small, 2001. Points that are 
    % close neighbors will end up with a higher value.
    prob=exp(-sqrt(sum((Y(1:M,:)-repmat(Y(xi,:),M,1)).^2,2))/rho);
    % A self-match will be exp(0)=1, which can be large compared to the
    % other values. It could be removed. Adding in this line appears to
    % produce decent surrogates but makes the optimization method
    % un-applicable.
%     prob(xi)=0;
    % Cummulative sum of the probability
    sum3=cumsum(prob);
    % A random number is chosen between 0 and the cummulative probability.
    % Where it goes above the cumsum that is chosen as the next point, +2.
    % Most of the values in prob have a very small value, the close
    % neighbors are the spikes.
    xi_n=[];
    ind=1;
    while isempty(xi_n) || xi_n>length(Y) || xi_n==xi
        xi_n=find(sum3<rand(1)*sum3(end),1,'last')+2;
        ind=ind+1;
        if ind==100
            error('a new value of xi could not be found, check that rho is not too low')
        end
    end
    xi=xi_n;
    
    % Add the new point to the surrogate time series.
    ys(i) = y(xi);
    yi(i) = xi;
end




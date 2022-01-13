function z =Surr_Theiler(y,algorithm)
% z=Surr_Theiler20200723(y,algorithm)
% inputs  - y, time series to be surrogated
%              algorithm - the type of algorithm to be completed
% outputs - z, surrogated time series
% Remarks
% - This code creates a surrogate time series according to Algorithm 0,
%   Algorithm 1 or Algorithm 2.
% Future Work
% - None.
% References
% - Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Doyne 
%   Farmer, J. (1992). Testing for nonlinearity in time series: the 
%   method of surrogate data. Physica D: Nonlinear Phenomena, 58(1–4), 
%   77–94. https://doi.org/10.1016/0167-2789(92)90102-S
% Jun 2015 - Modified by Ben Senderling
%          - Added function help section and plot commands for user
%            feedback
%          - The code was originally created as two algorithms. It was
%            modified so one code included both functions.
% Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Changed file and function name.
%          - Added reference.
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
%% Begin code
switch (algorithm)
    case 0
        z=randn(size(y));
        [~,idx]=sort(z);
        z=y(idx);
    case 1
        z=surr1(y,1);
    case 2
        z=surr1(y,2);
end

end

function z = surr1(x,algorithm)

[r,c] = size(x);

y= zeros(r,c);

if abs(algorithm)==2
    ra= randn(size(x));
    [sr,~]= sort(ra);
    [sx,xi]= sort(x);
    [~,xii]= sort(xi);
    for k=1:c
        y(:,k)= sr(xii(:,k));
    end
else
    y= x;
end
m= mean(y);
y= y-m(ones(r,1),:);

fy = fft(y);

% randomizing phase
phase= rand(r,1);
phase= phase(:,ones(1,c));

rot= exp(1) .^ (2*pi*sqrt(-1)*phase);
fyy= fy .* rot;

yy= real(ifft(fyy)) +  m(ones(r,1),:);

if abs(algorithm)==2
    [~,yyi] = sort(yy);
    [~,yyii] = sort(yyi);
    for k=1:c
        z(:,k) = sx(yyii(:,k));
    end
else
    z= yy;
end

end






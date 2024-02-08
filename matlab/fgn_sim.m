function series = fgn_sim(n, H)
    % Generate Fractional Gaussian Noise (FGN) time series
    % inputs:
    %   n (integer): Desired length of the series
    %   H (real number): Hurst exponent for the output series (0 < H < 1)
    %
    % outputs:
    %   series: Real-valued time series of length n with Hurst exponent H
    %
    % References:
    %   Beran, J. (1994). Statistics for long-memory processes. Chapman & Hall.

%% ========================================================================
%                          ------ EXAMPLE ------
      
%       - Create time series of 1000 datapoints to have an H of 0.7
%       n = 1000
%       H = 0.7
%       dat = fgn_sim(n, H)
  
%% ========================================================================

% Copyright 2024 Nonlinear Analysis Core, Center for Human Movement
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

        % FUNCTION:
        
        % Settings:
        mu = 0; % output mean
        sd = 1; % output standard deviation
        
        % Generate Sequence:
        z = randn(1,2*n);
        zr = z(1:n);
        
        zi = z((n+1):(2*n));
        zic = -zi;
        zi(1) = 0;
        zr(1) = zr(1)*sqrt(2);
        zi(n) = 0;
        zr(n) = zr(n)*sqrt(2);
        zr = [zr(1:n), zr(fliplr(2:(n-1)))];
        zi = [zi(1:n), zic(fliplr(2:(n-1)))];
        z = complex(zr, zi);
        
        k = 0:(n-1);
        gammak = ((abs(k-1).^(2*H))-(2*abs(k).^(2*H))+(abs(k+1).^(2*H)))/2;
        ind = [0:(n - 2), (n - 1), fliplr(1:(n-2))];
        gkFGN0 = ifft(gammak(ind + 1))*length(z); % needs to non-normalized
        gksqrt = real(gkFGN0);
        if (all(gksqrt > 0))
            gksqrt = sqrt(gksqrt);  
            z = z.*gksqrt;
            z = ifft(z)*length(z); 
            z = 0.5*(n-1).^(-0.5)*z;
            z = real(z(1:n));
        else
            gksqrt = 0*gksqrt;
            Error("Re(gk)-vector not positive")
        end
        
        % Standardize:
        % (z-mean(z))/sqrt(var(z))
        series = sd*z + mu;

end

function xSE = Ent_xSamp(x,y,m,R,norm)
% xSE = Ent_xSamp20180320(x,y,m,R,norm)
% Inputs - x, first data series
%        - y, second data series
%        - m, vector length for matching (usually 2 or 3)
%        - R, R tolerance to find matches (as a proportion of the average 
%             of the SDs of the data sets, usually between 0.15 and 0.25)
%        - norm, normalization to perform
%          - 1 = max rescale/unit interval (data ranges in value from 0 - 1
%            ) Most commonly used for RQA.
%          - 2 = mean/Zscore (used when data is more variable or has 
%            outliers) normalized data has SD = 1. This is best for cross 
%            sample entropy.
%          - Set to any value other than 1 or 2 to not normalize/rescale 
%            the data
% Remarks
% - Function to calculate cross sample entropy for 2 data series using the
%   method described by Richman and Moorman (2000).
% Sep 2015 - Created by John McCamley, unonbcf@unomaha.edu
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
% Check both sets of data are the same length
xl = length(x);
yl = length(y);
if xl ~= yl
    disp('The data series need to be the same length!')
end
N = length(x);
% normalize the data ensure data fits in the same "space"
if norm == 1 %normalize data to have a range 0 - 1
    xn = (x - min(x))/(max(x) - min(x));
    yn = (y - min(y))/(max(y) - min(y));
    r = R * ((std(xn)+std(yn))/2);
elseif norm == 2 % normalize data to have a SD = 1, and mean = 0
    xn = (x - mean(x))/std(x);
    yn = (y - mean(y))/std(y);
    r = R;
else disp('These data will not be normalized')
end

for i = 1:N-m
    for k = 1:m+1
        dij(:,k) = abs(xn(1+k-1:N-m+k-1)-yn(i+k-1));
    end
    dj = max(dij(:,1:m),[],2);
    dj1 = max(dij,[],2);
    d = find(dj<=r);
    d1 = find(dj1<=r);
    nm = length(d);
    Bm(i) = nm/(N-m);
    nm1 = length(d1);
    Am(i) = nm1/(N-m);
end

Bmr = sum(Bm)/(N-m);
Amr = sum(Am)/(N-m);

xSE = -log(Amr/Bmr);
end
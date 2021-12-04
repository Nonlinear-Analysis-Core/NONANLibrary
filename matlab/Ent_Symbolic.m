function [ NCSE ] = Ent_Symbolic( X, L )
% [ SymEnt ] = Ent_Symbolic20180320( X, L )
% symbolicEnt Calculates the Symbolic Entropy for given data.
% Input -   X: 1-Dimensional binary array of data
%           L: Word length
% Output -  NCSE: Normalized Corrected Shannon Entropy
% Remarks
% - This code calculates the Symbbolic Entropy value for the provided data
%   at a given word length described by - Aziz, W., Arif, M., 2006.
%   "Complexity analysis of stride interval time series by threshold
%   dependent symbolic entropy." Eur. J. Appl. Physiol. 98: 30-40.
% Jun 2017 - Created by William Denton, unonbcf@unomaha.edu
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
%% Begin code: (Do NOT Edit)
%% Correct orientation of array.
[r,c] = size(X);
if r > c
    X = X';
end
%% Convert binary values to decimal.
words = zeros(length(X)-L+1,1);
for i = 1:length(X)-L+1
    words(i,1) = bin2dec(num2str(X(i:i+L-1)));
end
%% Calculate probability.
max_words = 2^L;
for i = 1:max_words
    P(i) = sum(words == i-1)/length(words);
    H(i) = P(i)*log2(P(i));
end
H = -sum(H(~isnan(H)));
%% Normalized Corrected Shannon Entropy
So = length(unique(words));
Sm = max_words;
CSE = H+(So-1) / (2*Sm*log(2));
CSEm = -log2(1/Sm) + (Sm-1) / (2*Sm*log(2));
NCSE = CSE/CSEm;
%% Print out Symbolic Entropy Value.
fprintf('Normalized Corrected Shannon Entropy = %2.3f bits\r',NCSE);
end
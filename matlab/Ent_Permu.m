function [permEnt, hist] = Ent_Permu(data, m, tau)
% [permEnt, hist] = Ent_Permu20180320(data, m, tau)
% inputs -  data: 1-D array of data being analyzed
%           m: embedding dimension (order of permutation entropy) 
%           tau: time delay
% outputs - permuEnt: value calculated using a log base of 2
%           hist: number of occurences for each permutation order
% Remarks
% - It differs from the permutation entropy code found on MatLab Central in
%   one way (see MathWorks reference). The code on MatLab Central uses the 
%   log function (base e, natural log), whereas this code uses log2 (base 2
%   ), as per Bandt & Pompe, 2002. However, this code does include a lag 
%   (time delay) feature like the one on MatLab Central does.
% - Complexity parameters for time series based on comparison of 
%   neighboring values. Based on the distributions of ordinal patterns, 
%   which describe order relations between the values of a time series. 
%   Based on the algorithm described by Bandt & Pompe, 2002.
% References
% - Bandt, C., Pompe, B. Permutation entropy: A natural complexity measure 
%   for time series. Phys Rev Lett 2002, 88, 174102, 
%   doi:10.1103/PhysRevLett.88.174102
% - MathWorks: http:www.mathworks.com/matlabcentral/fileexchange/
%   37289-permutation-entropy)
% Jun 2016 - Created by Patrick Meng-Frecker, unonbcf@unomaha.edu
% Dec 2016 - Edited by Casey Wiens, email: unonbcf@unomaha.edu
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

N = length(data);  % length of time series
perm = perms(1:m);  % create all possible permutation vectors
hist(1:length(perm)) = 0;   % designate variable to store values

for cnt1=1:N-tau*(m-1)  % steps from 1 through length of data minus time delay multiplied by order minus 1
    [~, permVal] = sort(data(cnt1:tau:cnt1+tau*(m-1))); % creates permutation of selected data range
    for cnt2=1:length(perm) % steps through length of possible permutation vectors
        if perm(cnt2,:) - permVal == 0  % compares current permutation of selected data to possible permutation vectors
            hist(cnt2) = hist(cnt2) + 1;    % if above comparison is equal, then adds one to bin for appropriate permutation vector
        end
    end
end

histNew = hist(hist ~= 0);  % remove any permutation orders with 0 for proper calculation
per = histNew/sum(histNew);	% ratio of each permutation vector match to total matches
permEnt = -sum(per .* log2(per));   % performs entropy calucation




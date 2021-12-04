function [tau,v_AMI]=AMI_Stergiou(data,L,varargin)

% [tau,v_AMI]=AMI(data,L,bins)
% inputs    - data, column oriented time series
%           - L, maximal lag to which AMI will be calculated
%           - bins, number of bins to use in the calculation, if empty an
%             addaptive formula will be used
% outputs   - tau, first minimum in the AMI vs lag plot
%           - v_AMI, vector of AMI values and associated lags
% Remarks
% - This code uses average mutual information to find an appropriate lag
%   with which to perform phase space reconstruction. It is based on a
%   histogram method of calculating AMI.
% - In the case a value of tau could not be found before L the code will
%   automatically re-execute with a higher value of L, and will continue to
%   re-execute up to a ceiling value of L.
% - The sparse function was historically used in this code to calculate the
%   probability distribution of the numbers in the time series.
% - There are some cases where a weakly periodic time series will not have
%   a clear first minimum and the code will return unsatisfactory results.
%   For these cases the last row of v_AMI contains the time lag at 1/5 the
%   original AMI. The best way to confirm this choice is to lot the AMI
%   verses the time lag. (Abarbanel et al. 1993)
% Future Work
% - None currently.
% References
% - Abarbanel, H. D. I. I., Brown, R., Sidorowich, J. J., & Tsimring, L. S.
%   (1993). The analysis of observed chaotic datain physical systems. 
%   Reviews of Modern Physics, 65(4), 1331–1392.
%   https://doi.org/10.1103/RevModPhys.65.1331
% Mar 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
%          - Modified code to output a plot and notify the user if a value
%            of tau could not be found.
% Sep 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
%          - Previously the number of bins was hard coded at 128. This
%            created a large amount of error in calculated AMI value and
%            vastly decreased the sensitivity of the calculation to changes
%            in lag. The number of bins was replaced with an adaptive
%            formula well known in statistics. (Scott 1979
%          - The previous plot output was removed.
% Oct 2017 - Modified by Ben Senderling, email unonbcf@unomaha.edu
%          - Added print commands to display progress.
% May 2019 - Modified by Ben Senderling, email unonbcf@unomaha.edu
%          - In cases where L was not high enough to find a minimun the
%            code would reexecute with a higher L, and the binned data.
%            This second part is incorrect and was corrected by using
%            data2.
%          - The reexecution part did not have the correct input
%            parameters.
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

N=length(data);
epsilon = eps;      %or use epsilon = 1e-10;

% set number of bins
if isempty(varargin)
    bins=ceil(range(data)/(3.49*nanstd(data)*N^(-1/3))); % Scott 1979
%     fprintf('number of bins = %.0f\n',bins)
else
    bins=varargin{1};
end

data = data - min(data); % make all data points positive
data2 = 1+ floor(data/(max(data)/(bins-epsilon))); % scaling the data

v=zeros(L,1); % preallocate vector
overlap=N-L;

increment= 1/overlap;
one = ones(overlap,1); %create a column vector with all elements being one

% MUTUAL INFORMATION (not used, do not know why, BS ))
% I (time_lag) = sum [ p(x(t), x(t + time_lag))*log[(p(x(t),p(x + time_lag))/p(x(t))*p(x(t+time_lag))]
%find probability p(x(t))= pA

pA = sparse(data2(1:overlap),one,increment);
a=unique(data2(1:overlap));
for i=1:length(a)
    pA2(i,1)=sum(data2(1:overlap)==a(i))*increment;
end
%e.g. when overalp = N+1-L = 6001+1-32= 5970, max(data(1:overlap))=129, 
%creating a histogram with (129-1) bins
% sum(pA)= 1 --> 100 % in total 

tau=[];
v=zeros(L+1,2);

% fprintf('AMI: 00%%')

for lag = 0: L % used to be from 0:L-1 (BS)
    
    v(lag+1,1)=lag;
    
    %find probablity p(x(t+time_lag))=pB, sum(pB)=1
    pB = sparse(one, data2(1+lag:overlap+lag), increment);
    %find joing probability p(A,B)=p(x(t),x(t+time_lag))
    pAB = sparse(data2(1:overlap),data2(1+lag:overlap+lag),increment);
    [A, B, AB]=find(pAB);
    v(lag+1,2)=sum(AB.*log2(AB./(pA(A).*pB(B)')));  %Average Mutual Information
    v2(lag+1,2)=sum(AB.*log2(AB./(pA(A).*pB(B)')));  %Average Mutual Information
    
%     fprintf(repmat('\b',1,4));
%     msg=sprintf('%3.0d',floor(lag/L*100));
%     fprintf([msg '%%']);
    
end

% fprintf('\n')

tau=[];
for i=2:length(v)-1
    if (v(i-1,2)>=v(i,2))&&(v(i,2)<=v(i+1,2))
        tau(end+1,:)=[v(i,1) v(i,2)];
    end
end

ind=find(v(:,2)<=(0.2*v(1,2)),1,'first'); % finds lag at 20% initial AMI
if ~isempty(ind)
    tau(end+1,:)=v(ind,:);
end

v_AMI=v;

% Catch if no value of tau is found
if isempty(tau)
    if L*1.5>length(data/3)
        tau=9999;
    else
        fprintf('max lag needed to be increased for AMI_Stergiou\n')
        [tau,v_AMI]=AMI_Stergiou20190501(data,floor(L*1.5));
    end
end



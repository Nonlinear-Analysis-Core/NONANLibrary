function varargout=AMI_Stergiou(varargin)
% [tau,ami]=AMI_Stergiou20210329(data,L)
% inputs    - data, single column array
%           - L, maximal lag to which AMI will be calculated
%           - bins, number of bins to use in the calculation, if empty an
%             addaptive formula will be used
% outputs   - tau, first minimum in the AMI vs lag plot
%           - v_AMI, vector of AMI values and associated lags
% [tau,ami]=AMI_Stergiou20210329(data,L,bins)
% inputs    - bins, number of bins to use in the calculation
% [ami]=AMI_Stergiou20210329(x,y)
% inputs    - x, single column array with the same length as y
%           - y, single column array with the same length as x
% outputs   - ami, the average mutual information between the two arrays
% Remarks
% - The first and second implementations of this code uses average mutual
%   information to find an appropriate lag with which to perform phase
%   space reconstruction. It is based on a histogram method of calculating
%   AMI. The first implementation uses and adaptive algorithm to calculate
%   the number of bins. The second implementation allows control over the
%   number of bins.
% - In the case a value of tau could not be found before L the code will
%   return an empty tau and the ami vector.
% - The sparse function was historically used in this code to calculate the
%   probability distribution of the numbers in the time series.
% - There are some cases where a weakly periodic time series will not have
%   a clear first minimum and the code will return unsatisfactory results.
%   For these cases the last row of v_AMI contains the time lag at 1/5 the
%   original AMI. The best way to confirm this choice is to lot the AMI
%   verses the time lag. (Abarbanel et al. 1993)
% - The third implementation of this code returns the average mutual
%   information between array x and y.
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
% Apr 2021 - Modified by Ben Senderling, email bmchnonan@unomaha.edu
%          - Modified in conjunction with NONAN validation efforts.
%            Reorganized the code and added the third implementation.
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

dbstop if error

if (nargin==2 || nargin==3) && numel(varargin{2})==1
    
    data = varargin{1};
    L = varargin{2};
    N=length(data);
    
    if nargin==2
        bins = ceil(range(data)/(3.49*nanstd(data)*N^(-1/3))); % Scott 1979
    elseif nargin==3
        bins = varargin{3};
        if numel(bins)==1
            error('input bins must be a single number')
        end
    end
    
    epsilon = eps;      %or use epsilon = 1e-10;
    
    data = data - min(data); % make all data points positive
    y = 1+ floor(data/(max(data)/(bins-epsilon))); % scaling the data
    
    ami=zeros(L,1); % preallocate vector
    overlap=N-L;
    
    increment= 1/overlap;
    one = ones(overlap,1); %create a column vector with all elements being one
    
    pA = sparse(y(1:overlap),one,increment);
    a=unique(y(1:overlap));
    for i=1:length(a)
        pA2(i,1)=sum(y(1:overlap)==a(i))*increment;
    end
    
    tau=[];
    ami=zeros(L+1,2);
    
    % fprintf('AMI: 00%%')
    
    for lag = 0: L % used to be from 0:L-1 (BS)
        
        ami(lag+1,1)=lag;
        
        %find probablity p(x(t+time_lag))=pB, sum(pB)=1
        pB = sparse(one, y(1+lag:overlap+lag), increment);
        %find joing probability p(A,B)=p(x(t),x(t+time_lag))
        pAB = sparse(y(1:overlap),y(1+lag:overlap+lag),increment);
        [A, B, AB]=find(pAB);
        ami(lag+1,2)=sum(AB.*log2(AB./(pA(A).*pB(B)')));  %Average Mutual Information
        v2(lag+1,2)=sum(AB.*log2(AB./(pA(A).*pB(B)')));  %Average Mutual Information
        
    end
    
    tau=[];
    for i=2:length(ami)-1
        if (ami(i-1,2)>=ami(i,2))&&(ami(i,2)<=ami(i+1,2))
            tau(end+1,:)=[ami(i,1) ami(i,2)];
        end
    end
    
    ind=find(ami(:,2)<=(0.2*ami(1,2)),1,'first'); % finds lag at 20% initial AMI
    if ~isempty(ind)
        tau(end+1,:)=ami(ind,:);
    end
    
    varargout{1}=tau;
    varargout{2}=ami;
    
elseif nargin==2 && numel(varargin{1})>1 && numel(varargin{2})>1
    
    x = varargin{1};
    y = varargin{2};
    
    if numel(x)~=numel(y)
        error('x and y must be the same size')
    end
    
    increment = 1/length(y);
    one = ones(length(y),1);
    
    bins1=ceil(range(x)/(3.49*nanstd(x)*length(x)^(-1/3))); % Scott 1979
    bins2=ceil(range(y)/(3.49*nanstd(y)*length(y)^(-1/3))); % Scott 1979
    x = x - min(x); % make all data points positive
    x = 1 + floor(x/(max(x)/(bins1 - eps))); % scaling the data
    y = y - min(y); % make all data points positive
    y = 1 + floor(y/(max(y)/(bins2-eps))); % scaling the data
    
    pA = sparse(x, one, increment);
    pB = sparse(one, y, increment);
    pAB = sparse(x, y, increment);
    [A, B, AB] = find(pAB);
    ami = sum(AB.*log2(AB./(pA(A).*pB(B)')));  %Average Mutual Information
    
    varargout{1}=ami;
    
end


function [AE] = Ent_Ap( data, dim, r )
%Ent_Ap20120321
%   data : time-series data
%   dim : embedded dimension
%   r : tolerance (typically 0.2)
%
%   Changes in version 1
%       Ver 0 had a minor error in the final step of calculating ApEn
%       because it took logarithm after summation of phi's.
%       In Ver 1, I restored the definition according to original paper's
%       definition, to be consistent with most of the work in the
%       literature. Note that this definition won't work for Sample
%       Entropy which doesn't count self-matching case, because the count 
%       can be zero and logarithm can fail.
%
%   *NOTE: This code is faster and gives the same result as ApEn = 
%          ApEnt(data,m,R) created by John McCamley in June of 2015.
%          -Will Denton
%
%---------------------------------------------------------------------
% coded by Kijoon Lee,  kjlee@ntu.edu.sg
% Ver 0 : Aug 4th, 2011
% Ver 1 : Mar 21st, 2012
%---------------------------------------------------------------------

r = r*std(data);
N = length(data);
phim = zeros(1,2);
for j = 1:2
    m = dim+j-1;
    phi = zeros(1,N-m+1);
    dataMat = zeros(m,N-m+1);
    for i = 1:m
        dataMat(i,:) = data(i:N-m+i);
    end
    for i = 1:N-m+1
        tempMat = abs(dataMat - repmat(dataMat(:,i),1,N-m+1));
        AorB = any( (tempMat > r),1);
        phi(i) = sum(~AorB)/(N-m+1);
    end
    phim(j) = sum(log(phi))/(N-m+1);
end
AE = phim(1)-phim(2);
end
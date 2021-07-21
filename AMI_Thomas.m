function [tau,AMI]=AMI_Thomas(x,L)

% [tau,AMI]=AMI_Thomas(x,L)
% inputs:    x - time series, vertically orientedtrc files selected by user
%            L - Maximum lag to calculate AMI until
% outputs:   tau - first true minimum of the AMI vs lag plot
%            AMI - a vertically oriented vector containing values of AMI
%            from a lag of 0 up the input L
%
% Remarks
% - This code uses a published method of calculating AMI to find an 
%   acceptable lag with which to perform phase space reconstruction.
% - The algorithm is publically available in:
%   - An efficient algorithm for the computation of average mutual
%     information: Validation and implementation in Matlab. Journal of
%     Mathematical Psychology, 2014/08.
% - If a value of tau if not found the code will automatically re-execute
%   with a higher maximal lag. It will notify the user of this but will not
%   run for ever if a lag cannot be found.
% - If it does find multiple values of tau but no definative minimum it
%   will return all of these values.
%
% Future Work
% - Further validation methods could be added below, but this code is
%   pretty good.
%
% September 2015 - Adapted by Ben Senderling, email: bensenderling@gmail.com
%                  Below I've set the code published by Thomas, Semple and
%                  Strang to calculate AMI at various lags and to suggest
%                  an appropriate tau.
%
%% Validation
%
% Damped oscillator (approximate tau ~ 33)
%
% L=35;
% t=(1:500)';
% a=0.005;
% w=0.05;
% x=exp(-a*t).*sin(w*t);
% 
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

% check size of input x
[m,n]=size(x);
if m==1
    x=x';
elseif m>1 && n>1
    error('input time series is not a one dimensional vector')
end

% calculate AMI at each lag
AMI=zeros(L,2);
% fprintf('AMI: 00%%')
for i=1:L
    AMI(i,1)=i;
    X=x(1:end-i);
    Y=x(i+1:end);
    AMI(i,2)=average_mutual_information([X,Y]);
    
%     fprintf(repmat('\b',1,4));
%     msg=sprintf('%3.0d',floor(i/L*100));
%     fprintf([msg '%%']);
end

tau=[];
for i=2:length(AMI)-1
    if (AMI(i-1,2)>=AMI(i,2))&&(AMI(i,2)<=AMI(i+1,2))
        tau(end+1,:)=AMI(i,:);
    end
end
ind=find(AMI(:,2)<=(0.2*AMI(1,2)),1,'first'); % finds lag at 20% initial AMI
if ~isempty(ind)
    tau(end+1,:)=AMI(ind,:);
end
if isempty(tau)
    if L*1.5>length(x/3)
        tau=9999;
    else
        fprintf('max lag needed to be increased for AMI_Thomas\n')
        [tau,AMI]=AMI_Thomas20150901(x,floor(L*1.5));
    end
end

function AMI = average_mutual_information(data) 
% function AMI = average_mutual_information(data) 
% Calculates average mutual information between 
% two 
% columns of data. It uses kernel density 
% estimation, 
% with a globally adjusted Gaussian kernel. 
% 
% Input should be an n-by-2 matrix, with data sets 
% in adjacent 
% column vectors. 
% 
% Output is a scalar. 
n = length(data); 
X = data(:,1); 
Y = data(:,2); 
% Example below is for normal reference rule in 
% 2 dims, Scott (1992). 
hx = std(X)/(n^(1/6)); 
hy = std(Y)/(n^(1/6)); 
% Compute univariate marginal density functions. 
P_x = univariate_kernel_density(X, X, hx); 
P_y = univariate_kernel_density(Y, Y, hy); 
% Compute joint probability density. 
JointP_xy = bivariate_kernel_density(data, data, hx, hy); 
AMI = sum(log2(JointP_xy./(P_x.*P_y)))/n; 

function y = univariate_kernel_density(value, data, window) 
% function y = univariate_kernel_density(value, 
% data, window) 
% Estimates univariate density using kernel 
% density estimation. 
% Inputs are: value (m-vector), where density is 
% estimated; 
%             data (n-vector), the data used to 
%             estimate the density; 
%          window (scalar), used for the width of 
%             density estimation. 
% Output is an m-vector of probabilities. 
h = window; 
n = length(data); 
m = length(value); 
% We use matrix operations to speed up computation 
% of a double-sum. 
Prob = zeros(n, m); 
G = Extended(value, n); 
H = Extended(data', m); 
Prob = normpdf((G - H)/h); 
fhat = sum(Prob)/(n*h); 
y = fhat';

function y = bivariate_kernel_density(value, data, Hone, Htwo) 
% function y = bivariate_kernel_density(value, 
% data, Hone, Htwo) 
% Calculates bivariate kernel density estimates 
% of probability. 
% Inputs are: value (m x 2 matrix), where density 
% is estimated; 
%           data (n x 2 matrix), the data used to 
%             estimate the density; 
%          Hone (scalar) and Htwo (scalar) to use 
%           for the widths of density estimation. 
% Output is an m-vector of probabilities estimated 
% at the values in ’value’. 
s = size(data); 
n = s(1); 
t = size(value); 
number_pts = t(1); 
rho_matrix = corr(data); 
rho = rho_matrix(1,2); 
% The adjusted covariance matrix: 
W = [Hone^2 rho*Hone*Htwo; rho*Hone*Htwo Htwo^2]; 
Differences = linear_depth(value,-data); 
prob = mvnpdf(Differences,[0 0],W); 
Cumprob = cumsum(prob); 
y(1) = (1/n)*Cumprob(n); 
for i = 2:number_pts 
index = n*i; 
y(i) = (1/(n))* (Cumprob(index)-Cumprob(index - n)); 
i = i + 1; 
end 
y = y'; 

function y = linear_depth(feet, toes) 
% linear_depth takes a matrix ‘feet’ and lengthens 
% it in blocks, takes a matrix ‘toes’ and lengthens 
% it in Extended repeats, and then adds the
% lengthened ‘feet’ and ‘toes’ matrices to achieve 
% all sum combinations of their rows. 
% feet and toes have the same number of columns 
if size(feet, 2) == size(toes, 2) 
a = size(feet, 1); 
b = size(toes, 1); 
Blocks = zeros(a*b, size(toes, 2)); 
Bricks = Blocks; 
for i = 1:a 
Blocks((i-1)*b + 1: i*b,:) = Extended(feet(i,:),b); 
Bricks((i-1)*b + 1: i*b,:) = toes; 
i = i + 1; 
end 
end 
y = Blocks + Bricks; 

function y = Extended(vector,n) 
% Takes an m-dimensional row vector and outputs an 
% n-by-m matrix with n-many consecutive repeats of 
% the vector. Similarly,  it takes an 
% m-dimensional column vector and outputs an 
% m-by-n matrix. 
% Else, it returns the original input. 
M = vector; 
if size(vector,1) == 1 
M = zeros(n,length(vector)); 
for i = 1:n 
M(i,:) = vector; 
i = i + 1; 
end 
end 
if size(vector,2) == 1 
M = zeros(length(vector),n); 
for i = 1:n 
M(:,i) = vector; 
i = i + 1; 
end 
end 
y = M;
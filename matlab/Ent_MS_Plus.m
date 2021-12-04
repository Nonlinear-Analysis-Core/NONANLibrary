function [ RCMSE, CMSE, MSE, MSFE, GMSE ] = Ent_MS_Plus( x, tau, m, r )
% [ RCMSE, CMSE, MSE, MSFE ] = RCMS_Ent( x, tau, m, r )
% inputs - x, single column time seres
%        - tau, greatest scale factor
%        - m, length of vectors to be compared
%        - R, radius for accepting matches (as a proportion of the
%             standard deviation)
% output - RCMSE, Refined Composite Multiscale Entropy
%        - CMSE, Composite Multiscale Entropy
%        - MSE, Multiscale Entropy
%        - MSFE, Multiscale Fuzzy Entropy
%        - GMSE, Generalized Multiscale Entropy
% Remarks
% - This code finds the Refined Composite Multiscale Sample Entropy,
%   Composite Multiscale Entropy, Multiscale Entropy, Multiscale Fuzzy
%   Entropy and Generalized Multiscale Entropy of a data series using the
%   methods described by - Wu, Shuen-De, et al. 2014. "Analysis of complex
%   time series using refined composite multiscale entropy." Physics
%   Letters A. 378, 1369-1374.
% - Each of these methods calculates entropy at different scales. These
%   scales range from 1 to tau in increments of 1.
% - The Complexity Index (CI) is not calculated by this code. Because the scales
%   are incremented by 1 the C is the summation of all the elements in each
%   array. For example the CI of MSE would be sum(MSE).
% 20170828 Created by Will Denton, bmchnonan@unomaha.edu
% 20201001 Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Modifed to calculate all scales in this single code instead of
%            needing to be in an external for loop.
%% Begin Code

dbstop if error

R = r*std(x);
N = length(x);

for i=1:tau
    
    %Coarse-graining for GMSE
    o2 = zeros(length(1:i),length(1:N/i));
    for j = 1:N/i
        for k = 1:i
            try
                o2(k,j) = var(x((j-1)*i+k:j*i+k-1));
            catch
                o2(k,j) = NaN;
            end
        end
    end
    GMSE(i,1) = Samp_Ent(o2(1,:),m,r);
    
    %Coarse-graining for MSE and derivatives
    y_tau_kj = zeros(length(1:i),length(1:N/i));
    for j = 1:N/i
        for k = 1:i
            try
                y_tau_kj(k,j) = 1/i*sum(x((j-1)*i+k:j*i+k-1));
            catch
                y_tau_kj(k,j) = NaN;
            end
        end
    end
    
    %Multiscale Entropy (MSE)
    MSE(i,1) = Samp_Ent(y_tau_kj(1,~isnan(y_tau_kj(1,:))),m,R);
    
    %Multiscale Fuzzy Entropy (MFE)
    MSFE(i,1) = Fuzzy_Ent(y_tau_kj(1,~isnan(y_tau_kj(1,:))),m,R,2);
    
    %Composite Multiscale Entropy (CMSE)
    CMSE(i,1) = 0;
    for k = 1:i
        [~,nm(k,1),nm1(k,1)] = Samp_Ent(y_tau_kj(k,~isnan(y_tau_kj(k,:))),m,R);
        CMSE(i,1) = CMSE(i,1)+1/i*-log(nm1(k,1)/nm(k,1));
    end
    
    %Refined Composite Multiscale Entropy (RCMSE)
    n_m1_ktau = 1/i*sum(nm1);
    n_m_ktau = 1/i*sum(nm);
    RCMSE(i,1) = -log(n_m1_ktau/n_m_ktau);
    
end
end

function [SE,sum_nm,sum_nm1] = Samp_Ent(data,m,r)
% [SE,sum_nm,sum_nm1] = Samp_Ent(data,m,r)
% This is a faster version of the previous code - Samp_En.m

% inputs     - data, single column time seres
%            - m, length of vectors to be compared
%            - R, radius for accepting matches (as a proportion of the
%                 standard deviation)

% output     - SE, sample entropy
%            - sum_nm, total number of matches for vector length m
%            - sum_nm1, total number of matches for vector length m+1
%
% Remarks
% This code finds the sample entropy of a data series using the method
% described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
% time-series analysis using approximate entropy and sample entropy."
% Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.
%

% J McCamley May, 2016
% W Denton August, 2017 (Made count total number of matches for each vector length, necessary for CMSE and RCMSE)

N = length(data);
dij=zeros(N-m,m+1);
Bm=zeros(N-m,1);
Am=zeros(N-m,1);
sum_nm = 0;
sum_nm1 = 0;
for i = 1:N-m
    for k = 1:m+1
        dij(:,k) = abs(data(1+k-1:N-m+k-1)-data(i+k-1));
    end
    dj = max(dij(:,1:m),[],2);
    dj1 = max(dij,[],2);
    d = find(dj<=r);
    d1 = find(dj1<=r);
    nm = length(d)-1; % subtract the self match
    sum_nm = sum_nm+nm;
    Bm(i) = nm/(N-m);
    nm1 = length(d1)-1; % subtract the self match
    sum_nm1 = sum_nm1+nm1;
    Am(i) = nm1/(N-m);
end
Bmr = sum(Bm)/(N-m);
Amr = sum(Am)/(N-m);
SE = -log(Amr/Bmr);
end

function [FuzzyEn] = Fuzzy_Ent(series,dim,r,n)
%{
Function which computes the Fuzzy Entropy (FuzzyEn) of a time series. The
alogorithm presented by Chen et al. at "Charactirization of surface EMG
signal based on fuzzy entropy" (DOI: 10.1109/TNSRE.2007.897025) has been
followed.

INPUT:
        series: the time series.
        dim: the embedding dimesion employed in the SampEn algorithm.
        r: the width of the fuzzy exponential function.
        n: the step of the fuzzy exponential function.

OUTPUT:
        FuzzyEn: the FuzzyEn value.

PROJECT: Research Master in signal theory and bioengineering - University of Valladolid

DATE: 11/10/2014

VERSION: 1

AUTHOR: Jess Monge lvarez
%}
%% Checking the ipunt parameters:
control = ~isempty(series);
assert(control,'The user must introduce a time series (first inpunt).');
control = ~isempty(dim);
assert(control,'The user must introduce a embbeding dimension (second inpunt).');
control = ~isempty(r);
assert(control,'The user must introduce a width for the fuzzy exponential function: r (third inpunt).');
control = ~isempty(n);
assert(control,'The user must introduce a step for the fuzzy exponential function: n (fourth inpunt).');

%% Processing:
% Normalization of the input time series:
% series = (series-mean(series))/std(series);
N = length(series);
phi = zeros(1,2);
% Value of 'r' in case of not normalized time series:
r = r*std(series);

for j = 1:2
    m = dim+j-1; % 'm' is the embbeding dimension used each iteration
    % Pre-definition of the varialbes for computational efficiency:
    patterns = zeros(m,N-m+1);
    aux = zeros(1,N-m+1);
    
    % First, we compose the patterns
    % The columns of the matrix 'patterns' will be the (N-m+1) patterns of 'm' length:
    if m == 1 % If the embedding dimension is 1, each sample is a pattern
        patterns = series;
    else % Otherwise, we build the patterns of length 'm':
        for i = 1:m
            patterns(i,:) = series(i:N-m+i);
        end
    end
    % We substract the baseline of each pattern to itself:
    for i = 1:N-m+1
        patterns(:,i) = patterns(:,i) - (mean(patterns(:,i)));
    end
    
    % This loop goes over the columns of matrix 'patterns':
    for i = 1:N-m
        % Second, we compute the maximum absolut distance between the
        % scalar components of the current pattern and the rest:
        if m == 1
            dist = abs(patterns - repmat(patterns(:,i),1,N-m+1));
        else
            dist = max(abs(patterns - repmat(patterns(:,i),1,N-m+1)));
        end
        % Third, we get the degree of similarity:
        simi = exp(((-1)*((dist).^n))/r);
        % We average all the degrees of similarity for the current pattern:
        aux(i) = (sum(simi)-1)/(N-m-1); % We substract 1 to the sum to avoid the self-comparison
    end
    
    % Finally, we get the 'phy' parameter as the as the mean of the first
    % 'N-m' averaged drgees of similarity:
    phi(j) = sum(aux)/(N-m);
end

FuzzyEn = log(phi(1)) - log(phi(2));

end %End of the 'FuzzyEn' function

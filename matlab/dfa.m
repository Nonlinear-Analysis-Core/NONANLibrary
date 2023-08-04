function [a, r2, out_a, out_l] = dfa(ts, n_min, n_max, n_length, plotOption)
% [a, r2, n_bp, n, n10, F,F10, F_fit, n_fit, Xave, Yave, logF_fit] = dfa20200727(ts, n_min, n_max, n_length, plotOption)
% Inputs  - ts, input time series
%           n_min, minimum box size (default n_min = 10)
%           n_max, maximum box size (default n_max = N/8)
%           n_length, number of points to sample best fit
%           plotOption, plot log F vs. log n? (default true)
% Outputs - a, DFA scaling exponent
%           r2, r^2 value for best fit line
%         - out_a, contains incremental information on fluctuations
%           - column 1, box sizes
%           - column 2, fluctuation for a given box size
%         - out_l, contains incremental information on line fitting
%           - column 1, bin centers in 10x
%           - column 2, average fluctuation for bin in 10x
%           - column 3, best fit fluctuation for bin in 10x
% Remarks           
% - Given time series, returns detrended fluctuation analysis scaling 
%   exponent.
% Future Work
% - This code is newly compiled had has not seen many updates. There are
%   numerous areas of improvement: speed, readability, accuracy, ease of
%   use.
% References:
% - Damouras, S., Chang, M. D., Sejdi, E., & Chau, T. (2010). An empirical 
%   examination of detrended fluctuation analysis for gait data. Gait & 
%   posture, 31(3), 336-340.
% - Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
%   temporal correlations under extreme random dilution. Physica A: 
%   Statistical Mechanics and its Applications, 389(24), 5573-5580.
% - Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
%   Quantification of scaling exponents and crossover phenomena in 
%   nonstationary heartbeat time series. Chaos: An Interdisciplinary Journal 
%   of Nonlinear Science, 5(1), 82-87.
% Prior    - Created by Naomi Kochi
% Jul 2017 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - It is believed Naomi Kochi wrote the original version of this
%            code around 2010 and it strongly resembles the version used in
%            2017.
%          - Reformated comments. Combined the various functions into one 
%            script. Moved checking if ts is a column vector to dfa.m.
%          - Some code was commented out that cut down the number of box
%            sizes and was said to cut the processing time. Re-implementing
%            this produced NaN values. The commented-out lines were removed
%            entirely.
%          - Cut number of outputs from 12 to 4. Some of these are
%            log10/10x of other outputs so they are redundant.
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
%% Begin Code

dbstop if error

%% Input handling -------------------- Inputs are all out of numeric order

if size(ts, 1) > size(ts, 2) % ts should be row vector
    ts = ts';
end

if nargin < 5
    plotOption = 0; % do not produce plot by default
end

if nargin < 4
    n_length = 18; % default number of points to sample best fit
end

if nargin < 3
    n_max = length(ts)/9; % n_max as recommended by Damouras, et al., 2010
end

if nargin < 2
    n_min = 16; % n_min as recommended by Damouras, et al., 2010
end

n_bp =linspace(log10(n_min), log10(n_max), n_length+1)'; % calculate breakpoints for fit line (spaced evenly in log space)

n = (10:length(ts)/8)'; % calculate F for every possible n (makes plot nice) --------------- This seems to override the n_min and n_max arguments

F = dfa_fluct(ts, n); % calculate F for every n

[F_fit, n_fit, a, r2, logF_fit] = dfa_fit_average(n, F, n_bp); % fit line over n_fit using averaging method

if plotOption
    dfa_plot(n, F, n_fit, F_fit, logF_fit); % produce plot of log F vs log n
    string = ['\alpha = ', num2str(a, '%.2f'), newline, 'r^2 = ', num2str(r2, '%.2f')];
    x_lim = get(gca, 'XLim');
    y_lim = get(gca, 'YLim');
    text(x_lim(2)/2, y_lim(1)*2, string, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom')
end

out_a=[n,F];
out_l=[n_fit,F_fit,logF_fit];

end

function F = dfa_fluct(ts, n)
% F = dfa_fluct(ts, n)
% Inputs  - ts, input time series
%         - n, vector of box sizes
% Outputs - F, vector of fluctuations
% Remarks
% - Returns average detrended fluctuations in time series as a function of 
%   box size.
% - Deals with outlier and NaN values by removing them from the
%   analysis while preserving the temporal order, essentially leaving a 
%   'hole' in the data, rather than truncating the time series. This method 
%   is shown in Mirzayof and Ashkenazy (2010) to preserve alpha in 
%   experimental data even under extreme (>90%) dilution. -------------------------- Questionable decision to remove and then exptrapolate the data to replace
% References
% - Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
%   temporal correlations under extreme random dilution. Physica A: 
%   Statistical Mechanics and its Applications, 389(24), 5573-5580.
% - Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
%   Quantification of scaling exponents and crossover phenomena in 
%   nonstationary heartbeat time series. Chaos: An Interdisciplinary 
%   Journal of Nonlinear Science, 5(1), 82-87.
% Prior - Created by Naomi Kochi
% Jul 2020 - Modified by Ben Senderling
%          - Reformated comments. Removed commented out code assuming it 
%            was never used. Removed clears and replaced with array 
%            pre-allocation.
%          - Removed reliance on inpaint_nans from File Exchange with
%            native fillmissing,
%% Begin code

zero_th = .000001; % F < this value is equivalent to F = 0

F2=zeros(length(n),1);
F=zeros(length(n),1);
for nn = 1:length(n)
    num_boxes = floor(length(ts)/n(nn));
    B = ts(1:num_boxes*n(nn))';
    
    % "the ... time series (of total length N) is first
    % integrated, y(k) = sum(i=1:k)(B(i) - B_ave), where B(i) is the ith
    % [value of the time series] and B_ave is the average [value]" --------------------- I like the fact that the written description is used in the comments

    B_ave = nanmean(B);
    B_nonan = fillmissing(B,'linear','SamplePoints',1:length(B)); % deal with NaN values for integration step ----------------- this step
    y_nonan = cumsum(B_nonan - B_ave);
    y = y_nonan;
    y(isnan(B)) = NaN; % replace NaN values in integrated series
    
    y_n=zeros(num_boxes,1);
    % "the integrated time series is divided into boxes of equal length, n"
    for k = 0:num_boxes - 1
        
        % "in each box of length n, a least-squares line is fit to the data
        % (representing the trend in that box) ... denoted by y_n(k)"
        X = k*n(nn) + 1:(k + 1)*n(nn);
        Y = y_nonan(X); % fit using interpolated series
        m_b = polyfit(X, Y, 1);
        y_n(X) = polyval(m_b, X);
    end
    
    % "detrend the integrated time series, y(k), by subtracting the local
    % trend, y_n(k) .... The root-mean-square fluctuation ... is calculated
    % by [Equation 1]"
    F2(nn) = nanmean((y - y_n).^2); % ignores NaN values ---------------------- if you have already removed the NANs this is not necessary
    
    F(nn) = sqrt(F2(nn));
    
% "This computation is repeated over all time scales (box sizes)"
end

% "the fluctuations can be characterized by a scaling exponent a, the slope
% of the line relating log F(n) to log n"
F(F < zero_th) = NaN; % removes F = 0 from linear fit

end

function [F_fit, n_fit, a, r2, logF_fit] = dfa_fit_average(n, F, n_bp)
% [F_fit, n_fit, a, r2, Xave, Yave, logF_fit] = dfa_fit_average(n, F, n_bp)
% Inputs  - n, vector of box sizes
%         - F, vector of fluctuations
%         - n_bp, vector of bin break points
% Outputs - F_fit, vector of best fit values
%         - n_fit, vector of bin centers
%         - a, slope of best fit line
%         - r2, r^2 of best fit line
% Remarks
% - DFA_FIT Returns best fit line, slope and r^2 of log-log of fluctuations
%   vs. box size over given range of box sizes, using average log(F) over a
%   range of log(n_i) to log(n_i+1).
% References:
% - Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
%   Quantification of scaling exponents and crossover phenomena in 
%   nonstationary heartbeat time series. Chaos: An Interdisciplinary 
%   Journal of Nonlinear Science, 5(1), 82-87.
% Prior    - Created by Naomi Kochi
% Jul 2020 - Modified by Ben Senderling
%          - Reformated comments. Added array pre-allocation.
%% Begin code
% "the fluctuations can be characterized by a scaling exponent a, the slope
% of the line relating log F(n) to log n"

F10=log10(F)';
n10=log10(n)';

Xave=zeros(length(n_bp)-1,1); % --------------- why can't this be implemented in the main loop?
Yave=zeros(length(n_bp)-1,1);
for i = 1:length(n_bp) -1
    Xave(i) = mean((n_bp(i:i+1)))'; % calculate center of bin
    Yave(i) = mean((F10(n10 >= n_bp(i) & n10 < n_bp(i+1))))'; % average all values in bin
end
[P, S] = polyfit(Xave, Yave, 1); % can't fit log(0) or log(NaN)
logF_fit = polyval(P, Xave);

F_fit = 10.^Yave; % return best fit 'line' in linear space
n_fit = 10.^Xave;
logF_fit=10.^logF_fit;

a = P(1); % return DFA scaling exponent (slope of best fit line)
r2 = 1 - S.normr^2 / norm(Yave(isfinite(Yave)) - mean(Yave(isfinite(Yave))))^2; % return r^2 of best fit line

end

function hAxObj = dfa_plot(n, F, n_fit, F_fit, logF_fit)
% hAxObj = dfa_plot(n, F, n_fit, F_fit, logF_fit)
% Inputs  - n, vector of box sizes
%         - F, vector of fluctuations 
%         - n_fit, vector of box sizes to fit
%         - F_fit, vector of best fit values
% Outputs - hAxObj: handle to axis object
% Remarks
% - DFA_PLOT Plots fluctuations as a function of box size on log-log axis 
%   with best-fit line.
% Prior    - Created by Naomi Kochi
% Jul 2020 - Modified by Ben Senderling
%          - Reformated comments. Removed commented out code assuming it
%            was never used.
%% Begin Code

loglog(n, F, 'b.', n_fit, F_fit, 'ro', n_fit, logF_fit, '-r') % ----------------- this seems like it could also be in the one function

hAxObj = handle(gca);
x_lim = get(hAxObj, 'XLim');
y_lim = get(hAxObj, 'YLim');
x_decades = log10(x_lim(2)/x_lim(1));
y_decades = log10(y_lim(2)/y_lim(1));
if x_decades >= y_decades % set x- and y-axis to be same size (so a = 1 is on first diagonal)
    y_lim_new = y_lim(1)*10^x_decades;
    set(hAxObj, 'YLim', [y_lim(1), y_lim_new]);
else
    x_lim_new = x_lim(1)*10^y_decades;
    set(hAxObj, 'XLim', [x_lim(1), x_lim_new]);
end
axis square
xlabel('log box size')
ylabel('log RMS fluctuations')

end

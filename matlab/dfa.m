function [scales, fluctuation, alpha] = dfa(data, scales, order, plot)
    % Perform Detrended Fluctuation Analysis on data
    % Parameters:
    %   data (column vector): 1D numeric array containing time series data
    %   scales (numeric array): Array of scales to calculate fluctuations
    %   order (integer): Order of polynomial fit (1 = linear fit)
    %   plot (logical): Flag to enable or disable plotting (default = true)
    %
    % Returns:
    %   scales: The scales that were entered as input
    %   fluctuations: Variability measured at each scale with RMS
    %   alpha value: Value quantifying the variability in the time series
    %
    % References:
    %   Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
    %   Quantification of scaling exponents and crossover phenomena in 
    %   nonstationary heartbeat time series. Chaos: An Interdisciplinary 
    %   Journal of Nonlinear Science, 5(1), 82-87.

% Copyright 2023 Nonlinear Analysis Core, Center for Human Movement
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

%% ========================================================================
%                          ------ EXAMPLE ------

%      - Generate random data
%      data = randn(5000,1); 
      
%      - Create a vector of the scales you want to use
%      scales = [10, 20, 40, 80, 160, 320, 640, 1280, 2560];
      
%      - Set a detrending order. Use 1 for a linear detrend.
%      order = 1;
      
%      - run dfa function
%      [s, f, a] = dfa(data, scales, order, 1)

%% ========================================================================

% Check if the data is a column vector and if not, transpose to make it
% one.
if size(data, 2) > size(data, 1) % data should be column vector
    data = data';
end


    % Integrate the data
    integrated_data = cumsum(data - mean(data));

    fluctuation = zeros(size(scales));

    for idx = 1:length(scales)
        scale = scales(idx);

        % Divide data into non-overlapping chunks of size 'scale'
        chunks = floor(length(data) / scale);
        %disp(chunks)
        ms = 0;

        for i = 1:chunks
            chunk_start = (i - 1) * scale + 1;
            chunk_end = i * scale;
            this_chunk = integrated_data(chunk_start:chunk_end);
            x = 1:length(this_chunk);

            % Fit polynomial (default is linear, i.e., order=1)
            coeffs = polyfit(x, this_chunk, order);
            fit = polyval(coeffs, x);

            % Detrend and calculate RMS for this chunk
            ms = ms + mean((this_chunk' - fit).^2);
        end

        % Calculate average RMS for this scale
        fluctuation(idx) = sqrt(ms./chunks);
    end

    % Perform linear regression on the log-log data
    log_scales = log(scales);
    log_fluctuation = log(fluctuation);

    % Check for NaN values in log_fluctuation and remove them
    nan_indices = isnan(log_fluctuation);
    log_scales(nan_indices) = [];
    log_fluctuation(nan_indices) = [];

    p = polyfit(log_scales, log_fluctuation, 1);
    alpha = p(1);

    % Calculate R-squared value
    y_fit = polyval(p, log_scales);
    ssr = sum((log_fluctuation - y_fit).^2);
    sst = sum((log_fluctuation - mean(log_fluctuation)).^2);
    rsquared = 1 - ssr / sst;


    if plot
        % Plot scales vs. fluctuation values
        loglog(scales, fluctuation, 'o-k', MarkerFaceColor='red', MarkerSize = 8, LineWidth= 1.5);
        hold on;
        xlabel('Scale (log)');
        ylabel('Fluctuation (log)');

        % Add alpha and R-squared as text labels
        str = ['Alpha = ', num2str(alpha, '%.3f') newline 'R^2 = ', num2str(rsquared, '%.3f')];
        dim = [.7 .05 .2 .2]; % textbox location
        annotation('textbox',dim,'String',str,'FitBoxToText','on', 'LineWidth', 1, 'BackgroundColor', 'white');


        % Add the regression line to the plot
        loglog(scales, exp(polyval(p, log_scales)), 'Color', '#4DBEEE', 'LineStyle', '--', 'LineWidth', 1.5);

        title('Detrended Fluctuation Analysis');
        grid on;
        grid minor;
    end
end

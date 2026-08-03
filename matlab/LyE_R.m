function [varargout]=LyE_R(X,Fs,tau,dim,varargin)
% [out]=LyE_R20200619(X,Fs,tau,dim)
% inputs  - X, If this is a single dimentional array the code will use tau
%              and dim to perform a phase space reconstruction. If this is
%              a multidimentional array the phase space reconstruction will
%              not be used.
%         - Fs, sampling frequency in units s^-1
%         - tau, time lag
%         - dim, embedding dimension
% outputs - out, contains the starting matched pairs and the average line
%                divergence from which the slope is calculated. The matched
%                paris are columns 1 and 2. The average line divergence is
%                column 3.
% [LyES,LyEL,out]=LyE_Rosenstein_FC(X,Fs,tau,dim,slope,MeanPeriod,file)
% inputs  - slope, a four element array with the number of periods to find
%                  the regression lines for the short and long LyE. This is
%                  converted to indexes in the code.
%         - MeanPeriod, used in the slope calculation to find the short and
%                       long Lyapunov Exponents.
%         - file, a boolean specifying if a figure should be created
%                 displaying the regression lines. This figure is visible
%                 by default.
% [...]=LyE_R(...,'TheilerWindow',w)
% inputs  - w, half-width in SAMPLES of the temporal exclusion around each
%              point when searching for its nearest neighbour (the Theiler
%              window). Candidates within +/- w of a point are excluded, so
%              the match is a genuine return to the neighbourhood rather than
%              an adjacent sample on the same trajectory segment. The default
%              is round(tau*0.8), which preserves previous behaviour. Set it
%              from the dominant period, or from the first minimum of the
%              autocorrelation or AMI, when tau is small relative to the
%              orbit. Accepted in both the four-argument and the
%              slope-fitting call forms.
% outputs - LyES, short/local lyapunov exponent
%         - LyEL, long/orbital lyapunov exponent
% Remarks
% - This code is based on the algorithm presented by Rosenstein et al,
%   1992.
% - Recommendations for the slope input can be found in the references
%   below. It is possible a long term exponent can not be found with your
%   inputs. If your selection exceeds the length of the data LyEL will
%   return as a NaN.
% Future Work
% - It may be possible to sped it up conciderably by re-organizing the for
%   loops. A database for the matched points would need to be created.
% References
% - Rosentein, Collins and De Luca; "A practical method for calculating
%   largest Lyapunov exponents from small data sets;" 1992
% - Yang and Pai; "Can stability really predict an impending slip-related
%   fall among older adults?", 2014
% - Brujin, van Dieen, Meijer, Beek; "Statistical precision and sensitivity
%   of measures of dynamic gait stability," 2009
% - Dingwell, Cusumano; "Nonlinear time series analysis of normal and
%   pathological human walking," 2000
% Version History
% Jun 2008 - Created by Fabian Cignetti
%          - It is suspected this code was originally written by Fabian
%            Cignetti
% Apr 2017 - Revised by Ben Senderling
%          - Added comments section. Automated slope calculation. Added
%            calculation of orbital exponent.
% Jun 2020 - Revised by Ben Senderling
%          - Incorporated the subroutines directly into the code since they
%            were only used in one location. Converted various for loops
%            into indexed operations. This significantly improved the
%            speed. Added if statements to compensate for errors with the
%            orbital LyE. If the data is such an orbital LyE would not be
%            found with the hardcoded regression line bounds. Made this 
%            slope and the file input optional. Removed the MeanPeriod as 
%            an imput and made it a calculation in the code. Added the out
%            array so the matched pairs and average line distance can be
%            reviewed, or used to finf the slope. Removed the progress
%            output to the command window since it was sped up
%            conciderably. Edited the figure output. Added code that allows
%            a multivariable input to be entered as X.
% Aug 2020 - Revised by Ben Senderling
%          - Removed mean period calculation and turned it into an input.
%            This varies too widely between time series to have it
%            automatically calculated in the script. It was replaced with
%            tau to find paired points.
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

% If the code errors MATLAB will enter debug mode so the issue can be more
% easily identified.
dbstop if error

% Pull the optional 'TheilerWindow' name-value pair out of varargin before
% the legacy positional arguments {slope, MeanPeriod, file} are read, so both
% call forms keep working unchanged.
theilerWindow=[];
k=1;
while k < numel(varargin)
    if (ischar(varargin{k}) || isstring(varargin{k})) && ...
            strcmpi(varargin{k},'TheilerWindow')
        theilerWindow=varargin{k+1};
        varargin([k k+1])=[];
    else
        k=k+1;
    end
end
if isempty(theilerWindow)
    % Default preserves the historical behaviour: exclude +/- round(tau*0.8).
    theilerWindow=round(tau*0.8);
end
if ~isscalar(theilerWindow) || ~isfinite(theilerWindow) || theilerWindow < 0
    error('LyE_R:theilerWindow', ...
        'TheilerWindow must be a non-negative finite scalar, in samples.');
end
theilerWindow=round(theilerWindow);

% Checked that X is vertically oriented. If X is a single or multiple
% dimentional array the length is assumed to be longer than the width. It
% is re-oriented if found to be different.
[r,c]=size(X);
if c > r
    X=X';
end

%% Checks if a multidimentional array was entered as X.
if size(X,2)>1
    
    M=length(X);
    Y=X;
    
else
    
    % Calculate useful size of data
    N=length(X);
    M=N-(dim-1)*tau;
    
    Y=zeros(M,dim);
    for j=1:dim
        Y(:,j)=X((1:M)+(j-1)*tau)';
    end
    
end

%% Find nearest neighbors

IND2=zeros(M,1);
for i=1:M
    % Find nearest neighbor.
    Yinit=repmat(Y(i,:),M,1);
    Ydiff=(Yinit-Y(1:M,:)).^2;
    Ydisti=sqrt(sum(Ydiff,2));
    
    % Exclude points too close in time (the Theiler window). These are the
    % same trajectory segment rather than a separate pass near the same
    % point, so pairing with them measures interpolation, not divergence.
    %
    % The exclusion marker must be larger than any real distance. It was
    % previously the literal constant 1e5, which is not: once the data are
    % scaled so that inter-point distances exceed 1e5, the "excluded" points
    % become the MINIMUM and every point is paired with its own immediate
    % temporal neighbour. Measured on a Lorenz series, the exponent fell to
    % 8% of its correct value at scale 1e6, with 90% of pairs inside the
    % exclusion window - silently, and in the direction that reads as "more
    % stable". Inf cannot be beaten by any finite distance at any scale.
    range_exclude=i-theilerWindow:i+theilerWindow;
    range_exclude=range_exclude(range_exclude>=1 & range_exclude<=M);
    Ydisti(range_exclude)=Inf;
    
    % find minimum distance point for first pair
    [~,IND2(i,1)]=min(Ydisti);
end
% Preallocate all three columns. Column 3 is written conditionally below, so
% without this the function returns a 2-column array whenever no window has a
% non-zero distance sum, and the caller's out(:,3) fails somewhere unrelated.
out=[(1:M)',IND2,zeros(M,1)]; % Matched pairs and average line divergence.

%% Calculate distances between matched pairs.

DM=zeros(M-1,M-1);
for i=1:length(IND2)-1
    
   % The data can only be propagated so far from the matched pair.
    EndITL=M-IND2(i);
    if (M-IND2(i))>(M-i)
        EndITL=M-i;
    end
    
    % Finds the distance between the matched paris and their propagated
    % points to the end of the useable data.
    DM(1:EndITL,i)=sqrt(sum((Y(i+1:EndITL+i,:)-Y(IND2(i)+1:EndITL+IND2(i),:)).^2,2));
    
end

%% Calculaets the average line divergence.
[r,~]=size(DM);
for i=1:r
    distanceM=DM(i,:);
    if sum(distanceM)~=0
        AveLnDiv(i)=mean(log(distanceM(distanceM>0)));
        out(i,3)=AveLnDiv(i);
    end
end

%% Find LyES and LyEL

if isempty(varargin)
    
    varargout{1}=out;
    
else
    
    slope=varargin{1};
    MeanPeriod=varargin{2};
    file=varargin{3};
    
    
    time=(0:length(AveLnDiv)-1)/Fs/MeanPeriod;
    
    % The values in slope are assumed to be the number of periods. These
    % are converted into indexes.
    if slope(1)==0
        short(1)=1; % A value of 0 periods cannot be used.
    else
        short(1)=round(slope(1)*MeanPeriod*Fs);
    end
    short(2)=round(slope(2)*MeanPeriod*Fs);
    
    long(1)=round(slope(3)*MeanPeriod*Fs);
    long(2)=round(slope(4)*MeanPeriod*Fs);
    
    % If the index chosen exceeds the length of AveLnDiv then that exponent
    % is made a NaN.
    if short(2)<=length(AveLnDiv)
        slopeinterceptS=polyfit(time(short(1):short(2)), AveLnDiv(short(1):short(2)),1);
        LyES=slopeinterceptS(1);
        timeS=time(short(1):short(2));
        LyESline=polyval(slopeinterceptS,timeS);
    else
        LyES=NaN;
    end
    
    if long(2)<=length(AveLnDiv)
        slopeinterceptL=polyfit(time(long(1):long(2)), AveLnDiv(long(1):long(2)),1);
        LyEL=slopeinterceptL(1);
        timeL=time(long(1):long(2));
        LyELline=polyval(slopeinterceptL,timeL);
    else
        LyEL=NaN;
    end
    
    varargout{1}=LyES;
    varargout{2}=LyEL;
    varargout{3}=out;
    
    %% Plot data
    
    if file==1
        H=figure;
        set(H,'visible','on')
        plot(time,AveLnDiv,'k')
        hold on
        if ~isnan(LyES)
            plot(timeS,LyESline,'r','LineWidth',2)
            text(mean(timeS),mean(LyESline),['LyE\_Short = ' num2str(LyES)])
        end
        if ~isnan(LyEL)
            plot(timeL,LyELline,'g','LineWidth',2)
            text(mean(timeL),0.8*mean(LyELline),['LyE\_Long = ' num2str(LyEL)])
        end
        hold off
        title('LyE'), xlabel('Periods (s)'), ylabel('<ln(divergence)>')
    end
    
end

function [out,LyE] = LyE_W(X,Fs,tau,dim,evolve,varargin)
%% [out,LyE] = LyE_W(X,Fs,tau,dim,evolve)
% inputs  - X, time series
%         - Fs, sampling frequency (Hz)
%         - tau, time lag
%         - dim, embedding dimension
%         - evolve, parameter of the same name from Wolf's 1985 paper. This
%           code expects a number of frames as an input.
% outputs - out, matrix detailing variables at each iteration
%         - LyE, largest lyapunov exponent
%% [out,LyE] = LyE_W20200820(X,Fs,tau,dim,evolve,SCALEMX,SCALEMN,ANGLMX,ZMULT)
%         - SCALEMX, length of which the local structure of the attractor
%           is no longer being probed
%         - SCALEMN, length below which noise predominates the attractors
%           behavior
%         - ANGLMX, maximum angle used to constrain replacements
%         - ZMULT, multiplier used to increase SCALEMX, unused in the
%           current version of the code
% Remarks
% - This code calculates the largest lyapunov exponent of a time series
%   according to the algorithm detailed in Wolf's 1985 paper. This code has
%   been aligned with his code published on the Matlab file exchange in
%   2016. It will largely find the same replacement points, the remaining
%   difference being in the replacement algorithm.
% - The varargin can be used to specify some of the secondary parameters in
%   the algorithm. All of the extra arguements must be specified if any are
%   to be specified at all. Otherwise defaults are used.
% - ZMULT is not currently used in the code but was in a previous version.
%   Its place in the subroutine inputs and outputs was kept in case it is
%   put back in.
% - It should be noted that the process in the searching algorithm has a
%   significant impact on the resulting LyE.
% - The code expects evolve to be the number of frames to use but we
%   encourage you to report this as a time-value in publications.
% Prior - Created by Shane Wurdeman, unonbcf@unomaha.edu
%       - Adapted by Brian Knarr, unonbcf@unomaha.edu
%       - The code previously was influenced heavily by the FORTRAN syntax
%         published in Wolf's 1985 paper. These were modified to better
%         take advantage of MATLAB and speed up the code.
% Mar 2017 - Modified by Ben Senderling, unonbcf@unomaha.edu
%          - Changed parameter "n" to "evolve."
%          - Changed "ZMULT" back to 1.
%          - Aligned the code with Wolf's Matlab File Exchange submission
%            to find the same replacement points. This is now essential his
%            algorithm but retains the speed of previous versions.
% Apr 2019 - Modified by Ben Senderling, unonbcf@unomaha.edu
%          - Changed line 'range_exclude = range_exclude(range_exclude>=1 &
%            range_exclude<=NPT);' to say '>=1' instead of '>1' to prevent
%            self matches with the first point. This was indirectly
%            accounted for by setting distances less than SCALEMN to 0.
%          - '<SCALEMN' was removed from the code entirely and replaced with a
%            '<=0'. This was checked against joint angles and EMG data. The
%            change did not result in different pairs. This also removes an
%            input.
% Jul 2021 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Removed print commands to update command window.
%% Begin Code

if isempty(varargin)
    SCALEMX = (max(max(X))-min(min(X)))/10;
    ANGLMX = 30*pi/180;
    ZMULT = 1;
elseif nargin==8
    SCALEMX = varargin{1};
    ANGLMX = varargin{2};
    ZMULT = varargin{3};
else
    error('not enough input arguements')
end

DT=1/Fs;

%% Set up data

ITS=0;
distSUM=0;

if size(X,2)==1
    
    Y=psr_deneme(X,dim,tau);
    NPT=length(X)-(dim-1)*tau-evolve; % Size of useable data % NPT=length(X)-(dim)*tau-evolve; (BS)
    Y=Y(1:NPT+evolve,:);
    
else
    Y=X;
    NPT=length(Y)-evolve;
end
%% Start analysis

out=zeros(floor(NPT/evolve),9);
thbest=0;
OUTMX=SCALEMX;

for i=1:evolve:NPT
    
    current_point = i;
    
    % Find first pair
    if current_point==1
        
        % Distance from current point to all other points
        Yinit = repmat(Y(current_point,:),NPT,1);
        Ydiff =( Yinit - Y(1:NPT,:) ) .^2;
        Ydisti = sqrt( sum(Ydiff,2) );
        
        % Exclude points too close on path and close in distance
        range_exclude = current_point-10:current_point+10;
        range_exclude = range_exclude(range_exclude>=1 & range_exclude<=NPT);
        Ydisti(Ydisti<=0) = NaN;
        Ydisti(range_exclude) = NaN;
        
        % find minimum distance point for first pair
        [~, current_point_pair] = min(Ydisti);
        
    end
    
    % calculate starting and evolved distance
    start_dist = norm(Y(current_point,:) - Y(current_point_pair,:));
    end_dist = norm(Y(current_point+evolve,:) - Y(current_point_pair+evolve,:));
    
    
    % calculate total distance so far
    distSUM=distSUM+log2(end_dist/start_dist)/(evolve*DT); % DT is sampling rate?!
    ITS = ITS+1;  % count iterations
    LyE=distSUM/ITS; % max Lyapunov exponent
    
    %   CPP(i) = current_point_pair; % Store found pairs
    
    out(floor(i/evolve)+1,:)=[ITS,current_point,current_point_pair,start_dist,end_dist,LyE,OUTMX,thbest*180/pi,ANGLMX*180/pi];
    
    ZMULT=1;
    
    if end_dist<SCALEMX
        current_point_pair=current_point_pair+evolve;
        if current_point_pair>NPT
            current_point_pair=current_point_pair-evolve;
            flag=1;
            [current_point_pair,ZMULT,ANGLMX,thbest,OUTMX] = GetNextPoint(flag,Y,current_point,current_point_pair,NPT,evolve,SCALEMX,ZMULT,ANGLMX);
        end
        continue
    end
    % find point pairing for next iteration
    flag=0;
    [current_point_pair,ZMULT,ANGLMX,thbest,OUTMX] = GetNextPoint(flag,Y,current_point,current_point_pair,NPT,evolve,SCALEMX,ZMULT,ANGLMX);
    
end

fprintf('\n')

function [next_point,ZMULT,ANGLMX,thbest,SCALEMX] = GetNextPoint(flag,Y,current_point,current_point_pair,NPT,evolve,SCALEMX,ZMULT,ANGLMX)

% Distance from evolved point to all other points
Yinit = repmat(Y(current_point+evolve,:),NPT,1);
Ydiff =( Yinit - Y(1:NPT,:) ) .^2;
Ydisti = sqrt( sum(Ydiff,2) );

% Exclude points too close on path and close in distance than noise
range_exclude = (current_point+evolve)-10:(current_point+evolve)+10;
range_exclude = range_exclude(range_exclude>=1 & range_exclude<=NPT);
Ydisti(range_exclude) = NaN;

end_dist = norm(Y(current_point+evolve,:) - Y(current_point_pair+evolve,:));

% Vector from evolved point to all other points
Vnew = repmat(Y(current_point+evolve,:),NPT,1) - Y(1:NPT,:);

% Vector from evolved point to evolved point pair
PT1 = Y(current_point+evolve,:);
PT2 = Y(current_point_pair+evolve,:);
Vcurr = PT1-PT2;

% Angle between evolved pair vector and all other vectors
cosTheta = abs((Vcurr * Vnew')'./(Ydisti*end_dist));
theta = acos(cosTheta);

% Search for next point
next_point=0;
while next_point==0
    [next_point,ZMULT,ANGLMX,thbest,SCALEMX]=find_next_point(flag,theta,Ydisti,SCALEMX,ZMULT,ANGLMX);
end

function [next_point,ZMULT,ANGLMX,thbest,SCALEMX] = find_next_point(flag,theta,Ydisti,SCALEMX,ZMULT,ANGLMX)

% Restrict search based on distance and angle
PotenDisti=Ydisti;
PotenDisti(Ydisti<=0 | theta>=ANGLMX) = NaN;

next_point=0;
if flag==0
    [~,next_point]=min(PotenDisti); % find min distance after excluding points
    % if closest angle point is within angle range -> point found and reset
    % search space
    if PotenDisti(next_point) <= SCALEMX
        ANGLMX = 30*pi/180;
        thbest=abs(theta(next_point));
        return;
    else
        next_point=0;
        flag=1;
    end
end
if flag==1
    PotenDisti=Ydisti;
    PotenDisti(Ydisti<=0) = NaN;
    [~,next_point]=min(PotenDisti);
    thbest=ANGLMX;
    
end

function Y=psr_deneme(x,m,tau,npoint)

%Phase space reconstruction
%x : time series
%m : embedding dimension
%tao : time delay
%npoint : total number of reconstructed vectors
%Y : M x m matrix
% author:"Merve Kizilkaya"
N=length(x);
if nargin == 4
    M=npoint;
else
    M=N-(m-1)*tau;
end

Y=zeros(M,m);

for i=1:m
    Y(:,i)=x((1:M)+(i-1)*tau)';
end

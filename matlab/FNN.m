function [dim, dE] = FNN(data,tau,MaxDim,Rtol,Atol,speed)
% [FN,dim] = FNN20200714(data,tau,MaxDim,Rtol,Atol)
%   data - column oriented time series
%   tau - time delay
%   MaxDim - maximum embedding dimension
%   Rtol - threshold for the first criterion
%   Atol - threshold for teh second criterion
%   speed - a 0 for the code to calculate to the MaxDim or a 1 for the code
%           to finish once a minimum is found
% Remarks
% - This code determines the embedding dimension for a time series using
%   the false nearest neighbors method.
% - Recommended values are Rtol=15 and Atol=2;
% - Reference:   "Determining embedding dimension for phase-space
%                 reconstruction using a geometrical construction",
%                 M. B. Kennel, R. Brown, and H.D.I. Abarbanel,
%                 Physical Review A, Vol 45, No 6, 15 March 1992,
%                 pp 3403-3411.
% Future Work
% - Currently there are two methods of detecting a minimal percentage of
%   false nearest neighbors. One method checks for a minima or zero
%   percentage, the other looks for a limit. Currently only dim is
%   returned. This code can be modified to use a comprimise of the two.
% Prior - Created by someone
% Feb 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
%            No changes were made to the algorithm. Checks were added to
%            provide information to the user in case of an error. The two
%            methods described in future work were also modified to work
%            cooperatively. In a previous version the second method (dim)
%            overwrote the first method (dim2).
% Sep 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
%            Previously, dim was found after the for loop, this version has
%            been modified to allow the code to find the minimum as it
%            calculates FNN. This is set within the inputs.
%            The check that was previously put in has been commented out.
% Oct 2015 - Modified by John McCamley, email: unonbcf@unomaha.edu
%          - Embedded other required functions as subroutines.
% Mar 2017 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
%          - Removed global variables in favor of passing the variables
%            from function to function directly. This significantly
%            improved performance. Checked that the calculated percentages
%            of nearest neighbors are the same as the previous version.
% May 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Added if statement checkeding data orientation.
% Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Changed indexing throughout so the input data array doesn't
%            need to be reoriented. Changing this sped the code up an
%            average 11% on 10 test signals.
%          - Removed a couple small for loops and replaced with indexed
%            operations. Was also able to remove within function and
%            replaced with a single line of code.
%          - Removed perviously commented out lines of code that were no
%            longer used.
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
%% Begin algorithm

n=length(data)-tau*MaxDim;  % # of data points to be used
RA=std(data); % the nominal "radius" of the attractor

z = data(1:n);
y = [];

% search for the nearest point, closest will be equal to itself and its
% next neighbor
m_search = 2;

indx=(1:n)';
dim=[];

dE=zeros(MaxDim,1);

for j = 1:MaxDim
    
    y = [y,z]; % Adds additional dimension.
    z = data(1+tau*j:n+tau*j);
    L = zeros(n,1);
    
    [y_model,z_model,sort_list,node_list]=kd_part(y, z, 512); % put the data into 512-point
    
    for i = 1:length(indx)
        
        yq = y(indx(i),:); % set up the next point to check
        
        % set up the bounds, which start at +/- infinity
        b_upper = Inf*ones(size(yq));
        b_lower = -b_upper;
        
        % and set up storage for the results
        pqd = Inf*ones(m_search,1);
        pqr = [];
        pqz = [];
        L_done = 0;
        
        % [pqd]=kdsearch(1,m_search,yq); % start searching at the root (node 1)
        [pqd,y_model,z_model,~,~,pqz,~,~,sort_list,node_list] = kdsearch(1,m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
        
        distance = pqz(1) - pqz(2);
        
        if abs(distance) > pqd(2)*Rtol
            L(i) = 1;
        end
        
        if sqrt(pqd(2)^2+distance^2)/RA > Atol
            L(i) = 1;
        end
        
    end
    
    dE(j) = sum(L)/n; % was FN=[FN sum(L)/n]
    
    if speed==1
        % Stops calculation if %FNN=0
        if (dE(j)<=0)
            dim=j;
            break
        end
        % Stops calculation if a minimum %FNN is detected.
        if j>=3 && ((dE(j-2)>dE(j-1)&&(dE(j-1)< dE(j))))
            dim = j-1;
            break
        end
        % Stops calculation if the change in %FNN is below a threshold.
        if j>=2 && abs(dE(j)-dE(j-1))<=0.001
            dim = j;
            break
        end
    end
    
end

if speed==0
    for i = 2:length(dE)-1
        % Looks for 0 percentage or minimum.
        if (dE(i)<=0)||((dE(i-1)>dE(i)&&(dE(i)< dE(i+1))))
            dim= i;
            break
        end
        % Looks for a change in %FNN below a threshold.
        if dE(i-1)-dE(i)<=0.001
            dim = i-1;
            break
        end
        
    end
end

if isempty(dim)
    dim=MaxDim;
    fprintf('no dimension found, dim set to MaxDim\n')
end

end

%% kd_part function
function [y_model,z_model,sort_list,node_list] = kd_part(y_in, z_in, bin_size,~,~,~,~)

% Create a kd-tree and partitioned database for
% efficiently finding the nearest neighbors to a point
% in a d-dimensional space.
% Usage: [] = kd_part(y_in, z_in, bin_size);
%
% y_in: original phase space data
% z_in: original phase space data corresponding to y_in
% bin_size: maximum number of distinct points for each bin
% The outputs are placed into global variables used by
% kdsearch and its subroutines.

% The outputs are...
% sort_list(:,1): discriminator: dimension to use in dividing data
% sort_list(:,2): partition: boundary for dividing data
% node_list(i,:): contains data for the i-th partition
% node_list(:,1): 1st element in y of this partition
% node_list(:,2): last element in y of this partition
% node_list(:,3): location in node_list of left branch
% node_list(:,4): location in node_list of right branch
% y_model: phase space data partitioned into a binary tree
% z_model: phase space data corresponding to each y_model point

% Algorithms from:
%
% "Data Structures for Range Searching", J.L. Bently, J.H. Friedman,
% ACM Computing Surveys, Vol 11, No 4, p 397-409, December 1979
%
% "An Algorithm for Finding Best Matches in Logarithmic Expected Time",
% J.H. Friedman, J.L. Bentley, R.A. Finkel, ACM Transactions on
% Mathematical Software, Vol 3, No 3, p 209-226, September 1977.
%
% Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
%               - Formatted.
%
%%

% global y_model z_model sort_list node_list

y_model = y_in;
z_model = z_in;

% d: dimension of phase space
% n_y: number of points to put into partitioned database
[n_y,d] = size(y_model);

% Set up first node...
node_list = [1 n_y 0 0];
sort_list = [0 0];

% ...and the information about the number of nodes so far
node = 1;
last = 1;

while node <= last % check if the node can be divided
    
    segment = (node_list(node,1):node_list(node,2))';
    
    i=1:d;
    range = max(y_model(segment,i))'-min(y_model(segment,i))';
    
    if max(range) > 0 && length(segment)>= bin_size % it is divisible
        
        [~, index] = sort(range);
        yt = y_model(segment,:);
        zt = z_model(segment,:);
        [y_sort, y_index] = sort(yt(:,index(d)));
        
        % estimate where the cut should go
        [tlen,~] = size(yt);
        
        if rem(tlen,2) % yt has an odd number of elements
            cut = y_sort((tlen+1)/2);
        else % yt has an even number of elements
            cut = (y_sort(tlen/2)+y_sort(tlen/2+1))/2;
        end % of the median calculation
        
        L = y_sort <= cut;
        
        if sum(L) == tlen % then the right node will be empty...
            L = y_sort < cut;  % ...so use a slightly different boundary
            cut = (cut+max(y_sort(L)))/2;
        end % of the cut adjustment
        
        % adjust the order of the data
        y_model(segment,:) = yt(y_index,:);
        z_model(segment,:) = zt(y_index,:);
        
        % mark this as a non-terminal node
        sort_list(node,:) = [index(d) cut];
        node_list(node,3) = last + 1;
        node_list(node,4) = last + 2;
        last = last + 2;
        
        % add the information for the new nodes
        node_list = [node_list; segment(1) segment(1)+sum(L)-1 0 0];
        node_list = [node_list; segment(1)+sum(L) segment(tlen) 0 0];
        sort_list = [sort_list; 0 0; 0 0]; % assume they're terminal for the moment
        
    end % of the splitting process
    
    node = node + 1;
    
end % of the while loop
end

%% kdsearch function
function [pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list] = kdsearch(node,m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list)

% [] = kdsearch(node)
%   node - unknown
%
% Remarks
% - Search a kd_tree to find the nearest matches to the global variable
%   yq, a vector.  The nearest matches will be put in the global variable
%   pqr, and their distances in pqd.  See loclin_kd for a usage example.
%
% Future Work
% - This code could be commmented to be understood easier.
%
% Feb 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
%          - Commented and formated
%%

% global y_model z_model L_done pqr pqz b_upper b_lower sort_list node_list %m_search yq pqd

if L_done
    return
end

if node_list(node,3) == 0 % it's a terminal node, so...
    
    % first, compute the distances...
    yi = node_list(node,1:2); % index bounds of all y_model to consider
    yt = y_model(yi(1):yi(2),:);
    zt = z_model(yi(1):yi(2),:);
    
    d = length(yq); % get the dimension
    
    j=1:d;
    dist=sqrt(sum((yt(:,j)-yq(j)).^2,2));
    
    % and then sort them and load pqd, pqr, and pqz
    pqd = [dist;pqd]; % distances ^2
    pqr = [yt;pqr];  % current neares neighbors
    pqz = [zt;pqz]; % corresponding entries in z
    [pqd, index] = sort(pqd); % distances
    
    [len,~] = size(pqz);
    
    if length(index) > len
        pqr = pqr(index(1:length(pqz)),:);
        pqz = pqz(index(1:length(pqz)),:);
    else
        pqr = pqr(index,:);
        pqz = pqz(index,:);
    end
    
    % keep only the first m_search points
    if length(pqd) > m_search
        pqd = pqd(1:m_search);
    end
    
    [len,~] = size(pqz);
    
    if len > m_search
        pqr = pqr(1:m_search,:);
        pqz = pqz(1:m_search,:);
    end
    
    if any(abs(yq-b_lower)<=pqd(m_search) | abs(yq-b_upper)<=pqd(m_search))
        L_done = 1;
    end
    
    return
    
else % it's not a terminal node, so search a little deeper
    
    disc = sort_list(node,1);
    part = sort_list(node,2);
    
    if yq(disc) <= part % determine which child node to go to
        
        temp = b_upper(disc);
        b_upper(disc) = part;
        %         [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd)
        [pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list] = kdsearch(node_list(node,3),m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
        
        b_upper(disc) = temp;
        
    else
        
        temp = b_lower(disc);
        b_lower(disc) = part;
        %         [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd)
        [pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list] = kdsearch(node_list(node,4),m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
        
        b_lower(disc) = temp;
        
    end
    
    if L_done
        return
    end
    
    if yq(disc) <= part % determin whether other child node needs to be searched
        
        temp = b_lower(disc);
        b_lower(disc) = part;
        
        L = overlap(yq,m_search,pqd,b_upper,b_lower);
        if L
            %             [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd);
            [pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list] = kdsearch(node_list(node,4),m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
            
        end
        
        b_lower(disc) = temp;
        
    else
        
        temp = b_upper(disc);
        b_upper(disc) = part;
        
        L = overlap(yq,m_search,pqd,b_upper,b_lower);
        if L
            %             [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd);
            [pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list] = kdsearch(node_list(node,3),m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
            
        end
        
        b_upper(disc) = temp;
        
    end
    
    if L_done
        return
    end
    
end
%% End of kdsearch function
end

function L = overlap(yq,m_search,pqd,b_upper,b_lower)

% L = overlap
%   Inputs: none
%   Outputs: L - unknown
%
% Remarks
% - This code uses global variables. They must be defined as global here
%   and with the code this subroutine is called from. Because of their use
%   there are no inputs to this function.
% - References: - "Data Structures for Range Searching", J.L. Bently, J.H.
%                 Friedman, ACM Computing Surveys, Vol 11, No 4, p 397-409,
%                 December 1979.
%               - "An Algorithm for Finding Best Matches in Logarithmic
%                 Expected Time", J.H. Friedman, J.L. Bentley, R.A. Finkel,
%                 ACM Transactions on Mathematical Software, Vol 3, No 3,
%                 p 209-226, September 1977.
%
% Future Work
% - None.
%
% Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com

%%

% global yq m_search pqd b_upper b_lower sort_list node_list y_model z_model L_done pqr pqz

dist = pqd(m_search)^2;
sum = 0;

for i = 1:length(yq)
    
    if yq(i) < b_lower(i)
        sum = sum + (yq(i)-b_lower(i))^2;
        if sum > dist
            L = 0;
            return
        end % of the sum > dist if
    elseif yq(i) > b_upper(i)
        sum = sum + (yq(i)-b_upper(i))^2;
        if sum > dist
            L = 0;
            return
        end % of the sum > dist if
    end % of the yq(i) <> a bound if
    
end % of the i loop

L = 1;

%% End of overlap function
end
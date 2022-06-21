function [x,y] = embed(z,v,w)

% [x,y] or x= embed(z,lags) or embed(z,dim,lag)
% embed z using given lags or dim and lag
% embed(z,dim,lag) == embed(z,[0:lag:lag*(dim-1)])
% negative entries of lags are into future
%
% If return is [x,y], then x is the positive lags and y the negative lags
% Order of rows in x and y the same as sort(lags)
%
% defaults:
%  dim = 3
%  lag = 1
%  lags = [0 1 2]; or [-1 lags] when two outputs and no negative lags


% Copyright (c) 1994 by Kevin Judd.  
% Please see the copyright notice included in this distribution
% for full details.
%
% NAME embed.m
%   $Id$


if nargin==3
  v= 0:w:w*(v-1);
end;
if nargin==1
  v= [0 1 2];
end
if nargout==2 & min(v)>=0
  v= [-1 v];
end
lags= sort(v);

dim = length(lags);

[c,n] = size(z);
if c ~= 1
  z = z';
  [c,n] = size(z);
end
if c ~= 1
  error('Embed needs a vector as first arg.');
end

if n < lags(dim)
  error('Vector is too small to be embedded with the given lags');
end


w = lags(dim) - lags(1); 		% window
m = n - w; 				% Rows of x
t = (1:m)  + lags(dim); 		% embed times

x = zeros(dim,m);

for i=1:dim
  x(i,:) = z( t  -  lags(i) );
end

if nargout==2
  id= find(v<0);
  y= x(id,:);
  id= find(v>=0);
  x= x(id,:);
end;

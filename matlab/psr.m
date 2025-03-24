function xpsr = psr(x,tau,dim)
%   Phase space reconstruction
%   Input
%      x : Time series that needs phase space reconstruction
%      tau : Optimal time delay
%      dim : Optimal embedding dimension
%   Output
%      xpsr : M x dim*DIM matrix

% Get time series size
N = height(x);
DIM = width(x);

% Compute length of phase space reconstructed data
M = N-(dim-1)*tau;

% Time delay embedding
xpsr=zeros(M,dim*DIM);
for i=1:dim
    xpsr(1:M,(1:DIM)+DIM*(i-1)) = x((1+(i-1)*tau):(N-(dim-i)*tau),:);
end

end
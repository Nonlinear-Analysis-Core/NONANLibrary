function [xAP]=Ent_xAp(X,Y,M,r,k)

%[xAP]=Ent_xAp201603(X,Y,M,r,k)
%
% inputs:    X - first time series
%            Y - second time series
%            M - something vector length
%            r - R tolerance to find matches, proportion of the stdev
%            k - something lag
% outputs:   xAP - cross approximate entropy
%
% Remarks
% - This code finds the cross approximate entropy between two signals of
%   equal length.
%
% Future Work
% - This code should be looked over.
% - The scaling of the radius to the standard deviation may need to be
%   calculated from the average stdev of both signals and not just one.
% - The first for loop with m=M:k:M+k looks suspicious.
%
% Mar 2016 - Modified by Ben Senderling, email: bensenderling@gmail.com
%          - Moved the data normalization from the code that called this
%            one into this code.
%          - Changed the input r value from a percentage to a decimal for
%            consistency with other entropy code.
%
%% Begin Code

X=(X-mean(X))/std(X);
Y=(Y-mean(Y))/std(Y);

N=length(X);
Cm=[];
r=std(X)*r;
for m=M:k:M+k 
	C=[];
	for i=1:(N-m+1)
		V=[X(i:m+i-1)];
		count=0;
		for j=1:(N-m+1)
			Z=[Y(j:m+j-1)];
			dif=(abs(V-Z)<r);	%two subsequences are similar if the difference between any pair 
                 				%of corrisponding measurements in the pattern is less than r
			A=all(dif);
			count=count+A;
		end
	C=[C count/(N-m+1)];	%vector containing the 
                    		%Cim=(number of patterns similar to the one beginning at interval i)/total number  
                   		%of pattern with the same length M
                        %display(num2str(i))
	end
	Cm=[Cm sum(C)/(N-m+1)];%vector containing the means of the Cim for subsequences of length M and of length M+k%
end
xAP=log(Cm(1)/Cm(2));
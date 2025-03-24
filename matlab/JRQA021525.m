function [RP, RESULTS]=JRQA021525(data,tau,dim,param,threshold,options)
arguments
    data double {mustbeAtLeastTwoColumns}
    tau (1,1) {mustBeInteger, mustBePositive} = 1
    dim (1,1) {mustBeInteger, mustBePositive} = 1
    param (1,1) string {mustBeMember(param,["rad", "rec"])} = "rec"
    threshold (1,1) double {mustBePositive} = 2.5
    options.Zscore (1,1) {mustBeMember(options.Zscore,[0,1])} = 1
    options.Norm (1,1) {mustBeMember(options.Norm,["euc", "max", "min", "none"])} = "none"
    options.Dmin (1,1) {mustBeInteger, mustBePositive} = 2
    options.Vmin (1,1) {mustBeInteger, mustBePositive} = 2
    options.Plot (1,1) {mustBeMember(options.Plot,[0,1])} = 0
    options.Orient (1,1) {mustBeMember(options.Orient,["col", "row"])} = "col"
    options.Iter (1,1) {mustBeInteger, mustBePositive} = 20
end

%% Begin code
dbstop if error % If error occurs, enters debug mode

%% Change variable names for readability
dmin = options.Dmin;
vmin = options.Vmin;

%% Standardize data if zscore is true
% If zscore is selected then zscore the data
if options.Zscore
    data = zscore(data);
end

% Get number of time series
DIM = size(data, 2);

% Embed the data onto phase space
if dim > 1
    data = psr(data, tau, dim);
end

% Calculate distance matrix based on the type of RQA
for i = 1:DIM
    a{i}=pdist2(data(:,i:DIM:end),data(:,i:DIM:end));
    a{i}=abs(a{i})*-1;
end

% Normalize distance matrix
if contains(options.Norm, 'euc')
    for i = 1:length(a)
    b = mean(a{i}(a{i}<0));
    b = -sqrt(abs(((b^2)+2*(DIM*dim))));
    a{i} = a{i}/abs(b);
    end
elseif contains(options.Norm, 'min')
    for i = 1:length(a)
    b = max(a{i}(a{i}<0));
    a{i} = a{i}/abs(b);
    end
elseif contains(options.Norm, 'max')
    for i = 1:length(a)
    b = min(a{i}(a{i}<0));
    a{i} = a{i}/abs(b);
    end
end

% Compute weighted recurrence plot
wrp = a;
for i = 1:size(a,2)-1
    wrp{i+1} = wrp{i}.*wrp{i+1};
end
if i
    wrp = -(abs(wrp{i+1})).^(1/(i+1));
end
if iscell(wrp)
    wrp = wrp{1};
end

% Calculate recurrence plot
switch param
    case 'rad'
        [recurrence, diag_hist, vertical_hist,A] = linehist(data,a,threshold,'jrqa');
    case 'rec'
        radius_start = 0.01;
        radius_end = 0.5;
        [recurrence, diag_hist, vertical_hist, radius, A] = setradius(data,a,radius_start,radius_end,threshold,'jrqa',options.Iter);
end

%% Calculate RQA variabes
RESULTS.DIM = 1;
RESULTS.EMB = dim;
RESULTS.DEL = tau;
RESULTS.RADIUS = radius;
RESULTS.NORM = options.Norm;
RESULTS.ZSCORE = options.Zscore;
RESULTS.Size=length(A);
RESULTS.REC = recurrence;
if RESULTS.REC > 0
    RESULTS.DET=100*sum(diag_hist(diag_hist>=dmin))/sum(diag_hist);
    RESULTS.MeanL=mean(diag_hist(diag_hist>=dmin));
    RESULTS.MaxL=max(diag_hist(diag_hist>=dmin));
    [count,bin]=hist(diag_hist(diag_hist>=dmin),min(diag_hist(diag_hist>=dmin)):max(diag_hist(diag_hist>=dmin)));
    total=sum(count);
    p=count./total;
    del=find(count==0); p(del)=[];
    RESULTS.EntrL=-sum(p.*log2(p));
    RESULTS.LAM=100*sum(vertical_hist(vertical_hist>=vmin))/sum(vertical_hist);
    RESULTS.MeanV=mean(vertical_hist(vertical_hist>=vmin));
    RESULTS.MaxV=max(vertical_hist(vertical_hist>=vmin));
    [count,bin]=hist(vertical_hist(vertical_hist>=vmin),min(vertical_hist(vertical_hist>=vmin)):max(vertical_hist(vertical_hist>=vmin)));
    total=sum(count);
    p=count./total;
    del=find(count==0); p(del)=[];
    RESULTS.EntrV=-sum(p.*log2(p));
    RESULTS.EntrW=Ent_Weighted(wrp);
else
    RESULTS.DET=NaN;
    RESULTS.MeanL=NaN;
    RESULTS.MaxL=NaN;
    RESULTS.EntrL=NaN;
    RESULTS.LAM=NaN;
    RESULTS.MeanV=NaN;
    RESULTS.MaxV=NaN;
    RESULTS.EntrV=NaN;
    RESULTS.EntrW=NaN;
end
RP=imrotate(1-A,90);

%% Plot
if options.Plot
    RQA_plot(data, RP, RESULTS, tau, dim, DIM, options.Zscore, options.Norm, radius, wrp, 'jrqa');
end

end

% Custom validation function
function mustbeAtLeastTwoColumns(data)
% Test for size
if size(data,2) < 2
    error('Data must be at least two column vectors.')
end
end
function CoD=corrdim(x,tau,de,plotOption)
%Correlation Dimension
%Scaling region mid-one quarter of vertical axis -- mid + OneQuarter
%2/5/2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%x: a time series
%tau: time delay
%de: embedding diemnsion
%plotOption: set plotOption=1 to see plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bins=200;
n = length(x)-(de-1)*tau; % total number of reconstructed vectors
%Use embedding to calculate the distances between vectors
y = embed(x,de,tau);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the interval (epsilon2 < epsilon < epsilon1), where epsilon1 is the
%order of the size of the attractor in phase space while epsilon2 is the
%smallest spacing |yi - yj|.  If epsilon2 = 0, then esp is assigned to
%epsilon2 in order to avoid log(0) = -Inf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize epsilon1 and epsilon2
epsilon1 = 0;
epsilon2 = Inf;

k=de*tau;  % removing temporal correlations
for i = 1:n-k-1
    distance = sqrt(sum((y(:,i+k+1:n)-y(:,i)*ones(size(i+k+1:n))).^2));
    epsilon1 = max(max(distance),epsilon1);
    epsilon2 = min(min(distance), epsilon2);
end

if epsilon2==0   % in order to take log of epsilon2
    epsilon2=eps;
end

epsilon =linspace(log(epsilon2),log(epsilon1),bins);
%using natural log
epsilon=exp(1).^epsilon;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CORRELATION SUM
%   C(epsilon) = sum (H(epsilon - |yi-yj|)/N^2 as n --> infinity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CI=[];
for i = 1:n-k-1
    distance = sqrt(sum((y(:,i+k+1:n)-y(:,i)*ones(size(i+k+1:n))).^2));  %using Eucledian distance
    Sum_HeaviSide = histc(distance,epsilon);   %Heaviside function H(z)itself returns 1 for positive z and 0 otherwise.
    %Store the value into CI matrix
    if i==1
        CI=[CI; Sum_HeaviSide];
    else
        CI=sum([CI; Sum_HeaviSide]);
    end
end

%taking natural log of correlation integral
CI = log(cumsum(CI)/((n-k)^2));
CI = CI + log(n-k-1); % renormalizes to natural log of average pts in a neigborhood
epsilon=log(epsilon);

if isinf(min(CI))==1
    i=max(find(CI==-Inf));
    minCI=min(CI(i+1:end));
else
    minCI=min(CI);
end

if isinf(max(CI))==1
    i=min(find(CI==Inf));
    maxCI=max(CI(1:end-i-1));
else
    maxCI=max(CI);
end

%Find scaling region
midCIValue=((max(CI)-minCI)/2)+ minCI; % find the mid value of CI
OneQuarter=(max(CI)-minCI)/4;  %calculate 1/4 of CI
y_LowerBound = midCIValue; % let midCIValue be the lower bound on y axis (scaling region)
y_UpperBound=midCIValue+OneQuarter; %upper bound on y axis (scaling region)
intervals=intersect(find(CI > y_LowerBound),find(CI<y_UpperBound));
x_LowerBound=min(intervals);
x_UpperBound=max(intervals);
MidOneQuarter=x_LowerBound:x_UpperBound;

%find the slope
CoD = polyfit(epsilon(MidOneQuarter), CI(MidOneQuarter), 1);
slope1 = num2str(CoD(1));
fittedline = polyval(CoD,epsilon(MidOneQuarter));

%plots
if plotOption==1
    subplot(2,1,1)
    fsize=14;
    plot(epsilon,CI,'.','MarkerSize',6)
    hold on
    plot(epsilon(MidOneQuarter),fittedline,'r')
    ylim([minCI-1 maxCI+1])
    xlabel('ln(\epsilon)','FontWeight','bold','FontSize',fsize)
    ylabel('ln(C(\epsilon))','FontWeight','bold','FontSize',fsize)
    legend('ln{10}(\epsilon) vs ln(C(\epsilon))', ['CoD = ', slope1 ], 'Location', 'SouthEast');
    axis tight

    %Corresponding to View 2 (CDA)
    subplot(2,1,2)
    FindScalingRegion=diff(CI)./diff(epsilon);
    plot(epsilon(1:end-1),FindScalingRegion,'LineWidth',2)
    hold on
    plot(epsilon(MidOneQuarter),FindScalingRegion(MidOneQuarter),'r')
    xlabel('ln(\epsilon)','FontWeight','bold','FontSize',fsize)
    ylabel('\Delta ln(C(\epsilon)) /\Delta ln(\epsilon)','FontWeight','bold','FontSize',fsize)
    %axis tight
    hold off
    display(['Correlation Dimension = ', num2str(CoD(1))])  % Correlation Dimension
end

CoD=CoD(1);


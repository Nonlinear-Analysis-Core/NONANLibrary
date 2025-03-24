function [w_ent] = Ent_Weighted(wrp)
N = length(wrp); % get size of the weighted recurrence plot
for j = 1:N
    si(j) = sum(wrp(:,j)); % compute vertical weights sums
end

% Compute distribution of vertical weights sums
si_min = min(si);
si_max = max(si);
bin_size = (si_max - si_min)/49; % compute bin size
count = 1;
S = sum(si);
for s = si_min:bin_size:si_max
    P = sum(si(si>= s&si<(s+bin_size)));
    p1(count) = P / S;
    count = count+1;
end

% Compute weighted entropy
for I = 1:length(p1)
    pp(I) = (p1(I)*log(p1(I)));
end
pp(isnan(pp)) = 0;
w_ent = -1*(sum(pp));

end
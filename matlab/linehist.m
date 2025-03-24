function [recurrence, diag_hist, vertical_hist, a] = linehist(data, a, radius, type)

% Check if a is cell
if ~iscell(a)
    a = {a};
end

% Convert distance matrices to recurrence matrices
for i2 = 1:length(a)
    a{i2} = a{i2}+radius;
    a{i2}(a{i2} >= 0) = 1;
    a{i2}(a{i2} < 0) = 0;
end

% If a contains multiple recurrence matrices, compute dot product of all
% matrices...?
if length(a) > 1
    for i3 = 1:length(a)-1
        a{i3+1} = a{i3}.*a{i3+1};
    end
    a = a{i3+1};
else
    a = a{1};
end

% Caluculate diagonal line distribution
diag_hist = [];
vertical_hist = [];
for i4 = -(length(data)-1):length(data)-1
    c=diag(a,i4);
    % bwlabel is taking each diagonal line and looking for the 1's, it will
    % return increasing numbers for each new instance of 1's, for example
    % the input vector [0 1 1 0 1 0 1 1 0 0 1 1 1] will return
    %                  [0 1 1 0 2 0 3 3 0 0 4 4 4]
    d=bwlabel(c,8);
    % tabulate counts the instances of each integer, therefore the line
    % lengths
    if sum(d) ~= 0
        d = nonzeros(hist(d)); % This speeds up the code 30-40% and is simpler to understand.
    else
        d = [];
    end
    if i4 ~= 0
        d=d(2:end);
    end
    % diag_hist is creating one long array of all of the line lengths for
    % all of the diagonals
    diag_hist(length(diag_hist)+1:length(diag_hist)+length(d))=d;
end

% Remove the line of identity in RQA, jRQA, and mdRQA
if ~contains(type,'crqa')
    diag_hist=diag_hist(diag_hist<max(diag_hist));
    if isempty(diag_hist)
        diag_hist=0;
    end
end

% Calculate vertical line distribution
for i5=1:length(data)
    c=(a(:,i5));
    v=bwlabel(c,8);
    if sum(v) ~= 0
        v = nonzeros(hist(v)); % This speeds up the code 30-40% and is simpler to understand.
    else
        v = [];
    end
    if ~isempty(v)
        if v(1,1)~=length(data)
            v=v(2:end);
        end
    end
    vertical_hist(length(vertical_hist)+1:length(vertical_hist)+length(v))=v;
end

% Calculate percent recurrence
if ~contains(type,'crqa')
    recurrence = 100*(sum(sum(a))-length(a))/(length(a)^2-length(a));
else
    recurrence = 100*(sum(sum(a)))/(length(a)^2);
end

end
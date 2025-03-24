function [rec, diag_hist, vertical_hist, rad_final,A] = setradius(data,a,radius_start,radius_end,threshold,type,iter)
        % Find the radius to provide target percent recurrence
        % If radius_start is too small
        [rec, ~, ~, ~] = linehist(data,a,radius_start,type);
        while rec == 0 || rec > threshold
            disp('Minimum radius has been adjusted...');
            if rec == 0
                radius_start = radius_start*2;
            elseif rec > threshold
                radius_start = radius_start / 1.5;
            end
            [rec, ~, ~, ~] = linehist(data,a,radius_start,type);
        end

        % if radius_end is too large
        [rec, ~, ~, ~] = linehist(data,a,radius_end,type);
        while rec < threshold
            disp('Maximum radius has been increased...');
            radius_end = radius_end*2;
            [rec, ~, ~, ~] = linehist(data,a,radius_end,type);
        end

        % Search for radius with target percent recurrence
        % Create wait bar to display progress
        wb = waitbar(0,['Finding radius to give %REC = ',num2str(threshold), ' Please wait...']);    
        lv = radius_start; % set low value
        hv = radius_end; % set high value
        target = threshold;  % designate what percent recurrence is wanted
        for  i1 = 1:iter
            mid(i1) = (lv(i1)+hv(i1))/2; % find midpoint between hv and lv
            rad(i1) = mid(i1); % new radius for this iteration

            % Compute recurrence matrix with new radius
            [rec, diag_hist, vertical_hist,A] = linehist(data,a, rad(i1),type);
            rec_iter(i1) = rec;  % set percent recurrence
            if rec_iter(i1) < target
                % if percent recurrence is below target percent recurrence,
                % update low value
                hv(i1+1) = hv(i1);
                lv(i1+1) = mid(i1);
            else
                % if percent recurrence is above or equal to target percent
                % recurrence, update high value
                lv(i1+1) = lv(i1);
                hv(i1+1) = mid(i1);
            end
            waitbar(i1/iter,wb); % update wait bar
        end
        close(wb) % close wait bar
        rec_final = rec_iter(end); % set final percent recurrence
        rad_final = rad(end); % set radius for final percent recurrence
        disp(['% recurrence = ',num2str(rec_final),', radius = ',num2str((rad_final))])
    end
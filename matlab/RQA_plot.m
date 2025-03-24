function RQA_plot(data, rp, RESULTS, tau, dim, DIM, Zscore, Norm, radius, wrp, type)

scrsz = get(0,'ScreenSize');
f = figure('Position',[scrsz(3)/4 scrsz(4)/4 scrsz(3)/3 scrsz(4)/2]);tabgp = uitabgroup(f);
binary = uitab(tabgp,'Title','Binary');
heatmap = uitab(tabgp,'Title','Heatmap');
% Binary Plot (Tab 1)
a1 = axes('Parent',binary,'Position', [0 0 1 1], 'Visible', 'off');
ax(1) = axes('Parent',binary,'Position',[.375 .35 .58 .6], 'FontSize', 8);
imagesc(ax(1),rp); colormap(gray);
title(['DIM = ', num2str(DIM), '; EMB = ',num2str(dim), '; DEL = ', num2str(tau), '; RAD = ', num2str(radius), '; NORM = ',num2str(Norm), '; ZSCORE = ',num2str(Zscore)],'FontSize',8)
xlabel('X(i)','Interpreter','none', 'FontSize', 10);
ylabel('Y(j)','Interpreter','none', 'FontSize', 10);
set(gca,'XTick',[ ]);
set(gca,'YTick',[ ]);

switch type
    case {'rqa','mdrqa'}
        ax(2) = axes('Parent',binary,'Position',[.375 .1 .58 .15], 'FontSize', 8);
        plot(1:length(data(:,1)), data(:,1), 'k-');
        xlim([1 length(data(:,1))]);
        ax(3) = axes('Parent',binary,'Position',[.09 .35 .15 .6], 'FontSize', 8);
        plot(flip(data(:,1)), 1:length(data(:,1)), 'k-');
        ylim([1 length(data(:,1))]);
        set (ax(3),'Ydir','reverse');
    case 'crqa'
        ax(2) = axes('Parent',binary,'Position',[.375 .1 .58 .15], 'FontSize', 8);
        plot(1:length(data(:,1)), data(:,1), 'k-');
        xlim([1 length(data(:,1))]);
        ax(3) = axes('Parent',binary,'Position',[.09 .35 .15 .6], 'FontSize', 8);
        plot(flip(data(:,2)), 1:length(data(:,2)), 'k-');
        ylim([1 length(data(:,2))]);
        set (ax(3),'Ydir','reverse');
    case 'jrqa'
        for i = 1:DIM
            ax(2) = axes('Parent',binary,'Position',[.375 .1 .58 .15], 'FontSize', 8);
            plot(1:length(data(:,1)), data(:,i),'k-');
            xlim([1 length(data(:,1))]);
            ax(3) = axes('Parent',binary,'Position',[.09 .35 .15 .6], 'FontSize', 8);
            plot(flip(data(:,1)), 1:length(data(:,i)),'k-');
            ylim([1 length(data(:,1))]);
            set (ax(3),'Ydir','reverse');
        end
end

set(gcf, 'CurrentAxes', a1);
str(1) = {['%REC = ', sprintf('%.2f',RESULTS.REC)]};
text(.1, 0.27, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['%DET = ', sprintf('%.2f',RESULTS.DET)]};
text(.1, .24, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['MaxL = ', sprintf('%.0f',RESULTS.MaxL)]};
text(.1, .21, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['MeanL = ', sprintf('%.2f',RESULTS.MeanL)]};
text(.1, .18, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['EntrL = ', sprintf('%.2f',RESULTS.EntrL)]};
text(.1, .15, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['%LAM = ', sprintf('%.2f',RESULTS.LAM)]};
text(.1, .12, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['MaxV = ', sprintf('%.0f',RESULTS.MaxV)]};
text(.1, .09, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['MeanV = ', sprintf('%.2f',RESULTS.MeanV)]};
text(.1, .06, str, 'FontSize', 8, 'Color', 'k');
str(1) = {['EntrV = ', sprintf('%.2f',RESULTS.EntrV)]};
text(.1, .03, str, 'FontSize', 8, 'Color', 'k');

% Heatmap Plot (Tab 2)
a2 = axes('Parent',heatmap,'Position', [0 0 1 1], 'Visible', 'off');
ax(4) = axes('Parent',heatmap,'Position',[.375 .35 .58 .6], 'FontSize', 8);
imagesc(ax(4),imrotate(-1*wrp,90));
title(['DIM = ', num2str(DIM), '; EMB = ',num2str(dim), '; DEL = ', num2str(tau), '; RAD = ', num2str(radius), '; NORM = ',num2str(Norm), '; ZSCORE = ',num2str(Zscore)],'FontSize',8)
xlabel('X(i)','Interpreter','none', 'FontSize', 10);
ylabel('Y(j)','Interpreter','none', 'FontSize', 10);
set(gca,'XTick',[ ]);
set(gca,'YTick',[ ]);

switch type
    case {'rqa','mdrqa'}
        ax(5) = axes('Parent',heatmap,'Position',[.375 .1 .58 .15], 'FontSize', 8);
        plot(1:length(data(:,1)), data(:,1), 'k-');
        xlim([1 length(data(:,1))]);
        ax(6) = axes('Parent',heatmap,'Position',[.09 .35 .15 .6], 'FontSize', 8);
        plot(flip(data(:,1)), 1:length(data(:,1)), 'k-');
        ylim([1 length(data(:,1))]);
        set (ax(6),'Ydir','reverse');
    case 'crqa'
        ax(5) = axes('Parent',heatmap,'Position',[.375 .1 .58 .15], 'FontSize', 8);
        plot(1:length(data(:,1)), data(:,1), 'k-');
        xlim([1 length(data(:,1))]);
        ax(6) = axes('Parent',heatmap,'Position',[.09 .35 .15 .6], 'FontSize', 8);
        plot(flip(data(:,2)), 1:length(data(:,2)), 'k-');
        ylim([1 length(data(:,2))]);
        set (ax(6),'Ydir','reverse');
    case 'jrqa'
        for i = 1:DIM
            ax(5) = axes('Parent',binary,'Position',[.375 .1 .58 .15], 'FontSize', 8);
            plot(1:length(data(:,1)), data(:,i),'k-');
            xlim([1 length(data(:,1))]);
            ax(6) = axes('Parent',binary,'Position',[.09 .35 .15 .6], 'FontSize', 8);
            plot(flip(data(:,1)), 1:length(data(:,i)),'k-');
            ylim([1 length(data(:,1))]);
            set (ax(6),'Ydir','reverse');
        end
end

set(gcf, 'CurrentAxes', a2);
str(1) = {['EntrW = ', sprintf('%.2f',RESULTS.EntrW)]};
text(.1, 0.27, str, 'FontSize', 8, 'Color', 'k');
linkaxes(ax([1,4]),'xy');
linkaxes(ax([1,2,4,5]),'x');
linkaxes(ax([1,3,4,6]),'y');

end
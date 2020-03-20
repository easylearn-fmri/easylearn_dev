function lc_plot_average_diff_and_weights()
%% Plot average tvalues within or between networks
% Inputs
tvalues_medicated = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalues_medicated.mat');
tvalues_feu = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalues_feu.mat');
cohen_medicated = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_medicated1.mat');
cohen_feu = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_feu1.mat');
load D:\WorkStation_2018\SZ_classification\Data\Stat_results\weights.mat;
load  D:\My_Codes\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\netIndex.mat;
mycolormap = 'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\cmp_average_diff.mat';
mycolormap = 'D:\My_Codes\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\mycmap_average_diff.mat';

% Get average
cohen_medicated(tvalues_medicated == 0) = 0;
cohen_feu(tvalues_feu == 0) = 0;
average_cohen_medicated = get_average_diff(cohen_medicated, 0, [], netidx);
average_cohen_unmedicated = get_average_diff(cohen_feu, 0, [], netidx);
acm = average_cohen_medicated + average_cohen_medicated';
acu = average_cohen_unmedicated + average_cohen_unmedicated';
save('D:\WorkStation_2018\SZ_classification\Data\Stat_results\average_cohen_medicated.mat','acm');
save('D:\WorkStation_2018\SZ_classification\Data\Stat_results\average_cohen_unmedicated.mat','acu');

average_tvalues_medicated = get_average_diff(tvalues_medicated, 0, [], netidx);
average_tvalues_unmedicated = get_average_diff(tvalues_feu, 0, [], netidx);

% Plot average cohen
figure;
title('Cohen');
subplot(1, 2, 1);
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
mycmp = importdata(mycolormap);
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
matrixplot(average_cohen_medicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
% title('Average differences of pooled datasets', 'fontsize', 6);
colormap(mycmp);
caxis([-0.5,0.5])
colorbar

subplot(1, 2, 2)
matrixplot(average_cohen_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
% title('Average differences of unmedicated sub-dataset', 'fontsize', 6);
colormap(mycmp);
caxis([-0.5,0.5])
colorbar
saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\average_cohen.pdf')
% 
% % Plot average tvalues
% figure;
% title('T');
% subplot(1, 2, 1)
% matrixplot(average_tvalues_medicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
% % title('Average differences of pooled datasets', 'fontsize', 6);
% colormap(mycmp);
% caxis([-10,10])
% colorbar
% 
% subplot(1, 2, 2)
% matrixplot(average_tvalues_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
% % title('Average differences of unmedicated sub-dataset', 'fontsize', 6);
% colormap(mycmp);
% caxis([-10,10])
% colorbar
% saveas(gcf, 'D:\WorkStation_2018\SZ_classification\Figure\average_tvales.pdf');
end

function meanFC = get_average_diff(diff, is_mask, mask, netidx)
% Differences filter
if is_mask
    diff(mask == 0) = 0;
end
diff_full = diff + diff';
%
uniid = unique(netidx);
n_uniid = numel(uniid);
unjid = uniid;
n_unjid = numel(unjid);
meanFC = zeros(numel(uniid));
for i = 1 : n_uniid
    id = find(netidx == uniid(i));
    for j =  1: n_unjid
        fc = diff_full(id, find(netidx == unjid(j)));
        % if within fc, extract upper triangle matrix
        if (all(diag(fc)) == 0) && (size(fc, 1) == size(fc, 2))
            fc = fc(triu(ones(length(fc)),1) == 1);
        end
        % Exclude zeros in fc
        % TODO: Consider the source data  have zeros.
        fc(fc == 0) = [];
        % Mean
        meanFC(i,j) = mean(fc(:));
    end
end
% Post-Process meanFC
meanFC(isnan(meanFC)) = 0;
meanFC(triu(ones(size(meanFC)), 1) == 1)=0;
end
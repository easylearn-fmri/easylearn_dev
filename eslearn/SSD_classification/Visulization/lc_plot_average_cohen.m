function lc_plot_average_cohen()
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
mycmp = importdata(mycolormap);

% Get average
cohen_medicated(tvalues_medicated == 0) = 0;
cohen_feu(tvalues_feu == 0) = 0;
average_cohen_medicated = get_average_diff(cohen_medicated, 0, [], netidx);
average_cohen_unmedicated = get_average_diff(cohen_feu, 0, [], netidx);

tvalues_medicated(tvalues_medicated == 0) = 0;
tvalues_feu(tvalues_feu == 0) = 0;
average_tvalues_medicated = get_average_diff(tvalues_medicated, 0, [], netidx);
average_tvalues_unmedicated = get_average_diff(tvalues_feu, 0, [], netidx);

% Plot average cohen
figure;
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
ystr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};

subplot(2, 2, 1);
matrixplot(average_tvalues_medicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmp);
caxis([-5,5])
colorbar

subplot(2, 2, 2)
matrixplot(average_tvalues_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmp);
caxis([-5,5])
colorbar

subplot(2, 2, 3);
matrixplot(average_cohen_medicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmp);
caxis([-0.8,0.8])
colorbar

subplot(2, 2, 4)
matrixplot(average_cohen_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmp);
caxis([-0.8,0.8])
colorbar

saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\average_tvalues_cohen.pdf')
end

function meanFC = get_average_diff(diff, is_mask, mask, netidx)
% Differences filter
if is_mask
    diff(mask == 0) = 0;
end
uniid = unique(netidx);
n_uniid = numel(uniid);
unjid = uniid;
n_unjid = numel(unjid);
meanFC = zeros(numel(uniid));
for i = 1 : n_uniid
    id = find(netidx == uniid(i));
    for j =  1: n_unjid
        fc = diff(id, find(netidx == unjid(j)));
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
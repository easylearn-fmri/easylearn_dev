% This script is used to check the distribution of the fc across the four dataset.
% Inputs
dataset1_file = 'D:\WorkStation_2018\SZ_classification\Data\matrix_550.mat';
dataset2_file = 'D:\WorkStation_2018\SZ_classification\Data\matrix_206.mat';
dataset3_file = 'D:\WorkStation_2018\SZ_classification\Data\matrix_COBRE.mat';
dataset4_file = 'D:\WorkStation_2018\SZ_classification\Data\matrix_UCAL.mat';
net_index_path = 'D:\My_Codes\lc_rsfmri_tools_python\Workstation\SZ_classification\ML\netIndex.mat';
mycmap_distribution = 'D:\My_Codes\easylearn\Workstation\SZ_classification\Visulization\mycmap_distribution.mat';
legends = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
legned_fontsize = 7;
thred_cohen = 0.5;
if_add_mask = 0;
how_disp = 'all';
if_binary = 0;
which_group = 1;

% Load
dataset1 = importdata(dataset1_file);
dataset2 = importdata(dataset2_file);
dataset3 = importdata(dataset3_file);
dataset4 = importdata(dataset4_file);
mycmap_distribution = importdata(mycmap_distribution);
% Plot
subplot(1,4,1)
lc_netplot(dataset1, if_add_mask, [], how_disp, 0, which_group, net_index_path, 1, legends, legned_fontsize);
caxis([-0.5 1]);
title('Dataset 1');
axis square
subplot(1,4,2)

lc_netplot(dataset2, if_add_mask, [], how_disp, 0, which_group, net_index_path, 1, legends, legned_fontsize);
caxis([-0.5 1]);
title('Dataset 2');
axis square

subplot(1,4,3)
lc_netplot(dataset3, if_add_mask, [], how_disp, 0, which_group, net_index_path, 1, legends, legned_fontsize);
title('Dataset 3');
caxis([-0.5 1]);
axis square

subplot(1,4,4)
lc_netplot(dataset4, if_add_mask, [], how_disp, 0, which_group, net_index_path, 1, legends, legned_fontsize);
title('Dataset 4')
caxis([-0.5 1]);
axis square

colormap(mycmap_distribution);
% print('D:\WorkStation_2018\SZ_classification\Figure\distribution.tif', '-dtiff','-r1200' )
saveas(gcf, 'D:\WorkStation_2018\SZ_classification\Figure\distribution_cb.pdf');


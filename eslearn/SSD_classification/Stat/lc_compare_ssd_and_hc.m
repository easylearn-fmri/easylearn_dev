function lc_compare_ssd_and_hc()
% This function is used to compare the medicated SSD with HC, as well as first episode unmedicated SSD with HC.
% Statistical method is NBS.
% Refer and thanks to NBS (NBSglm and NBSstats)
%% Inputs
if nargin < 1
    cov_medicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cov_medicated.mat';
    fc_medicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\fc_medicated.mat';
    cov_unmedicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cov_unmedicated.mat';
    fc_unmedicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\fc_unmedicated.mat';
    is_save = 1;
    test_type = 'ttest';
end

%% Prepare
load(cov_medicated);
load(fc_medicated); 
load(cov_unmedicated);
load(fc_unmedicated);

% Medicated
group_label = cov_medicated(:,3);
uni_groups = unique(group_label);
n_groups = length(uni_groups);
group_design = zeros(size(group_label,1), 2);
for i =  1:n_groups
    group_design(:,i) = ismember(group_label, uni_groups(i));
end
design_medicated = [cov_medicated(:,[2,4,5]),group_design];
contrast_medicated = [0 0 0 -1 1];

% Unmedicated
group_label = cov_unmedicated(:,2);
uni_groups = unique(group_label);
n_groups = length(uni_groups);
group_design = zeros(size(group_label,1), 2);
for i =  1:n_groups
    group_design(:,i) = ismember(group_label, uni_groups(i));
end
design_unmedicated = [cov_unmedicated(:,[3,4]),group_design];
contrast_unmedicated = [0 0 -1 1];

%% GLM
STATS.thresh = 3;
STATS.alpha = 0.005; % Equal to two-tailed 0.01.
STATS.N = 246;
STATS.size = 'extent';
GLM.perms = 1000;

% Medicated 
GLM.X = design_medicated;
GLM.y = fc_medicated;
GLM.contrast = contrast_medicated;
GLM.test = test_type; 
[test_stat_medicated, pvalues_medicated]=NBSglm(GLM);

STATS.test_stat = abs(test_stat_medicated);  % two-tailed
[n_cnt_medicated, cont_medicated, pval] = NBSstats(STATS);


% Unmedicated >
GLM.X = design_unmedicated;
GLM.y = fc_unmedicated;
GLM.contrast = contrast_unmedicated;
GLM.test = test_type; 
[test_stat_unmedicated,pvalues_unmedicated]=NBSglm(GLM);
STATS.test_stat = abs(test_stat_unmedicated);  % two-tailed
[n_cnt_unmedicated, cont_unmedicated, pval]=NBSstats(STATS);

%% to original space (2d matrix)
n_row = 246;
n_col = 246;
mask_full = tril(ones(n_row, n_col),-1) == 1;

mask_sig_medicated = full(cont_medicated{1}) == 1;
tvalues_medicated = zeros(n_row,n_col);
tvalues_medicated(mask_full) = test_stat_medicated(1,:);
tvalues_medicated(~mask_sig_medicated) = 0;
tvalues_medicated = tvalues_medicated+tvalues_medicated';

mask_sig_unmedicated = full(cont_unmedicated{1}) == 1;
tvalues_feu = zeros(n_row,n_col);
tvalues_feu(mask_full) = test_stat_unmedicated(1,:);
tvalues_feu(~mask_sig_unmedicated) = 0;
tvalues_feu = tvalues_feu+tvalues_feu';

%% save
if is_save
    save('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalues_medicated.mat', 'tvalues_medicated');
    save('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalues_feu.mat', 'tvalues_feu');
end
fprintf('--------------------------All Done!--------------------------\n');
end
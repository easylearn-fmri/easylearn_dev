function el_cluster_HYDRATE(explained_cov,min_clustering_solutions, max_clustering_solutions, cvfold, output_dir)
% cluster_HYDRATE: using Heterogeneity through discriminative analysis (HYDRA) to cluster sample to subgroups.
% The follows are copy from HYDRA.
% --------------------------------------------------------------------------
% This software performs clustering of heterogenous disease patterns within patient group.
% The clustering is based on seperating the patient imaging features from the control imaging features using a convex polytope classifier.
% Covariate correction can be performed optionally.
% Reference: Varol, Erdem, Aristeidis Sotiras, Christos Davatzikos.
% "HYDRA: Revealing heterogeneity of imaging and genetic patterns
% through a multiple max-margin discriminative analysis framework."
% NeuroImage 145 (2017): 346-364.
% --------------------------------------------------------------------------
% Because the dimensions of the image data is very high, so I use PCA to reduce its dimensions.
% github: https://github.com/lichao312214129/HYDRA
% @author: Li Chao
% Email: lichao19870617@gmail.com
% --------------------------------------------------------------------------
% INPUT:
%   input_data_csv: file string (refer to HYDRA documentation).
%   explained_cov: float, When using pca to reduce the dimensions of features, specify how many cumulative explained covariances to retain
%   output_dir: directory string, which directory to save results.
%   cvfold: how many folds to perform cross validation.
%   [min_clustering_solutions, max_clustering_solutions]: integer, range of clustering solutions.
%
% OUTPUT:
%   CIDX: sub-clustering assignments of the disease population (positive
%         class).
%   ARI: adjusted rand index measuring the overlap/reproducibility of
%        clustering solutions across folds
%% ----------------------------------------------------------------------------
% DEBUG
if nargin < 1
    explained_cov = 0.95;  % 主成分保留多少（累积方差解释）
    min_clustering_solutions = 1;  % 最小的类别数 （亚组）
    max_clustering_solutions = 5;  % 最大的类别数（亚组）
    cvfold = 3; % 交叉验证参数
    output_dir = 'D:\workstation_b\xiaowei\To_chao';
    patient_dir = 'D:\workstation_b\xiaowei\To_chao\MDD';
    hc_dir = 'D:\workstation_b\xiaowei\To_chao\HC';
    mask_file = 'D:\workstation_b\xiaowei\To_chao\HC\smReHoMap_010.nii';  % 改为灰质模板
end

% Output directory
% output_dir = uigetdir(pwd,'Select folder to save results');

% Image data directory
% patient_dir = uigetdir(pwd,'Select patient folder');
% hc_dir = uigetdir(pwd,'Select HC folder');
% [mask_name, mask_path] = uigetfile({'*.nii;*.img;','All Image Files';'*.*','All Files'},'MultiSelect','off','Pick a mask');

% Get image files path
patient_struct = dir(patient_dir);
patient_name = {patient_struct.name}';
patient_name = patient_name(3:end);
patient_path = fullfile(patient_dir, patient_name);

hc_struct = dir(hc_dir);
hc_name = {hc_struct.name}';
hc_name = hc_name(3:end);
hc_path = fullfile(hc_dir, hc_name);

% Load all images
num_patient = length(patient_path);
num_hc = length(hc_path);
mask = y_Read(mask_file) ~= 0;

d_tmp_patients = y_Read(patient_path{1});
d_tmp_hc = y_Read(hc_path{1});

if (~all(size(d_tmp_patients) == size(d_tmp_hc)))
    disp('Dimension of the patients and HCs are different');
    return;
end
if  (~all(size(mask) == size(d_tmp_hc)))
    disp('Dimension of the mask and data are different');
    return;
end

data_patient = zeros(num_patient, sum(mask(:)));
data_hc = zeros(num_hc, sum(mask(:)));

for i = 1:num_patient
    data = y_Read(patient_path{i});
    data = data(mask);
    data_patient(i, :) = data;
end

for i = 1:num_hc
    data = y_Read(hc_path{i});
    data = data(mask);
    data_hc(i, :) = data;
end
% Concat data
data_all = cat(1,  data_patient, data_hc);
% Generate unique ID and label
label = [ones(num_patient, 1); zeros(num_hc, 1) - 1];
subj = cat(1, patient_name, hc_name);
ms = regexp( subj, '(?<=\w+)[1-9][0-9]*', 'match' );
nms = length(ms);
subjid = zeros(nms,1);
for i = 1:nms
    tmp = ms{i}{1};
    subjid(i) = str2double(tmp);
end

% PCA
[COEFF, data_all_reduced,~,~,explained] = pca(data_all);
n_comp = numel(explained);
cum_ex_list = zeros(n_comp, 1);
cum_ex = 0;
for i = 1:n_comp
    cum_ex = cum_ex + explained(i);
    cum_ex_list(i) = cum_ex;
end
loc_cutoff_cum_ex = find(cum_ex_list >= explained_cov*100);
loc_cutoff_cum_ex = loc_cutoff_cum_ex(1);
data_all_reduced = data_all_reduced(:,1:loc_cutoff_cum_ex);

% save to csv
data_to_csv = cat(2, subjid, data_all_reduced, label);
data_to_csv = cat(1, data_to_csv, data_to_csv);
csvwrite(fullfile(output_dir, 'cluster_tmp.csv'), data_to_csv);

% Run HYDRA
hydra('-i', fullfile(output_dir, 'cluster_tmp.csv'), '-o', output_dir, '-m', min_clustering_solutions, '-k', max_clustering_solutions, '-f', cvfold);

% Get sub-type uid
load(fullfile(output_dir, 'HYDRA_results.mat'));
idx = CIDX(:, ARI == max(ARI));
uid_subtype = setdiff(unique(idx),  -1);
n_subtype = numel(uid_subtype);
subtype_index = cell(n_subtype, 1);
for i =  1: n_subtype
    subtype_index{i} = ID(idx == uid_subtype(i));
end
save( fullfile(output_dir, 'subtype_index.mat'), 'subtype_index');
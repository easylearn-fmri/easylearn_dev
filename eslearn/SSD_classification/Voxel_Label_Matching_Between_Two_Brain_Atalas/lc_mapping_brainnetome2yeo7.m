%% mapping Brainnetome atalas to Yeo7
% Inputs
brainnetome = 'G:\BranAtalas\BrainnetomeAtlasViewer\BN_Atlas_246_3mm.nii.gz';
yeo7 = 'G:\BranAtalas\Template_Yeo2011\Yeo2011_7Networks_N1000.split_components.FSL_MNI152_3mm.nii';
yeo7_networklabel = 'G:\BranAtalas\Template_Yeo2011\7network_label.xlsx';

% Load nii and network labels
[brainnetome_mat,header] = y_Read(brainnetome);
yeo7_mat = y_Read(yeo7);
[~, yeo_networklabel_mat] = xlsread(yeo7_networklabel);

% Matching
[uni_voxellabel1, prop, matching_idx] = lc_voxel_label_matching_between_two_brain_atalas(brainnetome_mat,yeo7_mat);
matching_idx = cell2mat(matching_idx(1:210));  % exclude subcortex, because Yeo7 did not have subcortex.

% Give name to the subcortex
netlabel = {yeo_networklabel_mat{matching_idx,2}}';
netlabel(211:214) = {'Amyg'};
netlabel(215:218) = {'Hipp'};
netlabel(219:230) = {'BG'};
netlabel(231:246) = {'Tha'};
% Abbreviation of networks' name
uni_label = unique(netlabel);
netlabel(ismember(netlabel,'somatomotor')) = {'SomMot'};
netlabel(ismember(netlabel,'salience / ventral attention')) = {'Sal/VentAttn'};
netlabel(ismember(netlabel,'dorsal attention')) = {'DorsAttn'};
% Upper the initial of networks' name
myfun = @(s) cat(2,upper(s(1)),s(2:end));
netlabel = cellfun(myfun,netlabel,'UniformOutput',false);

%% Producing a new brain atalas1 that in the order of networks
% e.g., 1:12 are BG, so that the voxels' label in the same network are have the serial numbers adjacent to each other
% [sorted_netlabel,idx] = sort(netlabel);  % sorting label in the network order
unilabel = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
new_brainnetome_mat = zeros(size(brainnetome_mat));

num_unilabel = numel(unilabel);
label = 1;  % intial the label with 1, then iteratively increase with step size 1.
netidx = cell(1,num_unilabel);
for i = 1:num_unilabel
    % ith uni_label in all net label
    one_netlabel_loc = ismember(netlabel,unilabel(i));
    regionlabel_in_net = uni_voxellabel1(one_netlabel_loc);
    region_num_in_net = numel(regionlabel_in_net);
    netidx{i} = ones(1,region_num_in_net)+i-1;
    % jth region label in the current net
    for j = 1:region_num_in_net
        new_brainnetome_mat(ismember(brainnetome_mat,regionlabel_in_net(j)))=label;
        label = label + 1;  % increase with step size 1.
    end
end
netidx = cell2mat(netidx)';  % in the order of unilabel.
% Save net index and write ordered new brain atalas to nii.
% save('netIndex.mat','netidx');
% y_Write(new_brainnetome_mat,header,'sorted_brainnetome_atalas_3mm.nii');
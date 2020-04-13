%% mapping from_brain_atalas to to_brain_atalas
% Inputs
% ------------------------------------------
from_brain_atalas = 'G:\BranAtalas\BrainnetomeAtlasViewer\BN_Atlas_246_3mm.nii.gz';
to_brain_atalas = 'G:\BranAtalas\Template_Yeo2011\Yeo2011_7Networks_N1000.split_components.FSL_MNI152_3mm.nii';
% ------------------------------------------

% Load nii and network labels
[from_mat,header] = y_Read(from_brain_atalas);
to_mat = y_Read(to_brain_atalas);

% Matching
[uni_voxellabel1, prop, matching_idx] = lc_voxel_label_matching_between_two_brain_atalas(from_mat, to_mat);

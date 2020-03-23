<font >**lc_mapping_brainnetome2yeo7.m**</font> is used to mapp brainnetome atalas to Yeo 7 network, namely given each voxel in the brainnetome atalas a network label of Yeo 7 networks.  NOTE. The Yeo 7 networks atalas do not have subcortical areas, so the subcortical areas of generated atalas are the same as brainnetome atalas.  
<font >**sorted_brainnetome_atalas_3mm.nii**</font> is the mapped brainnetome atalas.  
<font >**netIndex.mat**</font> is the network index of sorted_brainnetome_atalas_3mm.nii file. The i-th item is the network index of the i-th node.  
The 11 brain networks in the order of the sorted_brainnetome_atalas_3mm are {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};  

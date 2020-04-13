function [uni_label_of_from_atalas, max_prop, matching_idx] = lc_voxel_label_matching_between_two_brain_atalas(from_brain_atalas, to_brain_atalas)
% This function is used to match the voxels' label between two different
% brain atalas.(mapping from_brain_atalas to brain_atalas2). Note. these two brain atalas must have the same dimension and in the same space (MNI)

%   Inputs:
%       from_brain_atalas and to_brain_atalas: two different brain atalas in .mat format.
%   Outputs:
%       uni_region: unique region in from_brain_atalas
%       matching_idx: idx that maching from_brain_atalas to to_brain_atalas.
%       if one matching idx have 2 or more item, then it means that this uni_region matching 2 or more region in brain_atalas2.
% Author: Li Chao
% github account: lichao312214129
% Citing information:
%   This function is part of the easylearn software, if you think this function is useful, citing the easylearn software in your paper or code would be greatly appreciated!
%   Citing link: https://github.com/easylearn-fmri/easylearn
%%

uni_label_of_from_atalas = unique(from_brain_atalas);
uni_label_of_from_atalas = setdiff(uni_label_of_from_atalas,0);  % de-zero
num_region = numel(uni_label_of_from_atalas);
max_prop = cell(num_region,1);
uni_voxellabel = cell(num_region,1);
matching_idx = cell(num_region,1);
for i = 1:num_region
    fprintf('Region %d/%d\n',i,num_region)
    one_regione_in_from_brain_atalas = from_brain_atalas == uni_label_of_from_atalas(i);
    [prop,uni_voxellabel{i,1}] = overlapping_ratio(one_regione_in_from_brain_atalas, to_brain_atalas);
    loc_max_prop=find(prop==max(prop));
    if ~isempty(loc_max_prop)
%         loc_max_prop=loc_max_prop(1);
        max_prop{i} = max(prop);
        matching_idx{i}=uni_voxellabel{i,1}(loc_max_prop);
    else
        matching_idx{i}=0;
    end
end
end


function [prop,uni_voxellabel]=overlapping_ratio(region, whole_brainatalas)
% This function is used to calculate the overlapping ratio of voxels' number between 
% a brain area and another whole brain atlas.
% Inputs:
%    region: a matrix containing all voxels's label in this region. Labels in this region is 1, labels not in this region is 0.
%    whole_brainatalas: a matrix containing all voxels's label in the whole brain atalas. 
%    E.g., if whole_brainatalas have 246 distinct labels, then numel(unique(whole_brainatalas)) == 246 is true.
% Example data:
%   region=rand(3,3,2);
%   region(region>0.6)=5;
%   region(region>0.5 & region <=0.6)=4;
%   region(region>0.4 & region <=0.5)=3;
%   region(region>0.3 & region <=0.4)=2;
%   region(region>0.2 & region <=0.3)=1;
%   region(region<=0.2)=0;
%   region(region~=1)=0;
% 
%   whole_brainatalas=rand(3,3,2);
%   whole_brainatalas(whole_brainatalas>0.6)=6;
%   whole_brainatalas(whole_brainatalas>0.5 & whole_brainatalas <=0.6)=5;
%   whole_brainatalas(whole_brainatalas>0.4 & whole_brainatalas <=0.5)=4;
%   whole_brainatalas(whole_brainatalas>0.3 & whole_brainatalas <=0.4)=3;
%   whole_brainatalas(whole_brainatalas>0.2 & whole_brainatalas <=0.3)=2;
%   whole_brainatalas(whole_brainatalas>0.1 & whole_brainatalas <=0.2)=1;
%   whole_brainatalas(whole_brainatalas <=0.1)=0;

cover_matrix=region.*whole_brainatalas;
cover_matrix(cover_matrix==0)=[];
num_cover_matrix = numel(cover_matrix);
uni_voxellabel=unique(cover_matrix);
prop=zeros(1,numel(uni_voxellabel));
num_uni_voxellabel = numel(uni_voxellabel);
for i=1:num_uni_voxellabel
    prop(i)=sum(cover_matrix==uni_voxellabel(i))/num_cover_matrix;
end
end

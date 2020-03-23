function [uni_label1, max_prop, matching_idx] = lc_voxel_label_matching_between_two_brain_atalas(brain_atalas1,brain_atalas2)
% This function is used to match the voxels' label between two different
% brain atalas.(mapping brain_atalas1 to brain_atalas2)
% Ref. : {Chronnectome fingerprinting: Identifying individuals and
% predicting higher cognitive functions using dynamic brain connectivity patterns}
%   input:
%       brain_atalas1 and brain_atalas1: two different brain atalas.
%   output:
%       uni_region: unique region in brain_atalas1, in the order of
%       matching idx
%       matching_idx: idx that maching brain_atalas1 to brain_atalas2.
%       if one matching idx have 2 or more item, then it means that this uni_region matching 2 or more region in brain_atalas2.
%%
uni_label1=unique(brain_atalas1);
uni_label1 = setdiff(uni_label1,0);  % de-zero
num_region=numel(uni_label1);
max_prop=cell(num_region,1);
uni_voxellabel=cell(num_region,1);
matching_idx=cell(num_region,1);
for i =1:num_region
    fprintf('Region %d/%d\n',i,num_region)
    one_regione_in_brain_atalas1=brain_atalas1==uni_label1(i);
    [prop,uni_voxellabel{i,1}]=overlapping_ratio(one_regione_in_brain_atalas1,brain_atalas2);
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

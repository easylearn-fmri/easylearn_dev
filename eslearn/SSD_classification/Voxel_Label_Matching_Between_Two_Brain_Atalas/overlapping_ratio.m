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
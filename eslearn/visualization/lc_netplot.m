function lc_netplot(varargin)
% LC_NETPLOT
% PURPOSE: plot functional connectivity network using grid format. 
% NOTO: This function will automatically sort network according to net_index.
% Parameters:
% -----------
%   REQUIRED:
%       [--net, -n]: path str | .mat matrix
%           Functional connectivity network that needs to be plotted.
%       [--net_index, -ni]: path str | .mat vector
%           network index, each node has a net_index indicating its network index
%   OPTIONAL:
%      [--if_add_mask, -iam]: int, 0 or 1
%           If add mask to net for filtering.
%      [--mask, -m]: path str | .mat matrix
%           Mask that used to filter the net.
%      [--how_disp, -hd]: str:: 'only_neg', 'only_pos' or 'all'
%           how display the network
%      [--if_binary, -ib]: int, 0 or 1
%           If binary the network.
%      [--which_group, -wg]: int
%           If the network .mat file has multiple 2D matrix, then choose which one to display.
%      [--linewidth, -lw]: float 
%           separation line width
%      [--linecolor, -lc]: color string 
%           separation line color
%      [--is_legend, -il]: int
%          If show legend.
%      [--legends, -lg]: cell
%          legends of each network.
%      [--legend_fontsize, -lgf]: float
%          fontsize of the legends
%   
% EXAMPLE:
% lc_netplot('-n', 'tvalue_medication.mat', '-ni', 'netIndex.mat')
% AUTHOR: Li Chao
% EMAIL: lichao19870617@gmail.com, lichao19870617@163.com
% If you use this code, please cite "easylearn":
% For example: "This figure was plotted using easylearn: https://github.com/easylearn-fmri/easylearn"

if nargin == 0
    help lc_netplot
    return;
end

[net, if_add_mask, mask, how_disp, if_binary, which_group, net_index, linewidth, linecolor, is_legend, legends, legend_fontsize] = ...
            parseInputs(varargin{:});

% net
if isa(net, 'char')
    net=importdata(net);
else
    net=net;
end

% show postive and/or negative
if strcmp(how_disp,'only_pos')
    net(net<0)=0;
elseif strcmp(how_disp,'only_neg')
    net(net>0)=0;
elseif strcmp(how_disp,'all')
    disp('show both postive and negative')
else
    disp('Did not specify show positive or negative!')
    return
end

% when matrix is 3D, show which (the 3ith dimension)
if numel(size(net))==3
    %     net=squeeze(net(which_group,:,:));
    net=squeeze(net(:,:,which_group));
end

% transfer the weighted matrix to binary
if if_binary
    net(net<0)=-1;
    net(net>0)=1;
end

% mask
if if_add_mask
    if isa(mask, 'char')
        mask=importdata(mask);
    else
        mask=mask;
    end
    
    % when mask is 3D, show which (the 3ith dimension)
    if numel(size(mask))==3
        mask=squeeze(mask(which_group,:,:));
    end
    
    % extract data in mask
    net=net.*mask;
end

% sort the matrix according to network index
net_index=importdata(net_index);
[index,re_net_index,re_net]=lc_ReorganizeNetForYeo17NetAtlas(net,net_index);

% plot: insert separate line between each network
lc_InsertSepLineToNet(re_net, re_net_index, linewidth, linecolor, is_legend, legends, legend_fontsize);
end


function [net, if_add_mask, mask, how_disp, if_binary, which_group, net_index, linewidth, linecolor, is_legend, legends, legend_fontsize] = ...
            parseInputs(varargin)
% Varargin parser

% Initialize
if_add_mask = 0;
mask = '';
how_disp='all';
if_binary=0;
which_group=1;
linewidth = 0.5;
linecolor = 'k';
is_legend = 0;
legends = '';
legend_fontsize = 10;

if( sum(or(strcmpi(varargin,'--net'),strcmpi(varargin,'-n')))==1)
    net = varargin{find(or(strcmpi(varargin,'--net'),strcmp(varargin,'-n')))+1};
else
    error('Please specify net!');
end

if( sum(or(strcmpi(varargin,'--if_add_mask'),strcmpi(varargin,'-iam')))==1)
    if_add_mask = varargin{find(or(strcmpi(varargin,'--if_add_mask'),strcmp(varargin,'-iam')))+1};
end

if( sum(or(strcmpi(varargin,'--mask'),strcmpi(varargin,'-m')))==1)
    mask = varargin{find(or(strcmpi(varargin,'--mask'),strcmp(varargin,'-m')))+1};
end

if( sum(or(strcmpi(varargin,'--how_disp'),strcmpi(varargin,'-hd')))==1)
    how_disp = varargin{find(or(strcmpi(varargin,'--how_disp'),strcmp(varargin,'-hd')))+1};
end

if( sum(or(strcmpi(varargin,'--if_binary'),strcmpi(varargin,'-ib')))==1)
    if_binary = varargin{find(or(strcmpi(varargin,'--if_binary'),strcmp(varargin,'-ib')))+1};
end

if( sum(or(strcmpi(varargin,'--which_group'),strcmpi(varargin,'-wg')))==1)
    which_group = varargin{find(or(strcmpi(varargin,'--which_group'),strcmp(varargin,'-wg')))+1};
end

if( sum(or(strcmpi(varargin,'--net_index'),strcmpi(varargin,'-ni')))==1)
    net_index = varargin{find(or(strcmpi(varargin,'--net_index'),strcmp(varargin,'-ni')))+1};
end

if( sum(or(strcmpi(varargin,'--linewidth'),strcmpi(varargin,'-lw')))==1)
    linewidth = varargin{find(or(strcmpi(varargin,'--linewidth'),strcmp(varargin,'-lw')))+1};
end

if( sum(or(strcmpi(varargin,'--linecolor'),strcmpi(varargin,'-lc')))==1)
    linecolor = varargin{find(or(strcmpi(varargin,'--linecolor'),strcmp(varargin,'-lc')))+1};
end

if( sum(or(strcmpi(varargin,'--is_legend'),strcmpi(varargin,'-il')))==1)
    is_legend = varargin{find(or(strcmpi(varargin,'--is_legend'),strcmp(varargin,'-il')))+1};
end

if( sum(or(strcmpi(varargin,'--legends'),strcmpi(varargin,'-lg')))==1)
   legends = varargin{find(or(strcmpi(varargin,'--legends'),strcmp(varargin,'-lg')))+1};
end

if( sum(or(strcmpi(varargin,'--legend_fontsize'),strcmpi(varargin,'-lgf')))==1)
   legend_fontsize = varargin{find(or(strcmpi(varargin,'--legend_fontsize'),strcmp(varargin,'-lgf')))+1};
end

end

function lc_InsertSepLineToNet(net, re_net_index, linewidth, linecolor, is_legend, legends, legend_fontsize)
% 此代码的功能：在一个网络矩阵种插入网络分割线，以及bar
% 此分割线将不同的脑网络分开
% 不同颜色的区域，代表一个不同的网络
% input
%   net:一个网络矩阵，N*N,N为节点个数，必须为对称矩阵
%   network_index: network index of each node.
%   location_of_sep:分割线的index，为一个向量，比如[3,9]表示网络分割线分别位于3和9后面
%% input
% if not given location_of_sep, then generate it using network_index;
uni_id = unique(re_net_index);
location_of_sep = [0; cell2mat(arrayfun( @(id) max(find(re_net_index == id)), uni_id, 'UniformOutput',false))];

if size(net,1)~=size(net,2)
    error('Not a symmetric matrix!\n');
end

%%

%% Gen new sep line and new network
n_node = length(net);
% New sep
num_sep = numel(location_of_sep);
location_of_sep_new = location_of_sep;
for i =  2 : num_sep
    location_of_sep_new(i:end) = location_of_sep_new(i:end) + 1;
end
% New network
net_insert_line = zeros(n_node + num_sep, n_node + num_sep);
for i = 1:num_sep-1
    % Rows iteration
    start_point =  location_of_sep_new(i) + 1;
    end_point = location_of_sep_new(i+1) - 1;
    start_point_old =  location_of_sep(i) + 1;
    end_point_old = location_of_sep(i+1);
    % Columns iteration
    for j = 1 : num_sep - 1
        start_point_j =  location_of_sep_new(j) + 1;
        end_point_j = location_of_sep_new(j+1) - 1;
        start_point_old_j =  location_of_sep(j) + 1;
        end_point_old_j = location_of_sep(j+1);
        net_insert_line(start_point : end_point, start_point_j : end_point_j) = ...
                    net(start_point_old : end_point_old, start_point_old_j : end_point_old_j);
    end
end
imagesc(net_insert_line); hold on;
x = repmat(location_of_sep_new', num_sep ,1);
y = repmat(location_of_sep_new,1, num_sep);
x(:,end) = x(:,end) + 1.5;
y(end,:) = y(end, :) + 1.5;
z = zeros(size(x));
mesh(x,y,z,...
    'EdgeColor',linecolor,...
    'FaceAlpha',0,...
    'LineWidth',linewidth);
view(2);
grid off
% lc_line(location_of_sep, n_node, linewidth, linecolor);
hold on;
% bar region
n_node_new = length(net_insert_line);
extend = n_node_new / 10;
xlim([0, n_node_new + extend]);
ylim([0, n_node_new + extend]);
lc_bar_region_of_each_network(location_of_sep_new, n_node_new, extend, is_legend, legends, legend_fontsize);
axis off
end

function lc_line(location_of_sep, n_node, linewidth, linecolor)
% nNode: node个数
n_net = length(location_of_sep);
for i=1:n_net
    if (i == 1)  
        % Y
        line([location_of_sep(i), location_of_sep(i)],[0, n_node],'color',linecolor,'LineWidth',linewidth);
        % X
        line([0, n_node],[location_of_sep(i), location_of_sep(i)],'color',linecolor,'LineWidth',linewidth);
    elseif (i == n_net)
        % Y
        line([location_of_sep(i) + 0.5, location_of_sep(i) + 0.5],[0, n_node],'color',linecolor,'LineWidth',linewidth);
        % X
        line([0, n_node],[location_of_sep(i) + 0.5, location_of_sep(i) + 0.5],'color',linecolor,'LineWidth',linewidth);
        
    else
        % Y
        line([location_of_sep(i) + 0.5, location_of_sep(i) + 0.5],[0, n_node],'color',linecolor,'LineWidth',linewidth);
        % X
        line([0, n_node],[location_of_sep(i) + 0.5, location_of_sep(i) + 0.5],'color',linecolor,'LineWidth',linewidth);
    end
    
end
end

function lc_bar_region_of_each_network(location_of_sep, n_node, extend, is_legend, legends, legend_fontsize)
% To plot bar with sevral regions, each region with a unique color
% representting a network.
n_net = length(location_of_sep);
randseed(1);
color = jet(n_net) / 1.2;
barwidth = abs((n_node + extend / 2) - (n_node+extend));
extend_of_legends = extend + 4 ;
h = zeros(n_net - 1, 1);
for i = 1 : n_net-1
    h(i) = fill([location_of_sep(i), location_of_sep(i+1), location_of_sep(i+1), location_of_sep(i)], [n_node + extend / 2, n_node + extend / 2, n_node+extend n_node + extend], color(i,:));
    fill([ n_node + barwidth, n_node + barwidth, n_node + extend, n_node + extend], [location_of_sep(i), location_of_sep(i+1), location_of_sep(i+1), location_of_sep(i)], color(i,:))
    if is_legend
        % Y axix
        text(n_node + extend_of_legends, (location_of_sep(i+1) - location_of_sep(i)) / 2 +  location_of_sep(i),...
            legends{i}, 'fontsize', legend_fontsize, 'rotation', 0);
         % X axix
        text((location_of_sep(i+1) - location_of_sep(i)) / 2 +  location_of_sep(i), n_node + extend_of_legends,...
            legends{i}, 'fontsize', legend_fontsize, 'rotation', -90);
    end
end
end
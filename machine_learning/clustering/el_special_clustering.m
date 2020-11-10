function [ C, L, D, Q, V ] = el_special_clustering(W, k)
% spectral clustering algorithm
% Input: 
% -----
%   W: adjacency matrix with N * N dimension, N is the number of samples; 
%   k: number of cluster k 

% return: 
% ------
%   C: cluster indicator vectors as columns in 
%   L: unnormalized Laplacian
%   D: degree matrix
%   Q: eigenvectors matrix
%   V: eigenvalues matrix
% NOTE. This function is revised from https://www.cnblogs.com/FengYan/archive/2012/06/21/2553999.html

% Calculate degree matrix
degs = sum(W, 2);
D = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;

% compute the eigenvectors corresponding to the k smallest eigenvalues
% diagonal matrix V is NcutL's k smallest magnitude eigenvalues 
% matrix Q whose columns are the corresponding eigenvectors.
[Q, V] = eigs(L, k, 'SA');

% use the k-means algorithm to cluster V row-wise
% C will be a n-by-1 matrix containing the cluster number for each data point
C = kmeans(Q, k);
end
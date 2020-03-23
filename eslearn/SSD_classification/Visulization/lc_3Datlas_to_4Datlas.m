function lc_3Datlas_to_4Datlas(filename, savename)
% Transform 3D image (fname) to 4D image (each 3D image only containing one ROI/component, namely each ROI/component per frame).
% Parameters:
%------------
%   filename: string
%        The nifti file name that needs to be transformed from 3D to 4D (each ROI/component per frame).
%   savename: string
%        The name of the output 4D nifti file.
%% ------------------------------------------------------------------------------------------------------
[path, name, suffix] = fileparts(filename);
[image_3d, header] = y_Read(filename);
uni_roi = setdiff(unique(image_3d(:)), 0);
n_roi = numel(uni_roi);
[i, j, k] = size(image_3d);
image_4d = zeros(i, j, k, n_roi);
for i = 1:n_roi
    i
    image_3d_i = image_3d;
    image_3d_i(image_3d_i ~= uni_roi(i)) = 0;
    image_4d(:,:,:,i) = image_3d_i;
end
y_Write(image_4d, header, savename);
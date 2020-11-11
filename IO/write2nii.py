def mat2nii(self, data, header, dimension_nii_data=(61, 73, 61)):
    """Transfer weight matrix to nii file

    I used the mask file as reference to generate the nii file
    """
    weight = np.squeeze(data)
    weight_mean = np.mean(weight, axis=0)

    # to orignal space
    weight_mean_orig = np.zeros(np.size(self.mask_all))
    weight_mean_orig[self.mask_all] = weight_mean
    weight_mean_orig =  np.reshape(weight_mean_orig, dimension_nii_data)
    # save to nii
    weight_nii = nib.Nifti1Image(weight_mean_orig, affine=self.mask_obj.affine)
    weight_nii.to_filename(os.path.join(self.path_out, 'weight.nii'))
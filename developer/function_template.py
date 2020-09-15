# -*- coding: utf-8 -*-
""" This module is used to perform dimension reduction

Created on Wed Jul  4 13:57:15 2018
@author: Li Chao
Email:lichao19870617@gmail.com
GitHub account name: lichao312214129
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 
License: MIT
"""


from sklearn.decomposition import PCA

def pca_apply(train_x, test_x, pca_n_component):
    """Fit pca from training data, then apply the pca model to test data

    Parameters
    ----------
        train_x : numpy.ndarray
            features in the training dataset
        test_x : numpy.ndarray
            features in the test dataset
        pca_n_component : float, range = (0, 1]
            how many percentages of the cumulatively explained variance to be retained. This is used to select the top principal components.
    Returns
    ------
        train_x_reduced: numpy.ndarray
            features in the training dataset after dimension reduction
        test_x_reduced: numpy.ndarray
            features in the test dataset after dimension reduction
    """

    train_x_reduced, trained_pca = pca(train_x, pca_n_component)
    test_x_reduced = trained_pca.transform(test_x)
    return train_x_reduced, test_x_reduced, trained_pca

def pca(x, n_components):
    """Just training a pca model

    Parameters
    ----------
        x : numpy.ndarray
            features in the training dataset
        pca_n_component : float, range = (0, 1]
            how many percentages of the cumulatively explained variance to be retained. This is used to select the top principal components.
    return
    ------
        x_reduced: numpy.ndarray
            features in the training dataset after dimension reduction
    """  

    trained_pca = PCA(n_components=n_components)
    reduced_x = trained_pca.fit_transform(x)
    return reduced_x, trained_pca


# The following code is used to debug.
if __name__ == "__main__":
    from sklearn import datasets
    x, y = datasets.make_classification(n_samples=500, n_classes=3,
                                        n_informative=50, n_redundant=3,
                                        n_features=600, random_state=1)
    
    x1, x2, _ = pca_apply(x, x, 0.9)
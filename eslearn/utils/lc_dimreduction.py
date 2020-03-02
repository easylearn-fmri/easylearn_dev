# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:57:15 2018
dimension reduction
current only pca
@author: lenovo
"""
from sklearn.decomposition import PCA

def pca_apply(train_x, test_x, pca_n_component):
    """apply pca to test data
    """
    train_x, trained_pca = pca(train_x, pca_n_component)
    test_x = trained_pca.transform(test_x)
    return train_x, test_x, trained_pca

def pca(x, n_components):
    """training a pca model
    """
    trained_pca = PCA(n_components=n_components)
    reduced_x = trained_pca.fit_transform(x)
    return reduced_x, trained_pca


if __name__ == "__main__":
    from sklearn import datasets
    x, y = datasets.make_classification(n_samples=500, n_classes=3,
                                        n_informative=50, n_redundant=3,
                                        n_features=600, random_state=1)
    
    x1, x2, _ = pca_apply(x, x, 0.9)
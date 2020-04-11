"""This function was advised from https://gist.github.com/satra/aa3d19a12b74e9ab7941 

I thanks Satrajit Ghosh.
"""


from scipy.spatial.distance import pdist, squareform
import numpy as np

def distcorr(X, Y):
    """ Compute the distance correlation function

    Parameters:
    ----------
        X: array
            feature array
        Y: array
            target array
    Returns:
    --------
        dcor: float

    """


    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


if __name__ == "__main__":
    a = [1,2,3,4,5]
    b = np.array([1,2,9,4,4])
    print(distcorr(a, b))
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
estimator= LinearRegression()
estimator = linear_model.BayesianRidge()


selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)

dir(selector)
s = selector.support_

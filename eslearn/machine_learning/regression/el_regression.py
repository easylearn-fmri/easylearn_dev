"""
This class is used to perform regression
"""


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor


class Regression():
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def linear_regression(self, x, y):
        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        
    def lasso_regression(self, x, y, alpha):
        reg = linear_model.Lasso(alpha=alpha)
        reg.fit(x, y)
        
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        
    def ridge_regression(self, x, y, alpha):
        reg = Ridge(alpha=alpha)
        reg.fit(x, y)
        
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        
    def elasticnet_regression(self, x, y, l1_ratio, alpha):
        reg = ElasticNet(l1_ratio=l1_ratio, alpha=alpha, random_state=0)
        reg.fit(x, y)
        
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        
    def svm_regression(self, x, y, kernel, C, gamma):
        reg = SVR(kernel=kernel, C=C, gamma=gamma)
        reg.fit(x, y)
        
        if kernel == "linear":
            self.coef_ = reg.coef_
            self.intercept_ = reg.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None
            
    def gaussian_process(self, x, y, kernel, alpha):
        reg = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=0)
        reg.fit(x, y)
        
    def random_forest(self, x, y, criterion, n_estimators, max_depth):
        reg = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        reg.fit(x, y)
    
    
if __name__ == "__main__":
    regr = Regression()
    regr.linear_regression(x, y)
    regr.lasso_regression(x, y, alpha)
    regr.ridge_regression(x, y, alpha)
    regr.elasticnet_regression(x, y, l1_ratio, alpha)
    regr.svm_regression(x, y, kernel, C, gamma)
    regr.gaussian_process(x, y, kernel, alpha)
    regr.random_forest(x, y, criterion, n_estimators, max_depth)
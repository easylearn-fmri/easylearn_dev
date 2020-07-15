"""
This class is used to perform regression
"""


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np


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
    
    
if __name__ == "__main__":
    reg = Regression()
    reg.linear_regression(x, y)
    reg.lasso_regression(x, y, alpha)
    reg.ridge_regression(x, y, alpha)
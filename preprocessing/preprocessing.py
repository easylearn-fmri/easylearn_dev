import pandas as pd
import numpy as np


class Preprocessing():
    """Preprocessing features
    """

    def __init__(self):
        pass


class Denan(Preprocessing):
    """Denan
    """

    def __init__(self, how="median"):
        super().__init__()
        self.how=how

    def fit(self, features, label=None):
        """ Handle extreme values
        
        Currently, we fillna with median
        TODO: Add other extreme values' handling methods

        Parameters: 
        ----------
        features: DataFrame or ndarray
            all features

        Return:
        ------
        self : object
        """
        
        
        if self.how == "median":
            self.value_ = np.median(pd.DataFrame(features).dropna(axis="index"), axis=0)
        elif self.how == "mean":
            self.value_ =np.mean(pd.DataFrame(features).dropna(axis="index"), axis=0)
        
        self.value_ = pd.Series(self.value_) 

        return self

    def transform(self, features, label=None):
        if np.isnan(features).any().sum() > 0:
            if not isinstance(features, pd.core.frame.DataFrame):
                features = pd.DataFrame(features)

            features = features.fillna(value=self.value_)

        return features

    def fit_transform(self, features, label=None):
        self.fit(features)
        return self.transform(features)

        return features
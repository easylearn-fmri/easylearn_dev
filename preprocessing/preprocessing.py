import pandas as pd
import numpy as np


def denan(features, how="median"):
    """ Handle extreme values
    
    Currently, we fillna with median
    TODO: Add other extreme values' handling methods

    Parameters: 
    ----------
    features: DataFrame or ndarray
        all features

    Return:
    ------
    features: DataFrames
        all features that be handled extreme values
    """
    
    value = None
    if np.isnan(features).any().sum() > 0:
        if not isinstance(features, pd.core.frame.DataFrame):
            features = pd.DataFrame(features)
            if how == "median":
                value = features.median()
                features = features.fillna(value=value)
            elif how == "mean":
                value = features.mean()
                features = features.fillna(value=value)  

    return features, value
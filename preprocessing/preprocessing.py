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
    
    
    if how == "median":
        value = np.median(pd.DataFrame(features).dropna(axis="index"), axis=0)
    elif how == "mean":
        value =np.mean(pd.DataFrame(features).dropna(axis="index"), axis=0)
    
    value = pd.Series(value)
                
    if np.isnan(features).any().sum() > 0:
        if not isinstance(features, pd.core.frame.DataFrame):
            features = pd.DataFrame(features)
            if how == "median":
                features = features.fillna(value=value)
            elif how == "mean":
                features = features.fillna(value=value)  

    return features, value
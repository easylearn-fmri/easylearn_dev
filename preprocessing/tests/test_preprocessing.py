import pandas as pd
import numpy as np
import pytest

from eslearn.preprocessing.preprocessing import denan


import pytest
@pytest.mark.parametrize("how",["median", "mean"])

def test_denan(how):
    features = np.random.randn(10,5)
    features[0,2] = np.nan
    features, value = denan(features, how)
    assert features.shape == (10,5)


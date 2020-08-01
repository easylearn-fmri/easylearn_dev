#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Unbalance treatment
"""

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=0)
feature_resampled, label_resampled = ros.fit_resample(feature, label)
from collections import Counter
print(f"After re-sampling, the sample size are: {sorted(Counter(label_resampled).items())}")
return feature_resampled, label_resampled
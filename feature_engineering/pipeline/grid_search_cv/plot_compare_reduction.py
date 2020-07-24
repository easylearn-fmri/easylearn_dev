#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.

###############################################################################
Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################

This section illustrates the use of a ``Pipeline`` with ``GridSearchCV``
"""

# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from joblib import Memory
from shutil import rmtree

print(__doc__)
location = 'cachedir'
memory = Memory(location=location, verbose=10)

pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', NMF()),
        ('feature_selection', SelectKBest(chi2)),
        ('classify', LogisticRegression(solver='saga', penalty='l1'))
    ], 
    memory=memory
)

N_FEATUREDIM_OPTIONS = [2, 4, 8]
N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1]
l1_ratio = [0, 0.1, 0.5, 1]

param_grid = [
    {
        'reduce_dim__n_components': N_FEATUREDIM_OPTIONS,
        'feature_selection__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__l1_ratio': l1_ratio,
    },
]

reducer_labels = ['PCA', 'KBest(chi2)']

grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
X, y = load_digits(return_X_y=True)
grid.fit(X, y)

# Delete the temporary cache before exiting
memory.clear(warn=False)
rmtree(location)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()
# import modules
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances


class PermutationImportance_(object):
    """
    see https://www.kaggle.com/dansbecker/permutation-importance for details

    Parameters:
    -----------
    model: model object that have predict method

    feature: ndarray with shape of [n_samples, n_features]
        Machine learning features

    target: ndarray with shape of [n_samples,]
        Machine learning features

    Attributes:
    --------
    weight_: ndarray with shape of [n_features, ]
        Feature importance (weight)
    """

    def __init__(self, model, feature, target):
        self.model = model
        self.feature = feature
        self.target = target

    def get_permutation_importance_accuracy(self):
        base_score, score_decreases = get_score_importances(self.score_acc, self.feature, self.target)
        self.weight_ = np.mean(score_decreases, axis=0)
        return self

    def score_acc(self, feature, target):
        y_pred, y_prob = self.model.predict(feature)
        accuracy = get_acc(target, y_pred)  # you should define get_acc, or you using eslearn.model_evaluator
        return accuracy

    @staticmethod
    def get_acc(target, y_pred):
        pass
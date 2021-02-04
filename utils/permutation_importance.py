import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances


class PermutationImportance_(object):
    """
    see https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
    and see https://www.kaggle.com/dansbecker/permutation-importance 
    for details

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

    def __init__(self, model, feature, target, metric="accuracy"):
        self.model = model
        self.feature = feature
        self.target = target
        self.metric = metric

        self.metric_dict = {"accuracy": self.score_acc, "f1": self.score_f1}  # you should update it

    def fit(self):
        base_score, score_decreases = get_score_importances(
                                        self.metric_dict[self.metric], 
                                        self.feature, 
                                        self.target
        )

        self.weight_ = np.mean(score_decreases, axis=0)
        return self

    def score_acc(self, feature, target):
        """Get accuracy
        """

        y_pred, y_prob = self.model.predict(feature)
        accuracy = get_acc(target, y_pred)  # you should define get_acc, or you using eslearn.model_evaluator
        return accuracy

    def score_f1(self, feature, target):
        """Get F1 score
        """

        y_pred, y_prob = self.model.predict(feature)
        accuracy = get_f1(target, y_pred)  # you should define get_f1, or you using eslearn.model_evaluator
        return accuracy

    @staticmethod
    def get_acc(target, y_pred):
        pass

    @staticmethod
    def get_f1(target, y_pred):
        pass


if __name__ == "__main__":
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    X, y = make_classification(n_features=4, random_state=0)
    model = LinearSVC(random_state=0, tol=1e-5)
    model.fit(X, y)

    permimp = PermutationImportance_(model, X, y)
    permimp.fit()
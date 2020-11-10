from abc import abstractmethod, ABCMeta

class AbstractSupervisedMachineLearningBase(metaclass=ABCMeta):
    """Abstract base class for supervised learning: _base_classificaition and _base_regression
     and _base_clustering

    """

    @abstractmethod
    def fit_(self):
        raise NotImplementedError

    @abstractmethod
    def predict_(self):
        raise NotImplementedError

    @abstractmethod
    def get_weights_(self):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """

        raise NotImplementedError


class AbstractUnsupervisedMachineLearningBase(metaclass=ABCMeta):
    """Abstract base class for unsupervised learning: _base_clustering

    """

    @abstractmethod
    def fit_(self):
        raise NotImplementedError

    @abstractmethod
    def predict_(self):
        raise NotImplementedError

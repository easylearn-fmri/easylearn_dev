from abc import abstractmethod, ABCMeta

class AbstractSupervisedMachineLearningBase(metaclass=ABCMeta):
    """Abstract base class for supervised learning: _base_classificaition and _base_regression
     and _base_clustering

    """

    @abstractmethod
    def fit_(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def get_weights_(self):
        raise NotImplementedError


class AbstractUnsupervisedMachineLearningBase(metaclass=ABCMeta):
    """Abstract base class for unsupervised learning: _base_clustering

    """

    @abstractmethod
    def fit_(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

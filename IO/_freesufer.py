from abc import abstractmethod, ABCMeta

class _Freesufer(metaclass=ABCMeta):
    """Abstract Freesufer classc

    """

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def write(self):
        raise NotImplementedError

import abc


class InputDeviceInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "timeout") and
                hasattr(subclass, "init") and
                callable(subclass.init) and
                hasattr(subclass, "read_sample") and
                callable(subclass.read_sample) and
                hasattr(subclass, "deinit") and
                callable(subclass.deinit))  # or
        # NotImplemented)

    @abc.abstractmethod
    def init(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def read_sample(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def deinit(self, **kwargs):
        raise NotImplementedError

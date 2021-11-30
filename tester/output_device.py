import abc


class OutputDeviceInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "timeout") and
                hasattr(subclass, "init") and
                callable(subclass.init) and
                hasattr(subclass, "write_sample") and
                callable(subclass.write_sample) and
                hasattr(subclass, "deinit") and
                callable(subclass.deinit))  # or
        # NotImplemented)

    @abc.abstractmethod
    def init(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def write_sample(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def deinit(self, **kwargs):
        raise NotImplementedError

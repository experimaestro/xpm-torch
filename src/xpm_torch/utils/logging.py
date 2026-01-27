import logging
from functools import cached_property


class EasyLogger:
    @cached_property
    def logger(self):
        clsdict = self.__class__.__dict__

        logger = clsdict.get("__LOGGER__", None)
        if logger is None:
            logger = logging.getLogger(self.__class__.__qualname__)
            self.__class__.__LOGGER__ = logger

        return logger


class LazyJoin:
    """Lazy join of an iterator"""

    def __init__(self, glue: str, iterator):
        self.glue = glue
        self.iterator = iterator

    def __str__(self):
        return self.glue.join(str(x) for x in self.iterator)

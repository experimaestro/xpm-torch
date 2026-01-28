import logging
from functools import cached_property


class EasyLogger:
    """
    A mixin that provides a lazy-loaded, class-level logger instance, so you can log with `self.logger.info('...')`.


    When the `logger` property is first accessed on an instance, it creates 
    a `logging.Logger` using the class's qualified name (e.g. 'MyModule.MyClass')
    and caches it on the class itself as `__LOGGER__`.
    
    All instances of the class share the same logger object.
    """
    @cached_property
    def logger(self):
        # Check if the class already has the logger cached
        if not hasattr(self.__class__, "__LOGGER__"):
            # If not, create it
            self.__class__.__LOGGER__ = logging.getLogger(self.__class__.__qualname__)
        
        return self.__class__.__LOGGER__

class LazyJoin:
    """Lazy join of an iterator"""

    def __init__(self, glue: str, iterator):
        self.glue = glue
        self.iterator = iterator

    def __str__(self):
        return self.glue.join(str(x) for x in self.iterator)

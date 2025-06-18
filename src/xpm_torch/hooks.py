import logging
from experimaestro import Param
from xpm_torch.context import TrainState, InitializationTrainingHook
from xpm_torch.parameters import ParametersIterator
from xpm_torch.utils.logging import easylog

logger = easylog()
logger.setLevel(logging.INFO)


from typing import Callable, DefaultDict, Dict, List, Type, TypeVar
from experimaestro import Config


class Hook(Config):
    """Base class for all hooks"""

    pass


HookType = TypeVar("HookType")


class InitializationHook(Hook):
    """Base class for hooks before/after initialization"""

    def after(self, context: "Context"):
        """Called after initialization"""
        pass

    def before(self, context: "Context"):
        """Called before initialization"""
        pass


class LayerFreezer(InitializationTrainingHook):
    """This training hook class can be used to freeze a subset of model
    parameters"""

    selector: Param[ParametersIterator]
    """How to select the layers to freeze"""

    def __init__(self):
        self._initialized = False

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, module, param, to_freeze in self.selector.iter():
                if to_freeze:
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False


class LayerSharer(InitializationTrainingHook):
    """This training hook class can be used to share parameters"""

    source: Param[ParametersIterator]
    """The parameters to share"""

    target: Param[ParametersIterator]
    """The parameters to be shared"""

    def __init__(self):
        self._initialized = False

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for source, target in zip(
                self.source.selected(), self.target.selected(), strict=True
            ):
                logger.info("Sharing layer %s -> %s", source.name, target.name)
                target.set(source.parameter)


class Context:
    """Generic computational context"""

    hooksmap: dict[Type, List[Hook]]
    """Map of hooks"""

    def __init__(self, hooks: List[Hook] = []):
        self.hooksmap = DefaultDict(lambda: [])
        for hook in hooks:
            self.add_hook(hook)

    def hooks(self, cls: Type[HookType]) -> List[HookType]:
        """Returns all the hooks"""
        return self.hooksmap.get(cls, [])  # type: ignore

    def call_hooks(self, cls: Type, method: Callable, *args, **kwargs):
        for hook in self.hooks(cls):
            method(hook, *args, **kwargs)

    def add_hook(self, hook):
        for cls in hook.__class__.__mro__:
            self.hooksmap[cls].append(hook)


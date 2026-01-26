from experimaestro import Param
from xpm_torch.parameters import ParametersIterator
from xpm_torch.trainers.context import TrainState, InitializationTrainingHook

import logging
from xpm_torch.utils.logging import easylog

logger = easylog()
logger.setLevel(logging.INFO)

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
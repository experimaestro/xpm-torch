import logging
from typing import Callable, ParamSpec
from experimaestro import Config, Param, Meta
import lightning.fabric.strategies as strategies
import lightning as L
logger = logging.getLogger("xpm_torch.configuration")

P = ParamSpec("P")

class Strategy(Config, strategies.Strategy):
    pass


class Configuration(Config):
    """Describe the computation device

    The backend is fabric, so the complete documentation can be found on
    https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    """

    num_nodes: Meta[int] = 1
    """Number of nodes"""

    devices: Meta[list[int] | int | str] = "auto"
    """List of devices to use"""

    strategy: Meta[str | Strategy] = "auto"

    accelerator: Meta[str] = "auto"
    """The accelerator to use"""

    precision: Param[str]
    """Precision to use"""
    
    def launch(self, train: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
        fabric = L.Fabric(accelerator=self.accelerator, devices=self.devices, strategy=self.strategy, num_nodes=1, loggers=[])
        fabric.run(train)

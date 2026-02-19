import logging
from typing import Callable, ParamSpec
from experimaestro import Config, Param, Meta
import lightning.fabric.strategies as strategies
import lightning as L

logger = logging.getLogger("xpm_torch.configuration")

P = ParamSpec("P")


class Strategy(Config, strategies.Strategy):
    pass


class FabricConfiguration(Config):
    """Describe the computation device

    The backend is fabric, so the complete documentation can be found on
    https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    """

    num_nodes: Meta[int] = 1
    """Number of nodes"""

    devices: Meta[str] = "auto"
    """List of devices to use"""

    strategy: Meta[str] = "auto"

    accelerator: Meta[str] = "auto"
    """The accelerator to use"""

    precision: Param[str] = "32-true"
    """Precision to use, e.g., '16-mixed', 'bf16-mixed', '32-true': 
    see Lightning documentation at https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
    """

    def get_instance(self, **kwargs):
        """instanciate the Fabric object"""
        return L.Fabric(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            **kwargs
        )

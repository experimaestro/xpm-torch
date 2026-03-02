import logging
from typing import Callable, Optional, ParamSpec
from experimaestro import Config, Param, Meta
import lightning.fabric.strategies as strategies
import lightning as L
import torch
logger = logging.getLogger("xpm_torch.configuration")

P = ParamSpec("P")


class Strategy(Config, strategies.Strategy):
    pass


class FabricConfiguration(Config):
    """Describe the computation device

    The backend is fabric, so the complete documentation can be found on
    https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    """
    #parameters - change Learner output
    precision: Param[str] = "32-true"
    """Precision to use, e.g., '16-mixed', 'bf16-mixed', '32-true': 
    see Lightning documentation at https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
    """

    torch_fp32_precision: Param[Optional[str]]
    """Torch precision for torch.float32 operations, see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    Automatically set depending on fabric_config.precision if not set, but can be overridden if needed (e.g., to force TF32 on Ampere GPUs while using bf16 precision for other operations)
    """
    
    # Meta - parameters - don't change output, just computing environment
    num_nodes: Meta[int] = 1
    """Number of nodes"""

    devices: Meta[str] = "auto"
    """List of devices to use"""

    strategy: Meta[str] = "auto"

    accelerator: Meta[str] = "auto"
    """The accelerator to use"""
    
    is_built = False

    def get_Fabric(self, **kwargs):
        """Builds the Fabric object and set the torch.float32 matmul precision based on the configuration. 
        This is called by the Learner before launching the training loop
        """
        if self.is_built:
            logger.warning("FabricConfiguration.get_Fabric called multiple times.")
            return None 
        
        self.is_built = True
        if self.torch_fp32_precision is None:
            #auto set torch.float32 precision based on fabric precision (if not set explicitly)
            if self.precision in ["16-mixed", "bf16-mixed"]:
                self.torch_fp32_precision = "medium"
            else:
                self.torch_fp32_precision = "high"
            logger.info(f"Setting torch.fp32 matmul precision to {self.torch_fp32_precision} based on fabric precision {self.precision}")
        
        torch.set_float32_matmul_precision(self.torch_fp32_precision)

        return L.Fabric(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            **kwargs
        )

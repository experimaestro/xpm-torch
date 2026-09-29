from abc import ABC, abstractmethod
import logging
from typing import Optional, ParamSpec
from experimaestro import field, Config, Param, Meta
import lightning.fabric.strategies as strategies
import lightning as L
import torch

logger = logging.getLogger("xpm_torch.configuration")

P = ParamSpec("P")


class Strategy(Config, strategies.Strategy):
    pass


class FabricConfigurationBase(Config, ABC):

    def get_fabric(self, **kwargs) -> L.Fabric:
        return self._get_fabric(**kwargs)

    @abstractmethod
    def _get_fabric(self, **kwargs) -> L.Fabric:
        """Builds the Fabric object based on the configuration."""
        ...


class FabricConfiguration(FabricConfigurationBase):
    """Describe the computation device

    The backend is fabric, so the complete documentation can be found on
    https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    """

    #parameters - change Learner output
    precision: Param[str] = field(default="32-true", ignore_default=True)
    """Precision to use, e.g., '16-mixed', 'bf16-mixed', '32-true':
    see Lightning documentation at https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
    """

    torch_fp32_precision: Param[Optional[str]]
    """Torch precision for torch.float32 operations, see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    Automatically set depending on fabric_config.precision if not set, but can be overridden if needed (e.g., to force TF32 on Ampere GPUs while using bf16 precision for other operations)
    """

    # Meta - parameters - don't change output, just computing environment
    num_nodes: Meta[int] = field(default=1, ignore_default=True)
    """Number of nodes"""

    devices: Meta[str] = field(default="auto", ignore_default=True)
    """Configure the devices to run on.
    See https://lightning.ai/docs/fabric/stable/api/fabric_args.html#devices for more details and options.
    Note that for multi-node training, you should specify the devices per node, e.g., devices="4" for 4 GPUs per node, not devices="16" for a total of 16 GPUs across 4 nodes.
    """

    strategy: Meta[str] = field(default="auto", ignore_default=True)
    """The strategy to use
    See https://lightning.ai/docs/fabric/stable/api/fabric_args.html#strategy for more details and options.
    """

    accelerator: Meta[str] = field(default="auto", ignore_default=True)
    """The accelerator to use
    See https://lightning.ai/docs/fabric/stable/api/fabric_args.html#accelerator for more details and options.
    """

    is_built = False


    def _get_fabric(self, **kwargs) -> L.Fabric:
        """Builds the Fabric object and set the torch.float32 matmul precision based on the configuration.
        This is called by the Learner before launching the training loop
        """
        if self.is_built:
            logger.warning("FabricConfiguration.get_Fabric called multiple times.")
            return None

        self.is_built = True
        if self.precision and "bf16" in str(self.precision).lower():
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                device_name = (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.device_count() > 0
                    else "CUDA device"
                )
                logger.warning(
                    f"bfloat16 precision ('{self.precision}') requested, but GPU ({device_name}) "
                    f"lacks native bfloat16 hardware instructions (Volta/Turing architecture, e.g. V100). "
                    f"This will cause software emulation slowdowns. Consider using '16-mixed' (FP16) or '32-true' on V100 GPUs."
                )

        if self.torch_fp32_precision is None:
            #auto set torch.float32 precision based on fabric precision (if not set explicitly)
            if self.precision in ["16-mixed", "bf16-mixed"]:
                self.torch_fp32_precision = "medium"
            else:
                self.torch_fp32_precision = "high"
            logger.info(f"Setting torch.fp32 matmul precision to '{self.torch_fp32_precision}' based on fabric precision '{self.precision}'")

        torch.set_float32_matmul_precision(self.torch_fp32_precision)

        strategy = self.strategy
        num_devices = 1
        if isinstance(self.devices, list):
            num_devices = len(self.devices)
        elif isinstance(self.devices, int):
            num_devices = self.devices
        elif isinstance(self.devices, str):
            if self.devices.isdigit():
                num_devices = int(self.devices)
            elif self.devices == "auto" and torch.cuda.is_available():
                num_devices = torch.cuda.device_count()

        if (num_devices > 1 or self.num_nodes > 1) and strategy in (None, "auto", "ddp"):
            strategy = "ddp_find_unused_parameters_true"
            logger.info(
                f"[Fabric] Multi-device training detected ({num_devices} devices). "
                f"Resolved strategy '{self.strategy}' to '{strategy}'."
            )

        if isinstance(strategy, str) and "ddp" in strategy.lower():
            if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
                torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

        fabric = L.Fabric(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=strategy,
            num_nodes=self.num_nodes,
            precision=self.precision,
            **kwargs
        )
        logging.info(f"Using Fabric with accelerator={fabric.accelerator.__class__.__name__}, devices={fabric.world_size}, strategy={fabric.strategy.__class__.__name__}")
        return fabric

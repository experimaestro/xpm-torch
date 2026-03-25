from experimaestro.experiments import configuration  # noqa: F401
from typing import List, Optional
from functools import cached_property
from xpm_torch.configuration import FabricConfiguration
from xpm_torch.schedulers import LinearWithWarmup
from xpm_torch.optim import (
    AdamW,
    Adam,
    Adafactor,
    SGD,
    ParameterOptimizer,
    RegexParameterFilter,
    get_optimizers,
)


@configuration
class TransformerOptimization:
    """Configuration for a transformer optimization"""

    scheduler: bool = True
    """Whether to use a scheduler (LinearWithWarmup)"""

    warmup_min_factor: float = 0
    """The minimum learning rate factor at the start of warmup"""

    num_warmup_steps: int = 1000
    """Number of steps for the warmup phase"""

    batch_size: int = 64
    """Training batch size"""

    max_epochs: int = 3200
    """Maximum number of training epochs"""

    steps_per_epoch: int = 32
    """Number of steps (batches) per epoch"""

    optimizer_name: str = "adam-w"
    """Name of the optimizer to use (adam-w, adam, sgd, adafactor)"""

    lr: float = 3.0e-6
    """Learning rate"""

    weight_decay: float = 1e-2
    """Weight decay for regularization"""

    eps: float = 1e-8
    """Term added to the denominator to improve numerical stability"""

    re_no_l2_regularization: List[str] = [r"\.bias$", r"\.LayerNorm\."]
    """Regular expression for layers (targets BERT parameters) that should not have L2 regularization"""

    def get_optimizer(self, regularization):
        """Returns the optimizer configuration based on the optimizer name and regularization"""
        # Set weight decay to 0 if no regularization
        weight_decay = self.weight_decay if regularization else 0

        if self.optimizer_name == "adam-w":
            return AdamW.C(
                lr=self.lr,
                weight_decay=weight_decay,
                eps=self.eps,
            )
        elif self.optimizer_name == "adam":
            return Adam.C(self.lr, weight_decay=weight_decay, eps=self.eps)
        elif self.optimizer_name == "sgd":
            return SGD.C(lr=self.lr, weight_decay=weight_decay)
        elif self.optimizer_name == "adafactor":
            return Adafactor.C(
                lr=self.lr, weight_decay=weight_decay, relative_step=self.lr is None
            )
        else:
            raise ValueError(f"Cannot handle optimizer named {self.optimizer_Name}")

    @cached_property
    def optimizer(self):
        """Returns the combined optimizer and scheduler configuration"""
        scheduler = (
            LinearWithWarmup.C(
                num_warmup_steps=self.num_warmup_steps,
                min_factor=self.warmup_min_factor,
            )
            if self.scheduler
            else None
        )
        if not self.re_no_l2_regularization:
            return get_optimizers(
                [
                    ParameterOptimizer.C(
                        scheduler=scheduler,
                        optimizer=self.get_optimizer(True),
                    ),
                ]
            )

        return get_optimizers(
            [
                ParameterOptimizer.C(
                    scheduler=scheduler,
                    optimizer=self.get_optimizer(False),
                    filter=RegexParameterFilter.C(
                        includes=self.re_no_l2_regularization
                    ),
                ),
                ParameterOptimizer.C(
                    scheduler=scheduler,
                    optimizer=self.get_optimizer(True),
                ),
            ]
        )

@configuration
class Fabric:
    """Experiment Configuration for building a Fabric Config"""
    ## Lighnting Fabric training Configuration
    # see https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.fabric.Fabric.html#lightning.fabric.fabric.Fabric
    strategy: str = "auto"
    """Distributed training strategy"""

    precision: Optional[str] = None
    """Precision to use - e.g., '16-mixed', 'bf16-mixed', etc."""

    accelerator: str = "auto"
    """ Accelerator to use """

    def get_config(self):
        return FabricConfiguration.C(
                    strategy=self.strategy,
                    precision=self.precision,
                    accelerator=self.accelerator,
                )
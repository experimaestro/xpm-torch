from abc import abstractmethod
from typing import Dict, Iterator, List, Optional
from experimaestro import Config, Param, Meta, field
import torch
import torch.nn as nn
import numpy as np

from lightning.fabric.wrappers import _FabricDataLoader


from xpm_torch import Module, Sampler
from xpm_torch.metrics import ScalarMetric
from xpm_torch.utils.logging import EasyLogger
from xpm_torch.batchers import Batcher
from xpm_torch.trainers.context import (
    TrainingHook,
    TrainerContext,
)
from torchdata.stateful_dataloader import StatefulDataLoader


class Trainer(Config, EasyLogger):
    """Generic trainer"""

    hooks: Param[List[TrainingHook]] = field(default=[], ignore_default=True)
    """Hooks for this trainer: this includes the losses, but can be adapted for
        other uses

        The specific list of hooks depends on the specific trainer
    """

    model: Param[Optional[Module]] = field(default=None, ignore_default=True)
    """If the model to optimize is different from the model passsed to Learn,
    this parameter can be used – initialization is still expected to be done at
    the learner level"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        self.random = random

        # Generic style
        if self.model is None:
            self.model = context.state.model

        # Old style (to be deprecated)
        self.ranker = self.model

        self.context = context

        for hook in self.hooks:
            self.context.add_hook(hook)

    def to(self, device):
        """Change the computing device (if this is needed)"""

        for hook in self.context.hooks(nn.Module):
            hook.to(device)

    def pre_train_setup(self, fabric):
        """Hook called after `initialize` and after the optimizer is set up, but
        BEFORE `fabric.setup(model)` wraps the model with DDP. Default is a
        no-op; subclasses (e.g. `LossTrainer`) override to drive batcher
        profiling on an unwrapped model. The timing matters: running rank-0-only
        forward+backward on a DDP-wrapped model would deadlock at the first
        gradient-bucket all_reduce.
        """
        return

    @abstractmethod
    def iter_batches(self) -> Iterator:
        """Returns a (serializable) iterator over batches"""
        ...

    @abstractmethod
    def process_batch(self, batch):
        """Process a batch of records, return the loss value that will be backpropagated"""
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict): ...

    @abstractmethod
    def state_dict(self): ...


class LossTrainer(Trainer):
    """Trainer based on a loss function

    Uses StatefulDataLoader + IterableDataset for data loading.
    """

    batcher: Meta[Batcher] = field(default_factory=Batcher.C)
    """How to batch samples together"""

    sampler: Param[Sampler]
    """The sampler to use"""

    batch_size: Param[int] = field(default=16, ignore_default=True)
    """Number of samples per batch"""

    num_workers: Param[int] = field(default=2, ignore_default=True)
    """Number of DataLoader workers"""

    dataloader: Optional[StatefulDataLoader] = None
    """StatefulDataLoader for training data"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        """Initialize the trainer, create the dataloader and initialize the loss function
        Args:
            random: Random state for initialization
            context: TrainerContext for the training process
        """
        super().initialize(random, context)

        self.sampler.initialize(random)
        self.batcher_worker = self.batcher.initialize(self.batch_size)

    def _create_dataloader(self, dataset, collate_fn):
        """Create a StatefulDataLoader from a dataset and collate function."""
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def iter_batches(self) -> Iterator:
        """Returns an iterator over batches."""
        assert self.dataloader is not None, (
            "dataloader not initialized — call _create_dataloader() first"
        )
        return iter(self.dataloader)

    def load_state_dict(self, state: Dict):
        if "dataloader" in state and self.dataloader is not None:
            if isinstance(self.dataloader, _FabricDataLoader):
                # If the dataloader is wrapped with Fabric, we need to load the state dict into the original dataloader
                self.dataloader._dataloader.load_state_dict(state["dataloader"])
            else:
                self.dataloader.load_state_dict(state["dataloader"])

    def state_dict(self):
        assert self.dataloader is not None, "dataloader not initialized"
        if isinstance(self.dataloader, _FabricDataLoader):
            # If the dataloader is wrapped with Fabric, we need to get the state dict from the original dataloader
            dataloader_state = self.dataloader._dataloader.state_dict()
        else:
            dataloader_state = self.dataloader.state_dict()

        return {"dataloader": dataloader_state}

    def pre_train_setup(self, fabric):
        """Drive batcher pre-train setup (e.g. profiling for a predictive
        batcher), then discard gradients accumulated during probing.

        Must be invoked after the optimizer is initialized and before
        `fabric.setup(model)`; see `Trainer.pre_train_setup` for the rationale.
        """
        self.batcher_worker.pre_train_setup(
            probe_fn=self.process_microbatch,
            fabric=fabric,
        )
        self.context.state.optimizer.zero_grad()

    def process_batch(self, batch: list):
        """Compute loss for a given batch of records - called by the learner.
        important: this method uses the batcher to split the batch into microbatches when needed
        """
        self.batcher_worker.process(batch, self.process_microbatch, raise_oom=True)

    def process_microbatch(self, records: list):
        """Combines a forward and backard

        This method can be implemented by specific trainers that use the gradient.
        In that case the regularizer losses should be taken into account with
        `self.add_losses`.
        """
        # Restrict losses to this context
        with self.context.losses() as losses:
            self.train_batch(records)
            nrecords = len(records)
            total_loss = torch.tensor(0.0, device=self.context.fabric.device)
            names = []

            for loss in losses:
                total_loss += loss.weight * loss.value
                names.append(loss.name)
                self.context.add_metric(
                    ScalarMetric(f"{loss.name}", float(loss.value.item()), nrecords)
                )

            # Reports the main metric
            if len(names) > 1:
                names.sort()
                self.context.add_metric(
                    ScalarMetric("+".join(names), float(total_loss.item()), nrecords)
                )
            self.context.backward(self.context.state.optimizer.scale(total_loss))

    def train_batch(self, records):
        """This method should report"""
        raise NotImplementedError

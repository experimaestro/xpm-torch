from abc import abstractmethod
from typing import Dict, Iterator, List, Optional
from experimaestro import Config, Param, field
import torch
import torch.nn as nn
import numpy as np

from datamaestro.record import Record, Item, record_type


from xpm_torch import Module, Sampler
from xpm_torch.metrics import ScalarMetric
from xpm_torch.utils.logging import EasyLogger
from xpm_torch.batchers import Batcher
from xpm_torch.trainers.context import (
    TrainingHook,
    TrainerContext,
)
from xpm_torch.utils.iter import SerializableIterator


class Trainer(Config, EasyLogger):
    """Generic trainer"""

    hooks: Param[List[TrainingHook]] = []
    """Hooks for this trainer: this includes the losses, but can be adapted for
        other uses

        The specific list of hooks depends on the specific trainer
    """

    model: Param[Optional[Module]] = None
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

        for hook in self.hooks: self.context.add_hooks(hook)

    def to(self, device):
        """Change the computing device (if this is needed)"""
        # foreach(self.context.hooks(nn.Module), lambda hook: hook.to(device))
        for hook in self.context.hooks(nn.Module):
            hook.to(device)

    @abstractmethod
    def iter_batches(self) -> Iterator:
        """Returns a (serializable) iterator over batches"""
        ...

    @abstractmethod
    def process_batch(self, batch):
        """Process a batch of records, return the loss value that will be backpropagated"""
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict):
        ...

    @abstractmethod
    def state_dict(self):
        ...


class LossTrainer(Trainer):
    """Trainer based on a loss function

    This trainer supposes that:

    - the `sampler_iter` is initialized – and is a serializable iterator over batches
    """

    batcher: Param[Batcher] = field(default_factory=Batcher.C)
    """How to batch samples together"""

    sampler: Param[Sampler]
    """The sampler to use"""

    batch_size: Param[int] = 16
    """Number of samples per batch"""

    sampler_iter: SerializableIterator
    """The iterator over batches"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)

        self.sampler.initialize(random)

        self.batcher_worker = self.batcher.initialize(self.batch_size)

    def iter_batches(self) -> SerializableIterator:
        """Returns the batchwise iterator"""
        return self.sampler_iter

    def load_state_dict(self, state: Dict):
        self.sampler_iter.load_state_dict(state["sampler"])

    def state_dict(self):
        return {"sampler": self.sampler_iter.state_dict()}

    def process_batch(self, batch: Record):
        """Compute loss for a given batch of records - called by the learner.
        important: this method uses the batcher to split the batch into microbatches when needed
        """
        self.batcher_worker.process(batch, self.process_microbatch, raise_oom=True)

    def process_microbatch(self, records: Record):
        """Combines a forward and backard

        This method can be implemented by specific trainers that use the gradient.
        In that case the regularizer losses should be taken into account with
        `self.add_losses`.
        """
        # Restrict losses to this context
        with self.context.losses() as losses:
            self.train_batch(records)
            nrecords = len(records)
            total_loss = torch.tensor(0.0)
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
            self.context.backward(
                self.context.state.optimizer.scale(total_loss)
            )

    def train_batch(self, records):
        """This method should report"""
        raise NotImplementedError

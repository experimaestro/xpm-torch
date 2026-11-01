from abc import abstractmethod
from typing import Dict, Iterator, List, Optional
import os
import logging
from experimaestro import Config, Param, Meta, field
import torch
import torch.nn as nn
import torch.distributed as dist
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

logger = logging.getLogger(__name__)


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
        local_bs = (
            context.get_local_batch_size(self.batch_size)
            if context is not None
            else self.batch_size
        )
        self.batcher_worker = self.batcher.initialize(local_bs)

    def _create_dataloader(self, dataset, collate_fn):
        """Create a StatefulDataLoader from a dataset and collate function."""
        if hasattr(dataset, "global_batch_size"):
            dataset.global_batch_size = self.batch_size

        batch_size = (
            self.context.get_local_batch_size(self.batch_size)
            if hasattr(self, "context") and self.context is not None
            else self.batch_size
        )
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
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
        if self.dataloader is None:
            return

        target_dl = (
            self.dataloader._dataloader
            if isinstance(self.dataloader, _FabricDataLoader)
            else self.dataloader
        )

        global_step = state.get("global_step")
        if (
            global_step is None
            and hasattr(self, "context")
            and self.context is not None
            and hasattr(self.context, "state")
            and self.context.state is not None
        ):
            global_step = self.context.state.steps

        if global_step is not None:
            dataset = getattr(target_dl, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_resume_step"):
                logger.info(
                    "LossTrainer: Setting dataset resume step to global_step=%d",
                    global_step,
                )
                dataset.set_resume_step(global_step)

        if dist.is_available() and dist.is_initialized():
            current_world_size = dist.get_world_size()
            rank = dist.get_rank()
        elif (
            hasattr(self, "context")
            and self.context is not None
            and self.context.fabric is not None
        ):
            current_world_size = self.context.fabric.world_size
            rank = self.context.fabric.global_rank
        else:
            rank = int(
                os.environ.get(
                    "SLURM_PROCID",
                    os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                )
            )
            current_world_size = int(
                os.environ.get(
                    "SLURM_NTASKS",
                    os.environ.get(
                        "WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")
                    ),
                )
            )

        saved_world_size = state.get("world_size")
        saved_num_workers = state.get("num_workers")

        topology_changed = False
        if saved_world_size is not None and saved_world_size != current_world_size:
            topology_changed = True
        if saved_num_workers is not None and saved_num_workers != self.num_workers:
            topology_changed = True

        if topology_changed:
            logger.info(
                "LossTrainer: Resuming with changed topology (saved world_size=%s, num_workers=%s vs "
                "current world_size=%s, num_workers=%s). "
                "Skipping StatefulDataLoader worker snapshots and resuming via dataset step offset (global_step=%s).",
                saved_world_size,
                saved_num_workers,
                current_world_size,
                self.num_workers,
                global_step,
            )
            return

        if "dataloader" in state:
            dl_state = state["dataloader"]
            if rank > 0 and isinstance(dl_state, dict):
                import copy

                dl_state = copy.deepcopy(dl_state)
                # Offset _base_seed for non-zero ranks (nested in _snapshot._main_snapshot or top-level)
                if "_snapshot" in dl_state and isinstance(dl_state["_snapshot"], dict):
                    main_snap = dl_state["_snapshot"].get("_main_snapshot", {})
                    if (
                        isinstance(main_snap, dict)
                        and "_base_seed" in main_snap
                        and main_snap["_base_seed"] is not None
                    ):
                        main_snap["_base_seed"] = (
                            main_snap["_base_seed"] + rank * 100_000
                        )
                elif "_base_seed" in dl_state and dl_state["_base_seed"] is not None:
                    dl_state["_base_seed"] = dl_state["_base_seed"] + rank * 100_000

            try:
                target_dl.load_state_dict(dl_state)
            except Exception as e:
                logger.warning(
                    "Failed to restore StatefulDataLoader state dict (e.g. worker mismatch): %s. "
                    "Relying on dataset step offset (global_step=%s) for resumption.",
                    e,
                    global_step,
                )

    def state_dict(self):
        assert self.dataloader is not None, "dataloader not initialized"
        if isinstance(self.dataloader, _FabricDataLoader):
            dataloader_state = self.dataloader._dataloader.state_dict()
        else:
            dataloader_state = self.dataloader.state_dict()

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        elif (
            hasattr(self, "context")
            and self.context is not None
            and self.context.fabric is not None
        ):
            world_size = self.context.fabric.world_size
        else:
            world_size = int(
                os.environ.get(
                    "SLURM_NTASKS",
                    os.environ.get(
                        "WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")
                    ),
                )
            )

        global_step = (
            self.context.state.steps
            if hasattr(self, "context")
            and self.context is not None
            and hasattr(self.context, "state")
            and self.context.state is not None
            else 0
        )

        return {
            "dataloader": dataloader_state,
            "global_step": global_step,
            "world_size": world_size,
            "num_workers": self.num_workers,
        }

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

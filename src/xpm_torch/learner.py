from enum import Enum
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Any, Optional, Union
from experimaestro import (
    Task,
    Config,
    Param,
    pathgenerator,
    Annotated,
    tqdm,
    Meta,
    field,
)

import lightning as L
from lightning.fabric.strategies.strategy import Strategy as l_Strategy

from xpm_torch import Random, ModuleInitMode
from xpm_torch.metrics import Metrics
from .batchers import RecoverableOOMError
from .optim import (
    Module,
    ModuleLoader,
    ParameterOptimizer,
    ScheduledOptimizer,
    OptimizationHook,
)
from xpm_torch.context import Hook, InitializationHook
from xpm_torch.utils.logging import EasyLogger
from xpm_torch.configuration import Configuration

from xpm_torch.trainers.context import (
    StepTrainingHook,
    TrainState,
    TrainerContext,
)
from xpm_torch.batchers import RecoverableOOMError
from xpm_torch.trainers import Trainer

logger = logging.getLogger(__name__)

class Strategy(Config, l_Strategy):
    """A Lightning strategy"""
    pass


class LearnerListenerStatus(Enum):
    NO_DECISION = 0
    STOP = 1
    DONT_STOP = 2

    def update(self, other: "LearnerListenerStatus") -> "LearnerListenerStatus":
        return LearnerListenerStatus(max(self.value, other.value))

class CheckpointModuleLoader(ModuleLoader):
    """Useful to load a specific checkpoint"""

    epoch: Param[Optional[int]] = None
    """The epoch of the checkpoint"""


class LearnerListener(Config):
    """Hook for learner

    Performs some operations after a learning epoch"""

    id: Meta[str]
    """Unique ID to identify the listener (ignored for signature)"""

    def initialize(self, learner: "Learner", context: TrainerContext):
        self.learner = learner
        self.context = context

    def __call__(self, state: TrainState) -> LearnerListenerStatus:
        """Process and returns whether the training process should stop"""
        return LearnerListenerStatus.NO_DECISION

    def update_metrics(self, metrics: Dict[str, float]):
        """Add metrics"""
        pass

    def init_task(self, learner: "Learner", dep):
        """Returns the initialization task that loads the associated checkpoint

        :param learner: The learner object
        :param dep: The function that adds a dependency
        """
        return None


class LearnerOutput(NamedTuple):
    """The data structure for the output of a learner. It contains a dictionary
    where the key is the name of the listener and the value is the output of
    that listener. It also allows to access the checkpoints saved during 
    the training"""

    listeners: Dict[str, Any]

    learned_model: ModuleLoader

    checkpoints: Dict[str, Any]


class Learner(Task, EasyLogger):
    """Model Learner

    The learner task is generic, and takes two main arguments: (1) the model
    defines the model (e.g. DRMM), and (2) the trainer defines how the model
    should be trained (e.g. pointwise, pairwise, etc.)

    When submitted, it returns a dictionary based on the `listeners`
    """

    # Training
    random: Param[Random]
    """The random generator"""

    trainer: Param[Trainer]
    """Specifies how to train the model"""

    model: Param[Module]
    """Defines the model to be learned. If multiple models are used, one can use
    MultipleModel.
    """

    max_epochs: Param[int] = 1000
    """Maximum number of epochs"""

    steps_per_epoch: Param[int] = 128
    """Number of steps for one epoch (after each epoch results are logged)"""

    optimizers: Param[List[ParameterOptimizer]]
    """The list of parameter optimizers"""

    listeners: Param[List[LearnerListener]]
    """Listeners are in charge of handling the validation of the model, and
    saving the relevant checkpoints"""

    checkpoint_interval: Param[int] = 1
    """Number of epochs between each checkpoint"""

    logpath: Annotated[Path, pathgenerator("runs")]
    """The path to the tensorboard logs"""

    checkpointspath: Annotated[Path, pathgenerator("checkpoints")]
    """The path to the checkpoints"""

    hooks: Param[List[Hook]] = []
    """Global learning hooks

    :class:`Initialization hooks <xpm_torch.context.InitializationHook>` are called
    before and after the initialization of the trainer and listeners.
    """
    
    # Fabric Parameters
    accelerator: Param[str] = "auto" 
    """e.g., 'gpu', 'cpu', 'tpu'"""

    devices: Param[Union[int, str]] = "auto"
    """Number of devices to use, see Lightning documentation"""

    strategy: Param[str] = "auto" # e.g., 'ddp', 'fsdp'
    """Strategy to use for distributed training, see Lightning documentation"""

    precision: Param[str] = "32-true" 
    """Precision to use, e.g., '16-mixed', 'bf16-mixed', '32-true': see Lightning documentation"""
    

    def __validate__(self):
        assert self.optimizers, "At least one optimizer should be defined"
        assert len(set(listener.id for listener in self.listeners)) == len(
            self.listeners
        ), "IDs of listeners should be unique"
        return super().__validate__()

    def task_outputs(self, dep) -> LearnerOutput:
        """Object returned when submitting the task"""
        return LearnerOutput(
            listeners={
                listener.id: listener.init_task(self, dep)
                for listener in self.listeners
            },
            learned_model=dep(
                CheckpointModuleLoader.C(
                    value=self.model,
                    path=self.last_checkpoint_path / TrainState.MODEL_PATH,
                )
            ),
            checkpoints={interval: dep(CheckpointModuleLoader.C(value=self.model, path=TrainerContext.get_checkpoint_path(self.checkpointspath, interval) / TrainState.MODEL_PATH, epoch=interval)) for interval in range(0, self.max_epochs, self.checkpoint_interval)} ,
        )

    @property
    def last_checkpoint_path(self):
        return self.checkpointspath / "last"

    def execute(self):
        # 1. Launch Fabric
        fabric = L.Fabric(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            precision=self.precision
        )
        fabric.launch()
        
        # 2. Fabric-aware execution = We pass fabric down instead of DeviceInformation
        self.fabric_execute(fabric)

    def fabric_execute(self, fabric: L.Fabric):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

        self.optimizer = ScheduledOptimizer()
        self.only_cached = False
        self.context = TrainerContext(
            self.logpath,
            self.checkpointspath,
            self.max_epochs,
            self.steps_per_epoch,
            self.trainer,
            self.model,
            self.optimizer,
            fabric=fabric,
        )

        for hook in self.hooks:
            self.context.add_hook(hook)

        # Call init hooks
        for hook in self.context.hooks(InitializationHook):
            hook.before(self.context)

        # Sets the random seed
        # WARNING - will still not be fully deterministic unless using (lot slower):
        # - torch.use_deterministic_algorithms(True) (PyTorch ≥1.8). 
        # - torch.backends.cudnn.deterministic = True
        # - torch.backends.cudnn.benchmark = False.
        # can also use fabric.seed_everything(self.random.state.randint((2**32) - 1))
        seed = self.random.state.randint((2**32) - 1)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize the scorer and trainer
        self.logger.info("model initialization")
        self.model.initialize(ModuleInitMode.DEFAULT.to_options(self.random.state))

        # Initialize the context and the listeners
        self.trainer.initialize(self.random.state, self.context)
        for listener in self.listeners:
            listener.initialize(self, self.context)

        self.logger.info("Moving model to device %s", fabric.device)
        

        num_training_steps = self.max_epochs * self.steps_per_epoch

        self.optimizer.initialize(
            self.optimizers,
            num_training_steps,
            self.model,
            use_scaler=False,
            hooks=[hook for hook in self.hooks if isinstance(hook, OptimizationHook)],
            trainer_context=self.context,
        )
        
        #wrap model and optimizers
        self.model, *self.optimizer.optimizers = fabric.setup(self.model, *self.optimizer.optimizers)

        for hook in self.context.hooks(InitializationHook):
            hook.after(self.context)

        self.logger.info("Starting to train")

        current = 0
        state = None

        with tqdm(
            total=self.max_epochs, desc=f"Training ({self.max_epochs} epochs)"
        ) as tqdm_epochs:
            for state in self.iter_train(fabric):
                # Report progress
                tqdm_epochs.update(state.epoch - current)
                current = state.epoch

                if state.epoch >= 0 and not self.only_cached:
                    message = f"epoch {state.epoch}"
                    if state.cached:
                        self.logger.debug(f"[train] [cached] {message}")
                    else:
                        self.logger.debug(f"[train] {message}")

                if state.epoch == -1:
                    continue

                if not state.cached and state.epoch % self.checkpoint_interval == 0:
                    # Save checkpoint if needed
                    self.context.save_checkpoint()
                    self.context.copy(self.last_checkpoint_path)

                # Call listeners
                decision = LearnerListenerStatus.NO_DECISION
                for listener in self.listeners:
                    # listener.__call__ returns True if we should stop
                    decision = decision.update(listener(state))

                if decision == LearnerListenerStatus.STOP:
                    self.logger.warning(
                        "stopping after epoch {epoch} ({early_stop} epochs) since "
                        "all listeners asked for it"
                    )
                    break

                # Stop if max epoch is reached
                if self.context.epoch >= self.max_epochs:
                    self.logger.warning(
                        "stopping after epoch {max_epochs} (max_epoch)".format(
                            **self.__dict__
                        )
                    )
                    break

            # End of the learning process
            if state is not None and not state.cached:
                # Set the hyper-parameters
                metrics = {}
                for listener in self.listeners:
                    listener.update_metrics(metrics)
                self.context.writer.add_hparams(getattr(self, "__tags__", {}), metrics)

    def iter_train(self, fabric: L.Fabric) -> Iterator[TrainState]:
        """Train iteration"""
        # Try to load a checkpoint

        if self.context.load_bestcheckpoint(self.max_epochs):
            yield self.context.state

        # Get an iterator over batches
        iter = self.trainer.iter_batches()

        while True:
            # Step to the next epoch
            self.context.nextepoch()

            # Train for an epoch
            with tqdm(
                leave=False,
                total=self.steps_per_epoch,
                ncols=100,
                desc=f"Train - epoch #{self.context.epoch}",
            ) as pbar:
                # Put the model into training mode (just in case)
                self.context.state.model.train()

                # Epoch: loop over batches
                metrics = Metrics()
                for b in range(self.steps_per_epoch):
                    # Get the next batch
                    batch = next(iter)
                    self.context.nextbatch()

                    while True:
                        try:
                            # Computes the gradient, takes a step and collect metrics
                            with self.context.step(metrics):
                                
                                # Call epoch hooks
                                for hook in self.context.hooks(StepTrainingHook):
                                    hook.before(self.context)

                                # Computes the gradient
                                self.trainer.process_batch(batch)

                                # Update metrics and counter
                                pbar.update(1)
                                break
                        except RecoverableOOMError:
                            logger.warning(
                                "Recoverable OOM detected"
                                " - re-running the training step"
                            )
                            continue

                    foreach(
                        self.context.hooks(StepTrainingHook),
                        lambda hook: hook.after(self.context),
                    )

                # Yields the current state (after one epoch)
                yield self.context.state

                # Report metrics over the epoch
                metrics.report(
                    self.context.state.step,
                    self.context.writer,
                    "train",
                )

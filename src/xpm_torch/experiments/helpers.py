from functools import cached_property

from experimaestro.experiments import ExperimentHelper
from xpm_torch.experiments.services import TensorboardService


class LearningExperimentHelper(ExperimentHelper):
    @cached_property
    def tensorboard_service(self) -> TensorboardService:
        """Returns a tensorboard service"""
        return self.xp.add_service(TensorboardService(self.xp, self.xp.resultspath / "runs"))


learning_experiment = LearningExperimentHelper.decorator
"""Wraps an experiment into an experiment where a model is learned

Provides:

1. Tensorboard service
"""


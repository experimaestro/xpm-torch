from functools import cached_property

from experimaestro.experiments import ExperimentHelper as ExperimentHelper
from xpm_mlboard import MonitoringService, TensorboardService


class LearningExperimentHelper(ExperimentHelper):
    """Wraps an experiment into one where a model is learned.

    Provides a monitoring service (TensorBoard by default) that aggregates the
    run directories of the submitted tasks. Override :attr:`monitoring_backend`
    (or subclass) to use a different backend.
    """

    #: Monitoring backend class to instantiate
    monitoring_backend: type[MonitoringService] = TensorboardService

    @cached_property
    def monitoring_service(self) -> MonitoringService:
        """Returns the (lazily created) monitoring service (under
        ``xp.resultspath / "runs"``; inactive outside the NORMAL run mode)"""
        return self.xp.add_service(self.monitoring_backend("runs"))

    @cached_property
    def tensorboard_service(self) -> MonitoringService:
        """Backwards-compatible alias for :attr:`monitoring_service`"""
        return self.monitoring_service


learning_experiment = LearningExperimentHelper.decorator
"""Wraps an experiment into an experiment where a model is learned

Provides:

1. A monitoring service (TensorBoard by default)
"""

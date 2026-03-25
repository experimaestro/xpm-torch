import logging

from experimaestro import (
    Config,
    Param,
    field,
)

from xpm_torch.learner import LearnerListener

logger = logging.getLogger(__name__)


class ValidationSettings(Config):
    """Settings for a validation-specific ModuleLoader.

    Attached as ``settings`` on the loader to distinguish validation
    checkpoints from other loaders with the same model and path.
    """

    listener: Param["LearnerListener"] = field(ignore_generated=True)
    """The listener (kept to change the loader identifier based
    on the learner listener configuration)"""

    key: Param[str]
    """The metric key for this validation checkpoint"""

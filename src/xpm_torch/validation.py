import logging
from experimaestro import (
    LightweightTask,
    Param,
    field,
)
from .optim import ModuleLoader

from xpm_torch.learner import LearnerListener

logger = logging.getLogger(__name__)


class ValidationModuleLoader(LightweightTask):
    """Wrapper around a ModuleLoader for validation checkpoints.

    Holds validation metadata (listener, key) and delegates loading
    to the inner loader (produced by ``Module.loader_config``).
    """

    loader: Param[ModuleLoader]
    """The actual loader (from loader_config)"""

    listener: Param["LearnerListener"] = field(ignore_generated=True)
    """The listener (kept there to change the validation loader identifier based
    on the learner listener configuration)"""

    key: Param[str]
    """The key for this listener"""

    @property
    def value(self):
        """The model config (delegates to the inner loader)."""
        return self.loader.value

    def execute(self):
        self.loader.execute()

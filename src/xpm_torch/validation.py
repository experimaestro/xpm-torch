import logging
from experimaestro import (
    Param,
    field,
)
from .optim import ModuleLoader

from xpm_torch.learner import LearnerListener

logger = logging.getLogger(__name__)


class ValidationModuleLoader(ModuleLoader):
    """Specializes the validation listener"""

    listener: Param["LearnerListener"] = field(ignore_generated=True)
    """The listener (kept there to change the validation loader identifier based
    on the learner listener configuration)"""

    key: Param[str]
    """The key for this listener"""

# TODO - need to decide how to deal with this - xpmir.letor.learner ..

# class ValidationListener(LearnerListener):
#     """Learning validation early-stopping

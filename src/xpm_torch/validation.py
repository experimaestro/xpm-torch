from collections import defaultdict
from enum import Enum
from statistics import mean
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Any, Optional, Union
from experimaestro import (
    Param,
    pathgenerator,
    Annotated,
    tqdm,
    Meta,
    field,
)
from datamaestro_text.data.ir import (
    Adhoc,
)
from xpm_torch import Module, Random
from xpm_torch.metrics import Metrics
from .batchers import RecoverableOOMError
from .optim import ModuleLoader

from xpm_torch.trainers.context import (
    TrainState,
    TrainerContext,
    ValidationHook,
)
from xpm_torch.learner import Learner, LearnerListener, LearnerListenerStatus

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

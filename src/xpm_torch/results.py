"""Base result classes for experiment outputs."""

from pathlib import Path
from typing import Dict, Optional

from experimaestro import Config, Meta, Param
from xpm_torch.module import ModuleLoader


class TrainingResults(Config):
    """Base class for experiment results that can be serialized.

    Subclass this in domain-specific libraries (e.g., xpmir's PaperResults)
    to add evaluation results and other metadata.

    Models should be :class:`~xpm_torch.module.ModuleLoader` instances
    (not wrappers like ValidationModuleLoader).
    """

    models: Param[Dict[str, ModuleLoader]]
    """ModuleLoaders keyed by identifier"""

    tb_logs: Meta[Optional[Dict[str, Path]]]
    """Tensorboard log dirs per model (metadata, not part of identity)"""

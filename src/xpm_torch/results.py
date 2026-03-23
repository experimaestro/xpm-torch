"""Base result classes for experiment outputs."""

from pathlib import Path
from typing import Dict, Optional

from experimaestro import Config, Param


class TrainingResults(Config):
    """Base class for experiment results that can be serialized.

    Subclass this in domain-specific libraries (e.g., xpmir's PaperResults)
    to add evaluation results and other metadata.
    """

    models: Param[Dict[str, Config]]
    """Model configs keyed by identifier"""

    tb_logs: Param[Optional[Dict[str, Path]]]
    """Tensorboard log dirs per model"""

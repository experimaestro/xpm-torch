"""Concrete :class:`~xpm_torch.metrics.MetricsLogger` implementations.

These adapt the backend-agnostic metrics-logging interface to a specific
visualization backend. TensorBoard is the default; additional backends (e.g.
Weights & Biases) plug in here without touching the training loop.
"""

from pathlib import Path
from typing import Mapping, Optional, TYPE_CHECKING

from xpm_torch.metrics import MetricsLogger

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


class TensorboardLogger(MetricsLogger):
    """Logs scalars to a TensorBoard event file via ``SummaryWriter``."""

    def __init__(self, logpath: Path, purge_step: Optional[int] = None):
        # Imported lazily so that importing this module does not pull in
        # TensorBoard unless a TensorBoard logger is actually instantiated.
        from torch.utils.tensorboard.writer import SummaryWriter

        self._writer = SummaryWriter(logpath, purge_step=purge_step)

    @property
    def writer(self) -> "SummaryWriter":
        """The underlying TensorBoard writer (escape hatch for raw access)."""
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: object, step: int) -> None:
        import torch

        if isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16:
            values = values.float()
        self._writer.add_histogram(tag, values, step)

    def log_hparams(
        self, hparams: Mapping[str, object], metrics: Mapping[str, float]
    ) -> None:
        self._writer.add_hparams(dict(hparams), dict(metrics))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class NullMetricsLogger(MetricsLogger):
    """Discards all metrics (used e.g. on non-zero ranks in distributed training)."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

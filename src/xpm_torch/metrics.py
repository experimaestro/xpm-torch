from typing import Dict, Mapping
import abc
import logging


class MetricsLogger(abc.ABC):
    """Backend-agnostic sink for scalar metrics.

    Concrete implementations forward metrics to a visualization backend
    (TensorBoard, Weights & Biases, ...). Decoupling metric production from any
    specific backend lets the monitoring backend be swapped without touching the
    training loop. See :mod:`xpm_torch.loggers` for implementations.
    """

    @abc.abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar ``value`` for ``tag`` at the given ``step``."""
        ...

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        """Log several scalars at once (defaults to repeated :meth:`log_scalar`)."""
        for tag, value in values.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: object, step: int) -> None:
        """Log a histogram of values (optional, no-op by default)."""
        pass

    def log_hparams(
        self, hparams: Mapping[str, object], metrics: Mapping[str, float]
    ) -> None:
        """Record hyper-parameters and their final metrics (optional, no-op by
        default). Called once at the end of training."""

    def flush(self) -> None:
        """Flush buffered metrics to storage (optional, no-op by default)."""
        pass

    def close(self) -> None:
        """Flush and release backend resources (optional, no-op by default)."""


class Metric:
    """Represents a generic metric"""

    key: str
    """The key for this metric (used for logging)"""

    count: int
    """Number of samples for this value"""

    def __init__(self, key, count):
        self.key = key
        self.count = count

    def merge(self, other: "Metric"):
        assert other.__class__ is self.__class__
        self._merge(other)

    def _merge(self, other: "Metric"):
        raise NotImplementedError(f"_merge in {self.__class__}")

    def report(self, step: int, logger: MetricsLogger, prefix: str):
        raise NotImplementedError(f"report in {self.__class__}")


class ScalarMetric(Metric):
    """Represents a scalar metric"""

    sum: float

    def __init__(self, key: str, value: float, count: int):
        super().__init__(key, count)
        self.sum = value * count

    def _merge(self, other: "ScalarMetric"):
        self.sum += other.sum
        self.count += other.count

    def compute(self):
        return self.sum / self.count

    def report(self, step: int, logger: MetricsLogger, prefix: str):
        if self.count == 0:
            logging.warning("Count is 0 when reporting metrics")
        logger.log_scalar(f"{prefix}/{self.key}", self.sum / self.count, step)


class Metrics:
    """Utility class to accumulate a set of metrics over batches
    of (potentially) different sizes"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}

    def add(self, metric: Metric):
        if metric.key in self.metrics:
            self.metrics[metric.key].merge(metric)
        else:
            self.metrics[metric.key] = metric

    def merge(self, other: "Metrics"):
        for value in other.metrics.values():
            self.add(value)

    def report(self, step, logger: MetricsLogger, prefix):
        for value in self.metrics.values():
            value.report(step, logger, prefix)

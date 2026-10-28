"""Base dataset classes for sharded, stateful iteration.

Replaces the custom SerializableIterator framework with PyTorch-native
IterableDataset + StatefulDataLoader (from torchdata).
"""

from abc import abstractmethod
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar, Any, Dict
import logging
import types
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ShardedIterableDataset(IterableDataset[T]):
    """Abstract base for datasets that support multi-GPU + multi-worker sharding.

    Supports two modes of operation:
    1. **Sequence Mode** (out-of-the-box elastic sharding):
       Pass an indexable ``source`` (HuggingFace Dataset, PyArrow table, list, array)
       or implement ``__len__`` and ``__getitem__``. ``iter_shard()`` automatically
       executes zero-skip, zero-duplicate elastic step-based sharding across arbitrary
       GPU and DataLoader worker counts.
    2. **Custom / Stream Mode**:
       Subclasses override ``iter_shard(shard_id, num_shards)`` to yield items for
       a specific shard (e.g. streaming line-delimited files or generators).

    Sharding combines two levels:
    1. **GPU-level** (``dist.get_rank()`` / ``dist.get_world_size()``)
    2. **DataLoader worker-level** (``get_worker_info()``)
    so that every worker on every GPU processes a disjoint subset of data.
    """

    def __init__(
        self,
        source: Optional[Any] = None,
        sample_builder: Optional[Callable[[Any], T]] = None,
        global_batch_size: int = 1,
    ):
        super().__init__()
        if global_batch_size <= 0:
            raise ValueError(
                f"[{type(self).__name__}] `global_batch_size` must be a positive integer, got {global_batch_size}."
            )
        self.source = source
        self.sample_builder = sample_builder
        self._global_batch_size = global_batch_size
        self.start_step: int = 0
        self._current_step: int = 0
        self._items_yielded: int = 0

    @property
    def global_batch_size(self) -> int:
        return self._global_batch_size

    @global_batch_size.setter
    def global_batch_size(self, value: int):
        if value <= 0:
            raise ValueError(
                f"[{type(self).__name__}] `global_batch_size` must be positive, got {value}."
            )
        self._global_batch_size = value

    def decompose_shard(
        self, shard_id: int, num_shards: int
    ) -> tuple[int, int, int, int]:
        """Decomposes (shard_id, num_shards) into (rank, world_size, worker_id, num_workers)."""
        if num_shards <= 0:
            raise ValueError(
                f"[{type(self).__name__}] `num_shards` must be >= 1, got {num_shards}."
            )
        if not (0 <= shard_id < num_shards):
            raise ValueError(
                f"[{type(self).__name__}] `shard_id` ({shard_id}) out of bounds for `num_shards` ({num_shards})."
            )

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        world_size = max(1, num_shards // num_workers)
        rank = shard_id // num_workers
        return rank, world_size, worker_id, num_workers

    def log_worker_init(
        self,
        shard_id: int,
        num_shards: int,
        total_samples: Optional[int] = None,
    ):
        """Logs worker initialization with rank, world_size, worker_id, global_step, and sample offset."""
        rank, world_size, worker_id, num_workers = self.decompose_shard(
            shard_id, num_shards
        )
        local_batch_size = max(1, self.global_batch_size // world_size)
        if total_samples:
            init_offset = (
                self.start_step * self.global_batch_size + rank * local_batch_size
            ) % total_samples
            init_epoch = (self.start_step * self.global_batch_size) // total_samples
            logger.info(
                "[%s] Initialized worker %d/%d on rank %d/%d (shard %d/%d): "
                "resuming at global step %d (sample index %d / %d, epoch %d, local_bs=%d)",
                type(self).__name__,
                worker_id,
                num_workers,
                rank,
                world_size,
                shard_id,
                num_shards,
                self.start_step,
                init_offset,
                total_samples,
                init_epoch,
                local_batch_size,
            )
        else:
            logger.info(
                "[%s] Initialized worker %d/%d on rank %d/%d (shard %d/%d): "
                "resuming at global step %d (local_bs=%d)",
                type(self).__name__,
                worker_id,
                num_workers,
                rank,
                world_size,
                shard_id,
                num_shards,
                self.start_step,
                local_batch_size,
            )

    def _resolve_sharding(self) -> tuple[int, int]:
        """Return ``(shard_id, num_shards)`` for the current context."""
        import os

        # GPU-level sharding
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            # Fallback to environment variables if torch.distributed is not initialized
            # (common when Fabric spawns processes without srun)
            rank = int(
                os.environ.get(
                    "SLURM_PROCID",
                    os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                )
            )
            world_size = int(
                os.environ.get(
                    "SLURM_NTASKS",
                    os.environ.get(
                        "WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")
                    ),
                )
            )
            if rank or world_size > 1:
                logger.error(
                    "torch.distributed not initialized, falling back to environment variables for sharding."
                    f"Detected rank={rank}, world_size={world_size} from environment."
                    "This may lead to incorrect sharding if running under a launcher like srun that sets SLURM_PROCID but does not initialize torch.distributed."
                )

        # DataLoader worker-level sharding
        worker_info = get_worker_info()
        if worker_info is not None:
            num_shards = world_size * worker_info.num_workers
            shard_id = rank * worker_info.num_workers + worker_info.id
        else:
            num_shards = world_size
            shard_id = rank

        return shard_id, num_shards

    def __iter__(self) -> Iterator[T]:
        shard_id, num_shards = self._resolve_sharding()
        self._items_yielded = 0
        for item in self.iter_shard(shard_id, num_shards):
            self._items_yielded += 1
            yield item

    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator[T]:
        """Yield items for the given shard - Default implementation uses elastic step-based sharding if ``source`` is provided.

        If a sequence ``source`` was provided (or this object implements ``__len__`` and ``__getitem__``),
        runs the elastic step-based sharding loop. Otherwise, raises ``NotImplementedError``.
        """
        src = self.source if self.source is not None else self
        is_sequence = hasattr(src, "__len__") and hasattr(src, "__getitem__")

        if not is_sequence and self.source is not None:
            raise TypeError(
                f"[{type(self).__name__}] `source` must be an indexable sequence supporting `__len__` and `__getitem__` "
                f"(got {type(self.source).__name__}). For streaming data sources, override `iter_shard(shard_id, num_shards)`."
            )

        if is_sequence:
            data_len = len(src)
            if data_len == 0:
                raise ValueError(
                    f"[{type(self).__name__}] Cannot shard an empty dataset (len=0)."
                )

            rank, world_size, worker_id, num_workers = self.decompose_shard(
                shard_id, num_shards
            )
            local_batch_size = max(1, self.global_batch_size // world_size)
            self.log_worker_init(shard_id, num_shards, data_len)

            step = self.start_step + worker_id
            while True:
                self._current_step = step
                step_offset = step * self.global_batch_size + rank * local_batch_size
                for i in range(local_batch_size):
                    sample_idx = (step_offset + i) % data_len
                    item = src[sample_idx]
                    yield self.sample_builder(item) if self.sample_builder else item
                step += num_workers
        else:
            raise NotImplementedError(
                f"[{type(self).__name__}] does not implement `iter_shard(shard_id, num_shards)` "
                "and did not provide an indexable `source`. You must either provide `source` to "
                "ShardedIterableDataset(...) or override `iter_shard` in your subclass."
            )

    def set_resume_step(self, step: int):
        """Set the global step to resume from."""
        self.start_step = step
        self._current_step = step

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        if (
            self.source is None
            and self._items_yielded > 0
            and self._current_step == self.start_step
        ):
            logger.warning(
                "[%s] Yielded %d items, but `self._current_step` was never updated from start_step=%d. "
                "If this subclass implements custom `iter_shard`, ensure it updates `self._current_step` "
                "so that checkpoint state can be restored accurately.",
                type(self).__name__,
                self._items_yielded,
                self.start_step,
            )
        return {"step": self._current_step}

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        if not isinstance(state, dict):
            raise TypeError(
                f"[{type(self).__name__}] Expected state to be a dict, got {type(state).__name__}"
            )
        if "step" not in state:
            logger.warning(
                "[%s] `step` key missing in loaded state dict (keys: %s). "
                "Stateful resumption cannot position the dataloader step.",
                type(self).__name__,
                list(state.keys()),
            )
        else:
            self.start_step = state["step"]
            self._current_step = state["step"]


class FileShardedDataset(ShardedIterableDataset[T]):
    """Shards a data source by splitting a range ``[0, data_size())`` across shards.

    Supports checkpoint/resume via ``position()`` → ``state_dict()`` →
    ``load_state_dict()`` → resume from ``_resume_position``.
    """

    _resume_position: Optional[int] = None

    @abstractmethod
    def data_size(self) -> int:
        """Total size of the data (e.g. file size in bytes, or number of items)."""
        ...

    @abstractmethod
    def iter_from(self, start: int, end: int) -> Iterator[T]:
        """Iterate over items in the range ``[start, end)``."""
        ...

    @abstractmethod
    def position(self) -> int:
        """Current position (matching the ``start`` semantics of ``iter_from``)."""
        ...

    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator[T]:
        size = self.data_size()
        start = size * shard_id // num_shards
        end = size * (shard_id + 1) // num_shards

        if self._resume_position is not None and self._resume_position > start:
            start = self._resume_position
            self._resume_position = None

        if start >= end:
            return

        yield from self.iter_from(start, end)

    def state_dict(self) -> Dict[str, Any]:
        return {"position": self.position()}

    def load_state_dict(self, state: Dict[str, Any]):
        self._resume_position = state.get("position")


class LineFileDataset(FileShardedDataset[T]):
    """Shards a line-delimited file by byte offset.

    Each line is parsed by ``parse_fn``.  When a shard starts mid-line,
    the partial first line is skipped (boundary recovery).
    """

    def __init__(self, path: Path, parse_fn: Callable[[str], T]):
        super().__init__()
        self.path = path
        self.parse_fn = parse_fn
        self._position: int = 0
        self._file_size: Optional[int] = None

    def data_size(self) -> int:
        if self._file_size is None:
            self._file_size = self.path.stat().st_size
        return self._file_size

    def position(self) -> int:
        return self._position

    def iter_from(self, start: int, end: int) -> Iterator[T]:
        with self.path.open("rb") as f:
            f.seek(start)

            # Boundary recovery: skip partial first line if not at start of file
            if start > 0:
                f.readline()  # discard remainder of partial line

            while f.tell() < end:
                self._position = f.tell()
                line = f.readline()
                if not line:
                    break
                yield self.parse_fn(line.decode("utf-8").rstrip("\n"))

            self._position = f.tell()


class QueryGroupedFileDataset(FileShardedDataset[T]):
    """Shards a query-grouped file (e.g. TREC run format) by byte offset.

    Groups consecutive lines by query ID.  When a shard boundary lands
    mid-query-group, the partial group is skipped (boundary recovery at
    query-group level).

    ``parse_line_fn`` extracts ``(query_key, parsed_line_data)`` from each line.
    ``build_group_fn`` builds a single record from ``(query_key, [line_data, ...])``.
    """

    def __init__(
        self,
        path: Path,
        parse_line_fn: Callable[[str], tuple[str, Any]],
        build_group_fn: Callable[[str, list], T],
        top_k: int = 0,
    ):
        super().__init__()
        self.path = path
        self.parse_line_fn = parse_line_fn
        self.build_group_fn = build_group_fn
        self.top_k = top_k
        self._position: int = 0
        self._file_size: Optional[int] = None

    def data_size(self) -> int:
        if self._file_size is None:
            self._file_size = self.path.stat().st_size
        return self._file_size

    def position(self) -> int:
        return self._position

    def iter_from(self, start: int, end: int) -> Iterator[T]:
        with self.path.open("rb") as f:
            f.seek(start)

            # Boundary recovery: skip to next query boundary
            if start > 0:
                # Skip partial line
                f.readline()
                # Read ahead to find query key at this position
                peek_pos = f.tell()
                peek_line = f.readline()
                if not peek_line:
                    return
                first_key, _ = self.parse_line_fn(
                    peek_line.decode("utf-8").rstrip("\n")
                )
                f.seek(peek_pos)

                # Skip remaining lines of this query group (it belongs to previous shard)
                while f.tell() < end:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        return
                    key, _ = self.parse_line_fn(line.decode("utf-8").rstrip("\n"))
                    if key != first_key:
                        # Found start of new group — seek back and start here
                        f.seek(pos)
                        break
                else:
                    return

            # Now iterate query groups
            current_key = None
            current_lines = []
            self._position = f.tell()

            while f.tell() < end or (current_key is not None and current_lines):
                line_pos = f.tell()
                raw = f.readline()
                if not raw:
                    # EOF: emit last group
                    if current_key is not None and current_lines:
                        yield self.build_group_fn(current_key, current_lines)
                    self._position = line_pos
                    return

                text = raw.decode("utf-8").rstrip("\n")
                if not text:
                    continue

                key, data = self.parse_line_fn(text)

                if current_key is None:
                    current_key = key

                # Query changed or reached top_k: emit group
                if key != current_key or (
                    self.top_k and len(current_lines) >= self.top_k
                ):
                    yield self.build_group_fn(current_key, current_lines)
                    self._position = line_pos

                    # If we've passed the shard end, stop after emitting
                    if line_pos >= end:
                        # But we need to finish reading this new query's group
                        # if we started it — actually we haven't added anything yet
                        return

                    current_key = key
                    current_lines = []

                current_lines.append(data)

            # Emit final group
            if current_key is not None and current_lines:
                yield self.build_group_fn(current_key, current_lines)
            self._position = f.tell()


class IndexedDataset(ShardedIterableDataset[T]):
    """Wraps a random-accessible sequence (list, array) as an elastic sharded dataset."""

    def __init__(
        self,
        source: Any,
        sample_builder: Optional[Callable[[Any], T]] = None,
        global_batch_size: int = 1,
    ):
        super().__init__(
            source=source,
            sample_builder=sample_builder,
            global_batch_size=global_batch_size,
        )

    def data_size(self) -> int:
        return len(self.source)


class TransformDataset(ShardedIterableDataset[T]):
    """Wraps another ShardedIterableDataset and applies a per-record transform.
    If transform returns None, skip the item.
    """

    def __init__(
        self,
        inner: ShardedIterableDataset,
        transform: Callable,
    ):
        super().__init__()
        self.inner = inner
        self.transform = transform

    @property
    def global_batch_size(self) -> int:
        return getattr(self.inner, "global_batch_size", self._global_batch_size)

    @global_batch_size.setter
    def global_batch_size(self, value: int):
        self._global_batch_size = value
        if hasattr(self.inner, "global_batch_size"):
            self.inner.global_batch_size = value

    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator[T]:
        for item in self.inner.iter_shard(shard_id, num_shards):
            transformed = self.transform(item)

            if transformed is None:
                continue

            # Check if the result is iterable (like your generator)
            # but not a single sample object
            if isinstance(transformed, (list, types.GeneratorType)):
                for sub_item in transformed:
                    if sub_item is not None:
                        yield sub_item
            else:
                yield transformed

    def set_resume_step(self, step: int):
        super().set_resume_step(step)
        self.inner.set_resume_step(step)

    def state_dict(self) -> Dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Dict[str, Any]):
        self.inner.load_state_dict(state)


class InfiniteDataset(ShardedIterableDataset[T]):
    """Wraps a ShardedIterableDataset to loop infinitely.

    Each pass through the underlying dataset constitutes one "epoch".
    The iterator never raises StopIteration — it restarts from the beginning
    of its shard when the underlying data is exhausted.
    """

    def __init__(self, inner: ShardedIterableDataset[T]):
        super().__init__()
        self.inner = inner

    @property
    def global_batch_size(self) -> int:
        return getattr(self.inner, "global_batch_size", self._global_batch_size)

    @global_batch_size.setter
    def global_batch_size(self, value: int):
        self._global_batch_size = value
        if hasattr(self.inner, "global_batch_size"):
            self.inner.global_batch_size = value

    def set_resume_step(self, step: int):
        super().set_resume_step(step)
        self.inner.set_resume_step(step)

    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator[T]:
        while True:
            yielded = False
            for item in self.inner.iter_shard(shard_id, num_shards):
                yielded = True
                yield item
            if not yielded:
                return  # Empty shard, don't loop forever

    def state_dict(self) -> Dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Dict[str, Any]):
        self.inner.load_state_dict(state)


ElasticShardedDataset = ShardedIterableDataset

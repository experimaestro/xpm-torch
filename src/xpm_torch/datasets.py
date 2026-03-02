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

    Subclasses implement ``iter_shard(shard_id, num_shards)`` to yield items for
    a specific shard.  Sharding combines two levels:

    1. **GPU-level** (``dist.get_rank()`` / ``dist.get_world_size()``)
    2. **DataLoader worker-level** (``get_worker_info()``)

    so that every worker on every GPU processes a disjoint subset of data.
    """

    def _resolve_sharding(self) -> tuple[int, int]:
        """Return ``(shard_id, num_shards)`` for the current context."""
        # GPU-level sharding
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

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
        return self.iter_shard(shard_id, num_shards)

    @abstractmethod
    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator[T]:
        """Yield items for the given shard."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]):
        pass


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


class IndexedDataset(FileShardedDataset[T]):
    """Wraps a random-accessible sequence (list, array) as a sharded dataset."""

    def __init__(self, source: Any):
        super().__init__()
        self.source = source
        self._position: int = 0

    def data_size(self) -> int:
        return len(self.source)

    def position(self) -> int:
        return self._position

    def iter_from(self, start: int, end: int) -> Iterator[T]:
        for i in range(start, end):
            self._position = i
            yield self.source[i]
        self._position = end


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

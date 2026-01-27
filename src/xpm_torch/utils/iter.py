import numpy as np
import atexit
from abc import ABC, abstractmethod
from queue import Full, Empty
import torch.multiprocessing as mp
from typing import (
    Generic,
    Callable,
    Dict,
    Tuple,
    List,
    Iterable,
    Iterator,
    Protocol,
    TypeVar,
    Any,
    TypedDict,
)
import logging

logger = logging.getLogger(__name__)
# --- Utility classes

State = TypeVar("State")
T = TypeVar("T")


class iterable_of(Generic[T]):
    def __init__(self, factory: Callable[[], Iterator[T]]):
        self.factory = factory

    def __iter__(self):
        return self.factory()


class SerializableIterator(Iterator[T], Generic[T, State]):
    """An iterator that can be serialized through state dictionaries.

    This is used when saving the sampler state
    """

    @abstractmethod
    def state_dict(self) -> State:
        ...

    @abstractmethod
    def load_state_dict(self, state: State):
        ...


class SerializableIteratorAdapter(SerializableIterator[T, State], Generic[T, U, State]):
    """Adapts a serializable iterator with a transformation function based on
    the iterator"""

    def __init__(
        self,
        main: SerializableIterator[T, State],
        generator: Callable[[SerializableIterator[T, State]], Iterator[U]],
    ):
        self.generator = generator
        self.main = main
        self.iter = generator(main)

    def load_state_dict(self, state):
        self.main.load_state_dict(state)
        self.iter = self.generator(self.main)

    def state_dict(self):
        return self.main.state_dict()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)


class BatchIteratorAdapter(SerializableIterator[List[T], State]):
    """Adapts a serializable iterator into a batchwise serializable iterator"""

    def __init__(self, iterator: SerializableIterator[T, State], size: int):
        self.iterator = iterator
        self.size = size

    def state_dict(self):
        return self.iterator.state_dict()

    def load_state_dict(self, state):
        self.iterator.load_state_dict(state)

    def __iter__(self):
        return self

    def __next__(self) -> List[T]:
        batch = []
        for _, record in zip(range(self.size), self.iterator):
            batch.append(record)
        return batch

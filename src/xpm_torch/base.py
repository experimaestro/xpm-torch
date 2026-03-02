from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Sequence, Iterator, Iterable
import numpy as np
from functools import cached_property
from experimaestro import Config, Param
from xpm_torch.utils.logging import EasyLogger

T = TypeVar("T")


class Random(Config):
    """Random configuration"""

    seed: Param[int] = 0
    """The seed to use so the random process is deterministic"""

    @cached_property
    def state(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def __getstate__(self):
        return {"seed": self.seed}


class Sampler(Config, EasyLogger):
    """Abstract data sampler"""

    def initialize(self, random: Optional[np.random.RandomState]):
        self.random = random or np.random.RandomState(random.randint(0, 2**31))


class SampleIterator(Config, Iterable[T], ABC):
    """Generic class to iterate over items or batch of items"""

    @abstractmethod
    def __iter__() -> Iterator[T]:
        pass

    def __batch_iter__(self, batch_size: int) -> Iterator[Sequence[T]]:
        """Batch iterations"""
        iterator = self.__iter__()
        data = []
        try:
            while True:
                data.append(next(iterator))
                if len(data) == batch_size:
                    yield data
        except StopIteration:
            pass

        if data:
            yield data

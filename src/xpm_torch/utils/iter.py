"""Multiprocess iterator utilities.

The SerializableIterator hierarchy has been replaced by
``xpm_torch.datasets`` (ShardedIterableDataset + StatefulDataLoader).
Only the multiprocess iterator is kept here for non-training use cases
(e.g. sparse index building).
"""

import atexit
from queue import Full, Empty
import torch.multiprocessing as mp
from typing import (
    Generic,
    Callable,
    Iterator,
    TypeVar,
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class iterable_of(Generic[T]):
    def __init__(self, factory: Callable[[], Iterator[T]]):
        self.factory = factory

    def __iter__(self):
        return self.factory()


class StopIterationClass:
    pass


STOP_ITERATION = StopIterationClass()


def mp_iterate(iterator, queue: mp.Queue, event: mp.Event):
    try:
        while not event.is_set():
            value = next(iterator)

            while True:
                try:
                    queue.put(value, timeout=1)
                    break
                except Full:
                    if event.is_set():
                        logger.warning("Stopping as requested by the main process")
                        queue.close()
                        break

    except StopIteration:
        logger.info("Signaling that the iterator has finished")
        queue.put(STOP_ITERATION)
    except Exception as e:
        logger.exception("Exception while iterating")
        queue.put(e)

    logger.info("End of multi-process iterator")
    queue.close()


class QueueBasedMultiprocessIterator(Iterator[T]):
    """This Queue-based iterator can be pickled when a new process is spawn"""

    def __init__(self, queue: "mp.Queue[T]", stop_process: mp.Event):
        self.queue = queue
        self.stop_process = stop_process
        self.stop_iteration = mp.Event()

    def __next__(self):
        # Get the next element
        while True:
            try:
                element = self.queue.get(timeout=1)
                break
            except Empty:
                if self.stop_iteration.is_set():
                    self.stop_process.set()
                    raise StopIteration()

        # Last element
        if isinstance(element, StopIterationClass):
            # Just in case
            self.stop_process.set()
            self.stop_iteration.set()
            raise StopIteration()

        # An exception occurred
        elif isinstance(element, Exception):
            self.stop_iteration.set()
            self.stop_process.set()
            raise RuntimeError("Error in iterator process") from element

        return element


class MultiprocessIterator(Iterator[T]):
    def __init__(self, iterator: Iterator[T], maxsize=100):
        self.process = None
        self.maxsize = maxsize
        self.iterator = iterator
        self.stop_process = mp.Event()
        self.mp_iterator = None

    def start(self):
        """Start the iterator process"""
        if self.process is None:
            self.queue = mp.Queue(self.maxsize)
            self.process = mp.Process(
                target=mp_iterate,
                args=(self.iterator, self.queue, self.stop_process),
                daemon=True,
            )

            # Start the process
            self.process.start()
            self.mp_iterator = QueueBasedMultiprocessIterator(
                self.queue, self.stop_process
            )

            atexit.register(self.close)
        return self

    def close(self):
        if self.mp_iterator:
            atexit.unregister(self.close)
            self.stop_process.set()
            self.mp_iterator = None
            logging.info("Signaled the mp_iterator to quit")

    def detach(self):
        """Produces an iterator only based on the multiprocess queue (useful
        when using torch mp.spawn)"""
        self.start()
        return self.mp_iterator

    def __next__(self):
        # Start a process if needed
        self.start()
        try:
            return next(self.mp_iterator)
        except StopIteration:
            atexit.unregister(self.close)

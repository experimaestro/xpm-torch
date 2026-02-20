"""Collate functions for use with StatefulDataLoader.

Each collate function takes a list of individual records and produces
a batched structure suitable for the corresponding trainer.
"""

from typing import List

import torch
from xpmir.letor.records import (
    PairwiseRecord,
    PairwiseRecords,
    ProductRecords,
)


def pairwise_collate(records: List[PairwiseRecord]) -> PairwiseRecords:
    """Collate individual PairwiseRecords into a PairwiseRecords batch."""
    batch = PairwiseRecords()
    for record in records:
        batch.add(record)
    return batch


def batchwise_collate(records: List[PairwiseRecord]) -> ProductRecords:
    """Collate PairwiseRecords into a ProductRecords batch with in-batch negatives.

    Builds a relevance matrix where the diagonal (positive docs) = 1
    and off-diagonal (other queries' negatives) = 0.
    """
    batch_size = len(records)
    relevances = torch.cat(
        (torch.eye(batch_size), torch.zeros(batch_size, batch_size)), 1
    )

    batch = ProductRecords()
    positives = []
    negatives = []
    for record in records:
        batch.add_topics(record.query)
        positives.append(record.positive)
        negatives.append(record.negative)
    batch.add_documents(*positives)
    batch.add_documents(*negatives)
    batch.set_relevances(relevances)
    return batch


def distillation_pairwise_collate(records: list) -> list:
    """Identity collate for pairwise distillation samples."""
    return records


def distillation_listwise_collate(records: list) -> list:
    """Identity collate for listwise distillation samples."""
    return records

"""Collate functions for use with StatefulDataLoader.

Each collate function takes a list of individual records and produces
a batched structure suitable for the corresponding trainer.
"""

from typing import Callable, List

import torch
from xpmir.letor.records import (
    PairwiseRecord,
    PairwiseRecords,
    ProductRecords,
)
from xpmir.letor.distillation.samplers import (
    PairwiseDistillationSample,
    ListwiseDistillationSample,
)
from xpmir.rankers import ScoredDocument


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
    Absorbs the logic previously in PairwiseInBatchNegativesSampler.batchwise_iter().
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
    """Identity collate for pairwise distillation samples.

    Records are already PairwiseDistillationSample namedtuples.
    """
    return records


def distillation_listwise_collate(records: list) -> list:
    """Identity collate for listwise distillation samples.

    Records are already ListwiseDistillationSample namedtuples.
    """
    return records


class HydratingCollate:
    """Wraps a base collate function with document/query hydration.

    Datasets yield ID-only records; this collate fetches text in batch
    before passing to the base collate. This is more efficient than
    per-record hydration because document stores can optimize batch access.
    """

    def __init__(
        self,
        base_collate: Callable,
        adapter,
    ):
        """
        Args:
            base_collate: The underlying collate function.
            adapter: A SampleTransform (e.g. SampleHydrator) providing
                transform_topics() and transform_documents().
        """
        self.base_collate = base_collate
        self.adapter = adapter

    def __call__(self, records):
        if not records:
            return self.base_collate(records)

        # For PairwiseRecord: hydrate topics and documents in batch
        if isinstance(records[0], PairwiseRecord):
            return self._hydrate_pairwise(records)
        elif isinstance(records[0], PairwiseDistillationSample):
            return self._hydrate_distillation_pairwise(records)
        elif isinstance(records[0], ListwiseDistillationSample):
            return self._hydrate_distillation_listwise(records)

        # For other types, apply adapter transform where possible
        return self.base_collate(records)

    def _hydrate_pairwise(self, records: List[PairwiseRecord]) -> PairwiseRecords:
        """Hydrate pairwise records in batch, then collate."""
        # Collect all topics and documents
        topics = [r.query for r in records]
        docs = []
        for r in records:
            docs.append(r.positive)
            docs.append(r.negative)

        # Batch transform
        transformed_topics = self.adapter.transform_topics(topics) or topics
        transformed_docs = self.adapter.transform_documents(docs) or docs

        # Rebuild records with hydrated data
        hydrated = []
        for i, r in enumerate(records):
            hydrated.append(
                PairwiseRecord(
                    transformed_topics[i],
                    transformed_docs[2 * i],
                    transformed_docs[2 * i + 1],
                )
            )

        return self.base_collate(hydrated)

    def _hydrate_distillation_pairwise(
        self, records: List[PairwiseDistillationSample]
    ) -> List[PairwiseDistillationSample]:
        """Hydrate distillation pairwise records in batch, then collate."""
        topics = [r.query for r in records]
        docs = [d for r in records for d in r.documents]

        transformed_topics = self.adapter.transform_topics(topics) or topics
        transformed_docs = self.adapter.transform_documents(docs) or docs

        hydrated = []
        for i, r in enumerate(records):
            hydrated_docs = (
                ScoredDocument(transformed_docs[2 * i], r.documents[0].score),
                ScoredDocument(transformed_docs[2 * i + 1], r.documents[1].score),
            )
            hydrated.append(
                PairwiseDistillationSample(transformed_topics[i], hydrated_docs)
            )
        return self.base_collate(hydrated)

    def _hydrate_distillation_listwise(
        self, records: List[ListwiseDistillationSample]
    ) -> List[ListwiseDistillationSample]:
        """Hydrate distillation listwise records in batch, then collate."""
        topics = [r.query for r in records]
        docs = [d for r in records for d in r.documents]

        transformed_topics = self.adapter.transform_topics(topics) or topics
        transformed_docs = self.adapter.transform_documents(docs) or docs

        hydrated = []
        doc_offset = 0
        for i, r in enumerate(records):
            num_docs = len(r.documents)
            record_docs = transformed_docs[doc_offset : doc_offset + num_docs]
            hydrated_docs = [
                ScoredDocument(doc, r.documents[j].score)
                for j, doc in enumerate(record_docs)
            ]
            hydrated.append(
                ListwiseDistillationSample(transformed_topics[i], hydrated_docs)
            )
            doc_offset += num_docs
        return self.base_collate(hydrated)

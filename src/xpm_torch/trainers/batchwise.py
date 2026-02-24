import sys
import torch
import numpy as np
from typing import List
from experimaestro import Param, initializer

from xpm_torch.losses.batchwise import BatchwiseLoss
from xpm_torch.trainers import TrainerContext, LossTrainer

from xpmir.letor.samplers import BatchwiseSampler
from xpmir.letor.records import BatchwiseRecords
from xpmir.letor.records import (
    PairwiseRecord,
    PairwiseRecords,
    ProductRecords,
)

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



class BatchwiseTrainer(LossTrainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    lossfn: Param[BatchwiseLoss]
    """A batchwise loss function"""

    @initializer
    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(context)

        dataset = self.sampler.as_dataset()
        self._create_dataloader(dataset, batchwise_collate)

    def train_batch(self, batch: BatchwiseRecords):
        # Get the next batch and compute the scores for each query/document
        # Get the scores
        rel_scores = self.ranker(batch, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        batch_scores = rel_scores.reshape(*batch.relevances.shape)
        self.lossfn.process(
            batch_scores, batch.relevances.to(batch_scores.device), self.context
        )

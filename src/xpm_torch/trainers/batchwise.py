import sys
import torch
from experimaestro import Param, initializer
from xpm_torch.losses.batchwise import BatchwiseLoss
from xpmir.letor.samplers import BatchwiseSampler
from xpmir.letor.records import BatchwiseRecords
from xpm_torch.trainers import TrainerContext, LossTrainer
import numpy as np


class BatchwiseTrainer(LossTrainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    sampler: Param[BatchwiseSampler]
    """A batch-wise sampler"""

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

        from xpm_torch.collate import batchwise_collate

        dataset = self.sampler.as_dataset()
        collate_fn = self.sampler.get_collate_fn(batchwise_collate)
        self._create_dataloader(dataset, collate_fn)

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

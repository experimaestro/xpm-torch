import sys
import torch
from torch.functional import Tensor
from experimaestro import Param
from xpm_torch.losses.pairwise import PairwiseLoss
from xpm_torch.metrics import ScalarMetric
from xpmir.letor.records import (
    PairwiseRecords,
)
from xpmir.utils.utils import foreach
import numpy as np
from xpmir.letor.samplers import PairwiseSampler
from xpm_torch.trainers import TrainerContext, LossTrainer
from xpm_torch.collate import pairwise_collate


class PairwiseTrainer(LossTrainer):
    """Pairwise trainer uses samples of the form (query, positive, negative)"""

    lossfn: Param[PairwiseLoss]
    """The loss function"""

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.ranker)
        foreach(context.hooks(PairwiseLoss), lambda loss: loss.initialize(self.ranker))
        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset()
        self._create_dataloader(dataset, pairwise_collate)

    def train_batch(self, records: PairwiseRecords):
        # Get the next batch and compute the scores for each query/document
        rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        self.lossfn.process(pairwise_scores, self.context)

        self.context.add_metric(
            ScalarMetric(
                "accuracy", float(self.acc(pairwise_scores).item()), len(rel_scores)
            )
        )

    def acc(self, scores_by_record) -> Tensor:
        with torch.no_grad():
            return (
                scores_by_record[:, 0] > scores_by_record[:, 1]
            ).sum().float() / len(scores_by_record)

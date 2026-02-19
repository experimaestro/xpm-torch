
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from experimaestro import Config, Param

from xpmir.rankers import ScorerOutputType
from xpmir.utils.utils import EasyLogger
from xpm_torch.losses import Loss, bce_with_logits_loss

from xpm_torch.trainers import TrainerContext

class PairwiseLoss(Config, nn.Module):
    """Base class for any pairwise loss"""

    NAME = "?"

    weight: Param[float] = 1.0
    """The weight :math:`w` with which the loss is multiplied (useful when
    combining with other ones)"""

    def initialize(self, ranker):
        pass

    def process(self, scores: Tensor, context: TrainerContext):
        value = self.compute(scores, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))

    def compute(self, scores: Tensor, info: TrainerContext) -> Tensor:

        """Computes the loss

        :param scores: A (batch x 2) tensor (positive/negative)
        :param info: the trainer context
        :return: a torch scalar
        """
        raise NotImplementedError()
    
### Child Losses

class CrossEntropyLoss(PairwiseLoss):
    r"""Cross-Entropy Loss

    Computes the cross-entropy loss

    Classification loss (relevant vs non-relevant) where the logit
    is equal to the difference between the relevant and the non relevant
    document (or equivalently, softmax then mean log probability of relevant documents)
    Reference: C. Burges et al., “Learning to rank using gradient descent,” 2005.

    *warning*: this loss assumes the score returned by the scorer is a logit

    .. math::

        \frac{w}{N} \sum_{(s^+,s-)} \log \frac{\exp(s^+)}{\exp(s^+)+\exp(s^-)}
    """
    NAME = "cross-entropy"

    def compute(self, rel_scores_by_record, info: TrainerContext):
        target = (
            torch.zeros(rel_scores_by_record.shape[0])
            .long()
            .to(rel_scores_by_record.device)
        )
        return F.cross_entropy(rel_scores_by_record, target, reduction="mean")


class HingeLoss(PairwiseLoss):
    r"""Hinge (or max-margin) loss

    .. math::

       \frac{w}{N} \sum_{(s^+,s-)} \max(0, m - (s^+ - s^-))

    """

    NAME = "hinge"

    margin: Param[float] = 1.0
    """The margin for the Hinge loss"""

    def compute(self, rel_scores_by_record, info: TrainerContext):
        return F.relu(
            self.margin - rel_scores_by_record[:, 0] + rel_scores_by_record[:, 1]
        ).mean()

class PointwiseCrossEntropyLoss(PairwiseLoss, EasyLogger):
    r"""Point-wise cross-entropy loss

    This is a point-wise loss adapted as a pairwise one.

    This loss adapts to the ranker output type:

    - If real, uses a BCELossWithLogits (sigmoid transformation)
    - If probability, uses the BCELoss
    - If log probability, uses a BCEWithLogLoss

    .. math::

        \frac{w}{2N} \sum_{(s^+,s-)} \log \frac{\exp(s^+)}{\exp(s^+)+\exp(s^-)}
        + \log \frac{\exp(s^-)}{\exp(s^+)+\exp(s^-)}

    """

    NAME = "pointwise-cross-entropy"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.rankerOutputType = ranker.outputType
        if ranker.outputType == ScorerOutputType.REAL:
            self.logger.info("Ranker outputs logits: using BCEWithLogitsLoss")
            self.loss = nn.BCEWithLogitsLoss()
        elif ranker.outputType == ScorerOutputType.PROBABILITY:
            self.logger.info("Ranker outputs probabilities: using BCELoss")
            self.loss = nn.BCELoss()
        elif ranker.outputType == ScorerOutputType.LOG_PROBABILITY:
            self.logger.info("Ranker outputs probabilities: using BCEWithLogLoss")
            self.loss = bce_with_logits_loss
        else:
            raise Exception("Not implemented")

    def compute(self, rel_scores_by_record, info: TrainerContext):
        device = rel_scores_by_record.device
        dim = rel_scores_by_record.shape[0]
        target = torch.cat(
            (torch.ones(dim, device=device), torch.zeros(dim, device=device))
        )
        return self.loss(rel_scores_by_record.T.flatten(), target)
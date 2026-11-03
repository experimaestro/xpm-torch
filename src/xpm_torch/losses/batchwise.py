import torch
import torch.nn.functional as F
from experimaestro import field, Config, Param
from xpm_torch.losses import Loss, ModuleOutputType
from xpm_torch.trainers import TrainerContext


class BatchwiseLoss(Config):
    NAME = "?"

    weight: Param[float] = field(default=1.0, ignore_default=True)
    """The weight of this loss"""

    def initialize(self, context: TrainerContext):
        pass

    def process(
        self, scores: torch.Tensor, relevances: torch.Tensor, context: TrainerContext
    ):
        value = self.compute(scores, relevances, context)
        context.add_loss(Loss(f"batch-{self.NAME}", value, self.weight))

    def compute(
        self, scores: torch.Tensor, relevances: torch.Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

        - scores: A (queries x documents) tensor
        - relevances: A (queries x documents) tensor
        """
        raise NotImplementedError()


class CrossEntropyLoss(BatchwiseLoss):
    NAME = "bce"

    def compute(self, scores, relevances, context):
        return F.binary_cross_entropy(scores, relevances, reduction="mean")


class SoftmaxCrossEntropy(BatchwiseLoss):
    NAME = "infonce"

    """Computes the probability of relevant documents for a given query"""

    temperature: Param[float] = field(default=1.0, ignore_default=True)
    """Temperature scaling factor applied to logits before softmax"""

    def __validate__(self):
        if self.temperature <= 0:
            raise ValueError(
                f"SoftmaxCrossEntropy temperature must be > 0, got {self.temperature}"
            )

    def initialize(self, context: TrainerContext):
        super().initialize(context)
        if self.temperature <= 0:
            raise ValueError(
                f"SoftmaxCrossEntropy temperature must be > 0, got {self.temperature}"
            )
        self.mode = context.state.model.outputType
        self.normalize = {
            ModuleOutputType.REAL: lambda x: F.log_softmax(x / self.temperature, -1),
            ModuleOutputType.LOG_PROBABILITY: lambda x: x,
            ModuleOutputType.PROBABILITY: lambda x: x.log(),
        }[context.state.model.outputType]

    def compute(self, scores, relevances, context):
        return -torch.logsumexp(
            self.normalize(scores) + (1 - 1.0 / relevances), -1
        ).sum() / len(scores)

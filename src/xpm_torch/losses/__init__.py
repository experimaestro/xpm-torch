import torch
from typing import NamedTuple

class Loss(NamedTuple):
    """A loss"""

    name: str
    value: torch.Tensor
    weight: float

import torch
from .utils import *

precision_to_dtype = {
    'bf16-true': torch.bfloat16,
    '16-true': torch.float16,
    '32-true': torch.float32,
    '64-true': torch.float64,
}

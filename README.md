# xpm-torch

PyTorch training framework built on [experimaestro](https://experimaestro-python.readthedocs.io/) and [Lightning Fabric](https://lightning.ai/docs/fabric/).

## Features

- **Module system**: `Module` base class combining experimaestro `Config` with `torch.nn.Module`, with safetensors serialization
- **Training**: `Learner` task with checkpointing, validation listeners, and TensorBoard logging
- **HuggingFace Hub**: Upload/download models via `ExperimaestroHFHub` (from experimaestro)
- **Distributed training**: Lightning Fabric integration for multi-GPU/node training

## Installation

```bash
pip install xpm-torch
```

## Quick Start

```python
from xpm_torch.module import Module
from experimaestro import Param
import torch
import torch.nn as nn

class MyModel(Module):
    input_dim: Param[int]
    hidden_dim: Param[int]

    def __initialize__(self):
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x):
        return self.fc1(x)

# Create config, then instance
cfg = MyModel.C(input_dim=50, hidden_dim=100)
model = cfg.instance()
model.initialize()

# Save/load with safetensors
from pathlib import Path
model.save_model(Path("checkpoint/model"))
model.load_model(Path("checkpoint/model"))
```

## License

GPL-3.0

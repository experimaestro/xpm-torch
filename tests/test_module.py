"""Tests for Module save_model/load_model and ModuleLoader."""

import torch
import torch.nn as nn
from pathlib import Path
from experimaestro import Param
from xpm_torch.module import Module, ModuleLoader


class SimpleModule(Module):
    input_dim: Param[int]
    output_dim: Param[int]

    def __initialize__(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.linear(x)


def test_save_load_model(tmp_path):
    """Test Module.save_model / load_model round-trip with safetensors."""
    cfg = SimpleModule.C(input_dim=10, output_dim=5)
    model = cfg.instance()
    model.initialize()

    # Run forward to ensure model works
    x = torch.randn(2, 10)
    out1 = model(x)

    # Save
    model_dir = tmp_path / "model"
    model.save_model(model_dir)
    assert (model_dir / "model.safetensors").exists()

    # Load into a new instance
    cfg2 = SimpleModule.C(input_dim=10, output_dim=5)
    model2 = cfg2.instance()
    model2.initialize()
    model2.load_model(model_dir)

    out2 = model2(x)
    assert torch.allclose(out1, out2), "Loaded model should produce same output"


def test_loader_config(tmp_path):
    """Test Module.loader_config returns a working ModuleLoader config."""
    cfg = SimpleModule.C(input_dim=10, output_dim=5)
    model = cfg.instance()
    model.initialize()

    # Save checkpoint (mimicking TrainState.save)
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    model_dir = checkpoint_path / "model"
    model.save_model(model_dir)

    # Get loader config and execute it (loader_config is called on the config)
    loader_cfg = cfg.loader_config(checkpoint_path)
    loader = loader_cfg.instance()
    loader.execute()

    # Verify loaded model works
    x = torch.randn(2, 10)
    out1 = model(x)
    out2 = loader.value(x)
    assert torch.allclose(out1, out2)

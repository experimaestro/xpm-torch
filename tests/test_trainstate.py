"""Tests for TrainState save/load with safetensors format."""

import torch
import torch.nn as nn
from experimaestro import Param
from xpm_torch.module import Module
from xpm_torch.trainers.context import TrainState


class DummyModule(Module):
    dim: Param[int]

    def __initialize__(self):
        self.linear = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        return self.linear(x)


class DummyTrainer:
    def state_dict(self):
        return {"dummy": 1}

    def load_state_dict(self, state):
        pass


class DummyOptimizer:
    def state_dict(self):
        return {"lr": 0.01}

    def load_state_dict(self, state):
        pass


def test_trainstate_save_creates_model_dir(tmp_path):
    """Test TrainState.save creates model/ directory with safetensors."""
    cfg = DummyModule.C(dim=4)
    model = cfg.instance()
    model.initialize()

    state = TrainState(model, DummyTrainer(), DummyOptimizer(), epoch=3, steps=100)

    save_path = tmp_path / "checkpoint"
    state.save(save_path)

    assert (save_path / "model").is_dir()
    assert (save_path / "model" / "model.safetensors").exists()
    assert (save_path / "info.json").exists()
    assert (save_path / "trainer.pth").exists()
    assert (save_path / "optimizer.pth").exists()


def test_trainstate_save_load_roundtrip(tmp_path):
    """Test TrainState save then load produces same model weights."""
    cfg = DummyModule.C(dim=4)
    model = cfg.instance()
    model.initialize()
    x = torch.randn(2, 4)
    out1 = model(x)

    state = TrainState(model, DummyTrainer(), DummyOptimizer(), epoch=5, steps=200)
    save_path = tmp_path / "checkpoint"
    state.save(save_path)

    # Create a new model and load
    cfg2 = DummyModule.C(dim=4)
    model2 = cfg2.instance()
    model2.initialize()
    state2 = TrainState(model2, DummyTrainer(), DummyOptimizer())
    state2.load(save_path)

    assert state2.epoch == 5
    assert state2.steps == 200

    out2 = model2(x)
    assert torch.allclose(out1, out2)


def test_trainstate_copy_model(tmp_path):
    """Test TrainState.copy_model copies model/ dir."""
    cfg = DummyModule.C(dim=4)
    model = cfg.instance()
    model.initialize()

    state = TrainState(model, DummyTrainer(), DummyOptimizer(), epoch=1, steps=10)
    save_path = tmp_path / "src"
    state.save(save_path)

    dest_path = tmp_path / "dest"
    dest_path.mkdir()
    state.copy_model(dest_path)

    assert (dest_path / "model").is_dir()
    assert (dest_path / "model" / "model.safetensors").exists()
    assert (dest_path / "info.json").exists()

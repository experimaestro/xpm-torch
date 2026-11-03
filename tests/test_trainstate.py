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


def _make_context(tmp_path):
    from xpm_torch.trainers.context import TrainerContext

    cfg = DummyModule.C(dim=4)
    model = cfg.instance()
    model.initialize()
    return TrainerContext(
        tmp_path / "runs",
        tmp_path / "checkpoints",
        max_epoch=35,
        steps_per_epoch=1,
        trainer=DummyTrainer(),
        model=model,
        optimizer=DummyOptimizer(),
        fabric=None,
    )


def _checkpoint_epochs(path):
    return sorted(int(p.name[len("epoch-") :]) for p in path.glob("epoch-*"))


def _train(context, start, end, interval):
    """Mimics the checkpointing part of Learner.execute"""
    for _ in range(start, end):
        context.nextepoch()
        if context.epoch % interval == 0:
            context.save_checkpoint()
            context.copy(context.path / "last")


def test_only_last_checkpoint_kept_with_interval(tmp_path):
    """Older checkpoints are removed even when checkpoint_interval > 1"""
    context = _make_context(tmp_path)
    _train(context, 0, 35, interval=10)

    assert _checkpoint_epochs(context.path) == [30]
    last = context.path / "last"
    assert (last / "model" / "model.safetensors").exists()
    assert (last / "info.json").read_text() == (
        context.path / "epoch-00000030" / "info.json"
    ).read_text()


def test_listener_copy_survives_checkpoint_removal(tmp_path):
    """A hard-linked copy (e.g. the best model) outlives its checkpoint"""
    context = _make_context(tmp_path)
    _train(context, 0, 10, interval=10)
    best = tmp_path / "best"
    context.copy(best)

    _train(context, 10, 20, interval=10)
    assert _checkpoint_epochs(context.path) == [20]
    assert (best / "model" / "model.safetensors").exists()


def test_stale_checkpoints_removed_after_resume(tmp_path):
    """Checkpoints left by a previous run are removed at the next save"""
    context = _make_context(tmp_path)
    _train(context, 0, 20, interval=1)
    assert _checkpoint_epochs(context.path) == [20]

    # Simulate leftovers from an interrupted run
    for epoch in (5, 10):
        TrainState(context.model, DummyTrainer(), DummyOptimizer(), epoch=epoch).save(
            context.path / f"epoch-{epoch:08d}"
        )

    resumed = _make_context(tmp_path)
    assert resumed.load_bestcheckpoint(35)
    assert resumed.epoch == 20
    _train(resumed, 20, 30, interval=10)
    assert _checkpoint_epochs(resumed.path) == [30]


def test_old_checkpoints_kept_when_new_one_is_invalid(tmp_path, monkeypatch):
    """If the new checkpoint does not verify, the older ones are kept"""
    import xpm_torch.trainers.context as ctx_module

    context = _make_context(tmp_path)
    _train(context, 0, 10, interval=10)
    assert _checkpoint_epochs(context.path) == [10]

    monkeypatch.setattr(
        ctx_module, "verify_checkpoint", lambda path: ["optimizer.pth: truncated"]
    )
    _train(context, 10, 20, interval=10)
    assert _checkpoint_epochs(context.path) == [10, 20]

    monkeypatch.undo()
    _train(context, 20, 30, interval=10)
    assert _checkpoint_epochs(context.path) == [30]

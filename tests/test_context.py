"""Tests for Context and TrainerContext ambient lifecycle, re-entrancy, and logging."""

from unittest.mock import MagicMock
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from xpm_torch.context import Context, Hook
from xpm_torch.losses import Loss
from xpm_torch.loggers import TensorboardLogger, NullMetricsLogger
from xpm_torch.trainers.context import TrainerContext


class DummyHook(Hook):
    def __call__(self, ctx, val):
        return val * 2


def test_context_ambient_lifecycle():
    """Verify Context.current() scope and Hook dispatch."""
    assert Context.current() is None

    hook = DummyHook()
    ctx = Context(hooks=[hook])

    assert Context.current() is None

    with ctx:
        assert Context.current() is ctx
        hooks = ctx.hooks(DummyHook)
        assert len(hooks) == 1
        assert hooks[0](ctx, 5) == 10

    assert Context.current() is None


def test_context_reentrancy():
    """Verify nested with-blocks correctly manage token stack."""
    ctx = Context()

    assert Context.current() is None
    with ctx:
        assert Context.current() is ctx
        with ctx:
            assert Context.current() is ctx
            with ctx:
                assert Context.current() is ctx
            assert Context.current() is ctx
        assert Context.current() is ctx
    assert Context.current() is None


def test_trainer_context_lifecycle_and_losses(tmp_path):
    """Verify TrainerContext.current() and with losses() accumulation."""
    ctx = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=10,
        trainer=None,
        model=nn.Linear(4, 1),
        optimizer=MagicMock(),
        fabric=None,
    )

    assert TrainerContext.current() is None

    with ctx:
        assert TrainerContext.current() is ctx

        with ctx.losses() as losses:
            assert TrainerContext.current() is ctx
            ctx.add_loss(Loss("reg_loss", 0.5, torch.tensor(1.0)))

        assert len(losses) == 1
        assert losses[0].name == "reg_loss"

    assert TrainerContext.current() is None


def test_activation_checkpointing_backward_guard(tmp_path):
    """Verify activation checkpointing recompute during backward does not duplicate losses."""
    ctx = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=1,
        trainer=None,
        model=None,
        optimizer=MagicMock(),
        fabric=None,
    )

    loss_call_count = 0

    class CheckpointedSubmodule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            nonlocal loss_call_count
            current = TrainerContext.current()
            if current is not None:
                loss_call_count += 1
                current.add_loss(Loss("chk_loss", 0.1, (self.linear(x) * 0.0).sum()))
            return self.linear(x)

    submodule = CheckpointedSubmodule()
    x = torch.randn(2, 4, requires_grad=True)

    with ctx:
        with ctx.losses() as losses:
            out = checkpoint(submodule, x, use_reentrant=True)
            total_loss = out.sum() + sum(loss.weight * loss.value for loss in losses)

        assert loss_call_count == 1
        assert len(losses) == 1

        # Backward pass recomputes forward; _is_backward guard must bypass add_loss:
        ctx.backward(total_loss)
        assert len(losses) == 1


def test_metrics_logger_and_writer_property(tmp_path):
    """Verify MetricsLogger API, histogram/flush methods, and context.writer escape hatch."""
    null_logger = NullMetricsLogger()
    null_logger.log_scalar("val", 1.0, 1)
    null_logger.log_histogram("hist", np.zeros(5), 1)
    null_logger.flush()
    null_logger.close()

    tb_logger = TensorboardLogger(tmp_path / "tb")
    tb_logger.log_scalar("scalar", 0.5, 1)
    tb_logger.log_histogram("hist", np.ones(5, dtype=np.float32), 1)
    tb_logger.flush()
    assert tb_logger.writer is not None
    tb_logger.close()

    ctx = TrainerContext(
        logpath=tmp_path / "logs",
        path=tmp_path / "ckpts",
        max_epoch=1,
        steps_per_epoch=1,
        trainer=None,
        model=None,
        optimizer=None,
        fabric=None,
    )
    assert ctx.writer is not None

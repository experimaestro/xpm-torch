"""Tests for checkpoint verification and cleanup (made-up data only)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from xpm_torch.trainers.checkpoints import (
    dir_size,
    job_status,
    last_target,
    plan_cleanup,
    verify_checkpoint,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "clean_checkpoints.py"


def make_checkpoint(path: Path, epoch: int, dim: int = 64):
    path.mkdir(parents=True)
    (path / "info.json").write_text(json.dumps({"epoch": epoch, "steps": epoch}))
    (path / "model").mkdir()
    save_file({"w": torch.randn(dim, dim)}, str(path / "model" / "model.safetensors"))
    torch.save({"step": epoch}, path / "trainer.pth")
    torch.save(
        {"state": {0: {"exp_avg": torch.randn(dim, dim)}}}, path / "optimizer.pth"
    )
    return path


def truncate(path: Path, ratio=0.5):
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * ratio)])


def make_job(root: Path, epochs, status="done", last=None):
    jobdir = root / "job"
    for epoch in epochs:
        make_checkpoint(jobdir / "checkpoints" / f"epoch-{epoch:08d}", epoch)
    # last and listener best copies are hard links
    last = epochs[-1] if last is None else last
    for target, epoch in (("checkpoints/last", last), ("out/best", epochs[0])):
        src = jobdir / "checkpoints" / f"epoch-{epoch:08d}"
        dst = jobdir / target / "model"
        dst.mkdir(parents=True)
        os.link(src / "model" / "model.safetensors", dst / "model.safetensors")
    if status:
        (jobdir / f"learner.{status}").write_text("{}")
    return jobdir


def test_verify_valid_checkpoint(tmp_path):
    path = make_checkpoint(tmp_path / "epoch-00000010", 10)
    assert verify_checkpoint(path) == []
    assert verify_checkpoint(path, load=False) == []


@pytest.mark.parametrize("name", ["model/model.safetensors", "optimizer.pth"])
@pytest.mark.parametrize("load", [True, False])
def test_verify_truncated(tmp_path, name, load):
    path = make_checkpoint(tmp_path / "epoch-00000010", 10)
    truncate(path / name)
    assert verify_checkpoint(path, load=load)


def test_verify_wrong_epoch_and_missing(tmp_path):
    path = make_checkpoint(tmp_path / "epoch-00000010", 20)
    (path / "trainer.pth").unlink()
    problems = verify_checkpoint(path)
    assert any("info.json" in p for p in problems)
    assert any("trainer.pth" in p for p in problems)


def ckpt(jobdir, epoch):
    return jobdir / "checkpoints" / f"epoch-{epoch:08d}"


def names(paths):
    return [p.name for p in paths]


def test_plan_falls_back_to_newest_valid(tmp_path):
    jobdir = make_job(tmp_path, [10, 20, 30, 40], last=20)
    for epoch in (30, 40):
        truncate(ckpt(jobdir, epoch) / "optimizer.pth")
    plan = plan_cleanup(jobdir / "checkpoints")
    assert plan.keep.name == "epoch-00000020"
    assert names(plan.remove) == ["epoch-00000010"]
    assert names(plan.corrupt) == ["epoch-00000030", "epoch-00000040"]
    assert names(plan.deletable()) == names(plan.remove + plan.corrupt)
    assert plan.last.name == "epoch-00000020" and not plan.last_corrupt


def test_plan_nothing_valid(tmp_path):
    jobdir = make_job(tmp_path, [10, 20])
    for epoch in (10, 20):
        truncate(ckpt(jobdir, epoch) / "optimizer.pth")
    plan = plan_cleanup(jobdir / "checkpoints")
    assert plan.keep is None and plan.remove == []
    assert len(plan.corrupt) == 2 and plan.deletable() == []


def test_plan_running_ignores_newest(tmp_path):
    """The newest checkpoint of a running job may be being written"""
    jobdir = make_job(tmp_path, [10, 20, 30])
    truncate(ckpt(jobdir, 30) / "optimizer.pth")
    plan = plan_cleanup(jobdir / "checkpoints", running=True)
    assert plan.in_progress.name == "epoch-00000030"
    assert plan.keep.name == "epoch-00000020"
    assert plan.corrupt == [] and names(plan.deletable()) == ["epoch-00000010"]


def test_last_target(tmp_path):
    jobdir = make_job(tmp_path, [10, 20, 30], last=20)
    assert last_target(jobdir / "checkpoints").name == "epoch-00000020"


def test_dir_size_ignores_hard_links(tmp_path):
    jobdir = make_job(tmp_path, [10, 20])
    ckpt = jobdir / "checkpoints" / "epoch-00000010"
    # The model is shared with out/best: only the other files are counted
    assert dir_size(ckpt) < dir_size(ckpt, unique_only=False)


def test_job_status(tmp_path):
    no_squeue = lambda: None  # noqa: E731
    jobdir = make_job(tmp_path, [10], status=None)
    assert job_status(jobdir, squeue=no_squeue) == "unknown"
    assert job_status(jobdir, squeue=lambda: (set(), set())) == "stopped"
    assert (
        job_status(jobdir, squeue=lambda: ({"1"}, {str(jobdir.resolve())})) == "running"
    )
    (jobdir / "learner.pid").write_text(json.dumps({"type": "slurm", "pid": "42"}))
    assert job_status(jobdir, squeue=no_squeue) == "unknown"
    assert job_status(jobdir, squeue=lambda: ({"42"}, set())) == "running"
    (jobdir / "learner.pid").unlink()
    (jobdir / "learner.done").write_text("{}")
    assert job_status(jobdir, squeue=no_squeue) == "done"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PATH": ""},  # no squeue
    ).stdout


def listing(jobdir):
    return sorted(p.name for p in (jobdir / "checkpoints").iterdir())


def test_script_dry_run_then_delete(tmp_path):
    jobdir = make_job(tmp_path, [10, 20, 30], last=20)
    truncate(ckpt(jobdir, 30) / "model" / "model.safetensors")
    before = listing(jobdir)

    out = run_script(tmp_path)
    assert "old     epoch-00000010 (would delete" in out
    assert "corrupt epoch-00000030 (would delete" in out
    assert "INVALID epoch-00000030" in out
    assert "WARNING" not in out
    assert listing(jobdir) == before

    run_script(jobdir, "--delete")
    assert listing(jobdir) == ["epoch-00000020", "last"]
    assert (jobdir / "out" / "best" / "model" / "model.safetensors").exists()
    assert last_target(jobdir / "checkpoints").name == "epoch-00000020"


def test_script_nothing_valid_deletes_nothing(tmp_path):
    jobdir = make_job(tmp_path, [10, 20])
    for epoch in (10, 20):
        truncate(ckpt(jobdir, epoch) / "optimizer.pth")
    out = run_script(jobdir, "--delete", "--fix-last")
    assert "no valid checkpoint" in out
    assert listing(jobdir) == ["epoch-00000010", "epoch-00000020", "last"]


def test_script_last_on_corrupt_checkpoint(tmp_path):
    jobdir = make_job(tmp_path, [10, 20])
    truncate(ckpt(jobdir, 20) / "model" / "model.safetensors")
    last_model = jobdir / "checkpoints" / "last" / "model" / "model.safetensors"

    out = run_script(jobdir, "--delete")
    assert "WARNING: last mirrors the corrupt epoch-00000020" in out
    assert "use --fix-last" in out
    assert listing(jobdir) == ["epoch-00000010", "last"]
    # last is left as is (the hard link keeps the corrupt model alive)...
    assert last_target(jobdir / "checkpoints") is None
    assert last_model.exists()

    # ... and is still detected as corrupt, then re-pointed
    out = run_script(jobdir, "--delete", "--fix-last")
    assert "WARNING: last is corrupt" in out
    assert "re-point last to epoch-00000010" in out
    assert listing(jobdir) == ["epoch-00000010", "last"]
    assert last_target(jobdir / "checkpoints").name == "epoch-00000010"


def test_script_skips_unfinished_jobs(tmp_path):
    jobdir = make_job(tmp_path, [10, 20], status=None)
    out = run_script(jobdir, "--delete")
    assert "SKIPPED" in out
    assert len(list((jobdir / "checkpoints").glob("epoch-*"))) == 2


def test_script_force_never_touches_newest(tmp_path):
    jobdir = make_job(tmp_path, [10, 20, 30], status=None)
    truncate(ckpt(jobdir, 30) / "optimizer.pth")  # being written
    out = run_script(jobdir, "--delete", "--force")
    assert "ignore  epoch-00000030" in out
    assert listing(jobdir) == ["epoch-00000020", "epoch-00000030", "last"]

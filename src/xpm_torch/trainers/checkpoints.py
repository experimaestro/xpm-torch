"""Checkpoint verification and cleanup helpers.

A checkpoint (see :class:`xpm_torch.trainers.context.TrainState`) is a folder
``epoch-XXXXXXXX`` containing ``info.json``, ``model/model.safetensors`` (or a
legacy ``model.pth``), ``trainer.pth`` and ``optimizer.pth``.

:func:`verify_checkpoint` checks that a checkpoint can be read back without
instantiating the model, so that it can be used both during training (before
removing older checkpoints) and offline (``scripts/clean_checkpoints.py``).
"""

import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("xpm_torch.trainer.checkpoints")

PREFIX = "epoch-"
EPOCH_RE = re.compile(rf"^{PREFIX}(\d+)$")

#: A file whose allocated blocks cover less than this fraction of its size is
#: considered to contain holes (e.g. a write interrupted by a full disk)
MIN_ALLOCATED_RATIO = 0.5


def checkpoint_epoch(path: Path) -> Optional[int]:
    """Returns the epoch of a checkpoint folder, or None if not a checkpoint"""
    m = EPOCH_RE.match(path.name)
    return int(m.group(1)) if m else None


def list_checkpoints(checkpointspath: Path) -> List[Path]:
    """Lists the checkpoint folders, sorted by increasing epoch"""
    if not checkpointspath.is_dir():
        return []
    paths = [
        p
        for p in checkpointspath.iterdir()
        if p.is_dir() and not p.is_symlink() and checkpoint_epoch(p) is not None
    ]
    return sorted(paths, key=checkpoint_epoch)


def _check_allocated(path: Path, problems: List[str]):
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None and st.st_size > 0:
        if blocks * 512 < MIN_ALLOCATED_RATIO * st.st_size:
            problems.append(
                f"{path.name}: sparse file ({blocks * 512} bytes allocated for "
                f"{st.st_size}), probably not fully written"
            )


def _verify_safetensors(path: Path, problems: List[str], load_tensors: bool):
    import struct

    _check_allocated(path, problems)
    size = path.stat().st_size
    try:
        with path.open("rb") as fp:
            (header_size,) = struct.unpack("<Q", fp.read(8))
            if header_size + 8 > size:
                problems.append(f"{path.name}: header larger than file")
                return
            header = json.loads(fp.read(header_size))
    except Exception as e:
        problems.append(f"{path.name}: cannot read header ({e})")
        return

    # Check that the tensor byte ranges cover exactly the data section
    end = 0
    for key, info in header.items():
        if key == "__metadata__":
            continue
        start, stop = info["data_offsets"]
        end = max(end, stop)
    if 8 + header_size + end != size:
        problems.append(
            f"{path.name}: size mismatch (expected {8 + header_size + end}, got {size})"
        )
        return

    if not load_tensors:
        return

    try:
        import torch
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    problems.append(f"{path.name}: non-finite values in {key}")
    except Exception as e:
        problems.append(f"{path.name}: cannot load tensors ({e})")


def _verify_torch_file(path: Path, problems: List[str], load: bool):
    _check_allocated(path, problems)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                # Cheap: the central directory (at the end of the file) is
                # readable, and all the entries fit in the file
                size = path.stat().st_size
                for info in zf.infolist():
                    if info.header_offset + info.compress_size > size:
                        problems.append(f"{path.name}: truncated ({info.filename})")
                        return
                if load:
                    # Reads all the entries and checks their CRC
                    bad = zf.testzip()
                    if bad is not None:
                        problems.append(f"{path.name}: corrupted entry {bad}")
                        return
        elif not load:
            problems.append(f"{path.name}: not a valid zip file (truncated?)")
            return
        if load:
            import torch

            # Same as TrainState.load (weights_only=False), but memory-mapped
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        problems.append(f"{path.name}: cannot load ({e})")


def _verify_model(path: Path, problems: List[str], load: bool):
    model_dir = path / "model"
    if model_dir.is_dir():
        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            problems.append("model/: no safetensors file")
        for f in files:
            _verify_safetensors(f, problems, load)
    elif (path / "model.pth").is_file():
        _verify_torch_file(path / "model.pth", problems, load)
    else:
        problems.append("no model/ folder nor model.pth")


def verify_model(path: Path, load: bool = True) -> List[str]:
    """Checks the model files of a checkpoint (or of ``last``)"""
    problems: List[str] = []
    _verify_model(Path(path), problems, load)
    return problems


def verify_checkpoint(path: Path, load: bool = True) -> List[str]:
    """Checks that a checkpoint folder can be loaded back

    :param path: the checkpoint folder
    :param load: if False, only cheap checks are performed (structure, file
        sizes, safetensors headers, zip directories); otherwise, all the
        tensors are loaded and the zip CRCs checked
    :returns: a list of problems (empty if the checkpoint is valid)
    """
    path = Path(path)
    problems: List[str] = []

    try:
        with (path / "info.json").open("rt") as fp:
            info = json.load(fp)
        epoch = checkpoint_epoch(path)
        if epoch is not None and info.get("epoch") != epoch:
            problems.append(
                f"info.json: epoch {info.get('epoch')} does not match {path.name}"
            )
    except Exception as e:
        problems.append(f"info.json: cannot read ({e})")

    _verify_model(path, problems, load)

    for name in ("trainer.pth", "optimizer.pth"):
        f = path / name
        if not f.is_file():
            problems.append(f"{name}: missing")
        else:
            _verify_torch_file(f, problems, load)

    return problems


def dir_size(path: Path, unique_only: bool = True) -> int:
    """Disk usage of a folder (bytes).

    With ``unique_only``, files having other hard links (e.g. the ``last``
    or a listener best copy) are not counted since deleting them does not
    free space."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            st = os.lstat(os.path.join(root, name))
            if unique_only and st.st_nlink > 1:
                continue
            total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
    return total


LAST = "last"
"""Folder holding hard links to the model of the last saved checkpoint
(``Learner.last_checkpoint_path``); not used when resuming, which only looks
at the ``epoch-*`` folders"""


def _model_inodes(path: Path) -> Set[Tuple[int, int]]:
    """(device, inode) of the model files of a checkpoint (or of ``last``)"""
    inodes = set()
    for f in [*(path / "model").glob("*"), path / "model.pth"]:
        if f.is_file():
            st = f.stat()
            inodes.add((st.st_dev, st.st_ino))
    return inodes


def last_target(checkpointspath: Path) -> Optional[Path]:
    """Returns the checkpoint that ``last`` mirrors (shares model files with)"""
    inodes = _model_inodes(checkpointspath / LAST)
    if not inodes:
        return None
    for path in list_checkpoints(checkpointspath):
        if _model_inodes(path) & inodes:
            return path
    return None


@dataclass
class CleanupPlan:
    checkpointspath: Path
    keep: Optional[Path] = None
    """The checkpoint that is kept (newest valid one)"""

    remove: List[Path] = field(default_factory=list)
    """Checkpoints older than ``keep``"""

    corrupt: List[Path] = field(default_factory=list)
    """Checkpoints newer than ``keep`` that failed verification (removable
    only if ``keep`` is not None)"""

    invalid: dict = field(default_factory=dict)
    """Problems of the checkpoints that failed verification"""

    in_progress: Optional[Path] = None
    """Newest checkpoint of a running job: ignored (it may be being written)"""

    last: Optional[Path] = None
    """The checkpoint mirrored by ``last``"""

    last_problems: List[str] = field(default_factory=list)
    """Problems of the model in ``last`` (checked directly when ``last`` does
    not mirror any checkpoint)"""

    @property
    def last_corrupt(self) -> bool:
        return bool(self.last_problems) or (
            self.last is not None and self.last in self.invalid
        )

    def deletable(self) -> List[Path]:
        """Checkpoints that can be safely deleted"""
        if self.keep is None:
            # Nothing valid: keep everything (in case of a false positive)
            return []
        return [*self.remove, *self.corrupt]


def plan_cleanup(
    checkpointspath: Path, verify=True, load: bool = True, running: bool = False
) -> CleanupPlan:
    """Finds the newest checkpoint that loads, the older ones and the corrupt
    ones (newer than the kept one)

    :param verify: if False, the newest checkpoint is kept without checks
    :param running: the job is running: the newest checkpoint is left
        untouched (not verified nor deleted) since it may be being written
    """
    plan = CleanupPlan(checkpointspath)
    checkpoints = list_checkpoints(checkpointspath)
    plan.last = last_target(checkpointspath)
    if running and checkpoints:
        plan.in_progress = checkpoints.pop()

    for ix in range(len(checkpoints) - 1, -1, -1):
        path = checkpoints[ix]
        problems = verify_checkpoint(path, load=load) if verify else []
        if problems:
            plan.invalid[path] = problems
            plan.corrupt.insert(0, path)
            continue
        plan.keep = path
        plan.remove = checkpoints[:ix]
        break

    last = checkpointspath / LAST
    if verify and plan.last is None and last.is_dir():
        plan.last_problems = verify_model(last, load=load)
    return plan


def repoint_last(checkpointspath: Path, target: Path):
    """Makes ``last`` mirror ``target`` (hard links, as
    :meth:`TrainState.copy_model`)"""
    last = checkpointspath / LAST
    assert checkpoint_epoch(target) is not None and target.parent == checkpointspath
    if last.exists():
        shutil.rmtree(last)
    last.mkdir()
    os.link(target / "info.json", last / "info.json")
    if (target / "model").is_dir():
        shutil.copytree(target / "model", last / "model", copy_function=os.link)
    else:
        os.link(target / "model.pth", last / "model.pth")


def remove_checkpoints(paths: Iterable[Path]):
    for path in paths:
        assert checkpoint_epoch(path) is not None, f"{path} is not a checkpoint"
        logger.info("Removing checkpoint %s", path)
        try:
            shutil.rmtree(path)
        except OSError as e:
            logger.error("OS Error while trying to remove directory %s", e)


# --- Job status (experimaestro job folders)


def _squeue() -> Optional[Tuple[Set[str], Set[str]]]:
    """Returns the (job ids, work dirs) of the user's active SLURM jobs, or
    None if squeue is unavailable"""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-o", "%i|%Z", "-u", os.environ.get("USER", "")],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except Exception:
        return None
    ids, workdirs = set(), set()
    for line in out.splitlines():
        job_id, _, workdir = line.partition("|")
        ids.add(job_id.strip())
        workdirs.add(workdir.strip())
    return ids, workdirs


def job_status(jobdir: Path, squeue=_squeue) -> str:
    """Returns one of 'done', 'failed', 'running', 'stopped' or 'unknown'

    'unknown' is returned when it cannot be decided whether the job is
    running (e.g. a SLURM job but no squeue command)."""
    jobdir = Path(jobdir).resolve()
    queue = squeue()

    pidfiles = list(jobdir.glob("*.pid"))
    for pidfile in pidfiles:
        try:
            info = json.loads(pidfile.read_text())
        except Exception:
            return "unknown"
        if info.get("type") == "slurm":
            if queue is None:
                return "unknown"
            if str(info.get("pid")) in queue[0]:
                return "running"
        elif info.get("type") == "local":
            try:
                os.kill(int(info["pid"]), 0)
                return "running"
            except (OSError, ValueError, KeyError):
                pass
        else:
            return "unknown"

    if queue is not None and str(jobdir) in queue[1]:
        return "running"

    if list(jobdir.glob("*.done")):
        return "done"
    if list(jobdir.glob("*.failed")):
        return "failed"
    return "stopped" if queue is not None or pidfiles else "unknown"

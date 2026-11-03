#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.16"
# dependencies = ["xpm-torch"]
#
# [tool.uv.sources]
# xpm-torch = { path = "..", editable = true }
# ///
"""Removes the old checkpoints of xpm-torch learner jobs.

For each job folder, keeps the newest checkpoint (``checkpoints/epoch-*``)
that loads properly, and removes the older ones as well as the newer ones
that are corrupt (fail verification). Nothing is removed when no checkpoint
is valid. ``checkpoints/last`` and the listeners' copies (``out/...``) are
hard links and are not touched (except ``last`` with ``--fix-last``); the
space of a model shared with them is not freed (nor counted).

Running jobs are skipped; with ``--force``, their newest checkpoint (which
may be being written) is ignored.

Dry-run by default: use ``--delete`` to actually remove the checkpoints.

Can also be run with any Python where xpm-torch is installed, e.g.
``.venv/bin/python scripts/clean_checkpoints.py``.
"""

import argparse
import logging
import sys
from pathlib import Path

from xpm_torch.trainers.checkpoints import (
    LAST,
    dir_size,
    job_status,
    list_checkpoints,
    plan_cleanup,
    remove_checkpoints,
    repoint_last,
)


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n:.1f}{unit}"
        n /= 1024


def find_jobs(paths):
    """A path is either a job folder (with checkpoints/), or any folder
    containing job folders (searched up to two levels below)"""
    for path in paths:
        path = Path(path).expanduser().resolve()
        if (path / "checkpoints").is_dir():
            yield path
            continue
        found = sorted(
            {p.parent.parent for p in path.glob("*/checkpoints/epoch-*")}
            | {p.parent.parent for p in path.glob("*/*/checkpoints/epoch-*")}
        )
        if not found:
            logging.warning("No job with checkpoints found in %s", path)
        yield from found


def process_job(jobdir: Path, args) -> int:
    """Returns the number of bytes freed (or that would be freed)"""
    checkpointspath = jobdir / "checkpoints"
    checkpoints = list_checkpoints(checkpointspath)
    status = job_status(jobdir)
    print(f"\n== {jobdir} [{status}], {len(checkpoints)} checkpoint(s)")
    running = status in ("running", "unknown")
    if running and not args.force:
        print(f"   SKIPPED: job is {status} (use --force to process it)")
        return 0
    if not checkpoints:
        print("   nothing to do")
        return 0

    plan = plan_cleanup(
        checkpointspath,
        verify=args.verify != "none",
        load=args.verify == "full",
        running=running,
    )
    if plan.in_progress is not None:
        print(f"   ignore  {plan.in_progress.name} (newest, job is {status})")
    for path in plan.corrupt:
        print(f"   INVALID {path.name}: {'; '.join(plan.invalid[path])}")

    if plan.keep is None:
        if plan.corrupt:
            print("   no valid checkpoint: nothing removed (check by hand)")
        else:
            print("   nothing to do")
        return 0

    print(f"   keep    {plan.keep.name}")
    prefix = "" if args.delete else "would "
    freed = 0
    for path in plan.remove:
        size = dir_size(path)
        freed += size
        print(f"   old     {path.name} ({prefix}delete, {human(size)})")
    for path in plan.corrupt:
        size = dir_size(path)
        freed += size
        print(f"   corrupt {path.name} ({prefix}delete, {human(size)})")

    if plan.last_corrupt:
        what = (
            f"mirrors the corrupt {plan.last.name}"
            if plan.last is not None
            else f"is corrupt ({'; '.join(plan.last_problems)})"
        )
        print(
            f"   WARNING: {LAST} {what}: the learned model exported from this "
            "job is broken"
        )
        if running:
            print(f"   {LAST} not re-pointed since the job is {status}")
        elif args.fix_last:
            print(f"   {prefix}re-point {LAST} to {plan.keep.name}")
        else:
            print(f"   use --fix-last to re-point {LAST} to {plan.keep.name}")

    if args.delete:
        remove_checkpoints(plan.deletable())
        if plan.last_corrupt and args.fix_last and not running:
            repoint_last(checkpointspath, plan.keep)
    return freed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Job folders or folders containing them"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete (default: dry-run)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Also process running/unknown jobs (their newest checkpoint is "
        "never touched)",
    )
    parser.add_argument(
        "--fix-last",
        action="store_true",
        help="If last mirrors a corrupt checkpoint, re-point it to the kept one "
        "(with --delete, for jobs that are not running)",
    )
    parser.add_argument(
        "--verify",
        choices=["full", "quick", "none"],
        default="full",
        help="full: load every tensor and state file (reads everything); "
        "quick: structure, sizes and safetensors headers only; "
        "none: keep the newest checkpoint without checking it",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.delete and args.verify == "none":
        parser.error("--delete requires --verify full or quick")

    total = sum(process_job(jobdir, args) for jobdir in find_jobs(args.paths))
    print(f"\nTotal {'freed' if args.delete else 'reclaimable'}: {human(total)}")
    if not args.delete:
        print("(dry-run: use --delete to remove)")


if __name__ == "__main__":
    sys.exit(main())

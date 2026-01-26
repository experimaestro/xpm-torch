import threading
from pathlib import Path

from experimaestro import (
    tagspath,
    Task,
    experiment,
    RunMode,
)
from experimaestro.scheduler import Job, Listener
from experimaestro.utils import cleanupdir
from xpm_torch.utils.logging import easylog
from experimaestro.scheduler.services import WebService

logger = easylog()

class TensorboardServiceListener(Listener):
    def __init__(self, source: Path, target: Path):
        self.source = source
        self.target = target
        if not self.target.exists():
            logger.info("Creating tensorboard target directory %s", self.target)
            self.target.mkdir(parents=True, exist_ok=True)

    def job_state(self, job: Job):
        #create symlink even if job has not been launched yet
        if not self.source.is_symlink():
            try:
                self.source.symlink_to(self.target)
            except Exception:
                logger.exception(
                    "Cannot symlink %s to %s", self.source, self.target
                )


class TensorboardService(WebService):
    id = "tensorboard"

    def __init__(self, xp: experiment, path: Path):
        super().__init__()

        self.path = path
        self.url = None
        self.run_mode = xp.run_mode

        if self.run_mode == RunMode.NORMAL:
            cleanupdir(self.path)
            self.path.mkdir(exist_ok=True, parents=True)
            logger.info("You can monitor learning with:")
            logger.info("tensorboard --logdir=%s", self.path)

    def add(self, task: Task, path: Path):
        # Wait until config has started
        if self.run_mode == RunMode.NORMAL:
            if job := task.__xpm__.job:
                if job.scheduler is not None:
                    tag_path = tagspath(task)
                    if tag_path:
                        job.scheduler.addlistener(
                            TensorboardServiceListener(self.path / tag_path, path)
                        )
                    else:
                        logger.error(
                            "The task is not associated with tags: "
                            "cannot link to tensorboard data"
                        )
                else:
                    logger.warning("No scheduler: not adding the tensorboard data")
            else:
                logger.error(
                    "Task was not started: cannot link to tensorboard job path"
                )

    def description(self):
        return "Tensorboard service"

    def close(self):
        if self.server and self.run_mode == RunMode.NORMAL:
            self.server.shutdown()

    def _serve(self, running: threading.Event):
        if self.run_mode != RunMode.NORMAL:
            return

        import tensorboard as tb

        try:
            logger.info("Starting %s service", self.id)
            self.program = tb.program.TensorBoard()
            self.program.configure(
                host="localhost",
                logdir=str(self.path.absolute()),
                path_prefix=f"/services/{self.id}",
                port=0,
            )
            self.server = self.program._make_server()

            self.url = self.server.get_url()
            running.set()
            self.server.serve_forever()
        except Exception:
            logger.exception("Error while starting tensorboard")
            running.set()

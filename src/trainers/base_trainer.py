import csv
import os

from ..nn import callbacks as cbs

from lightning import Trainer
from lightning.pytorch.loggers.csv_logs import CSVLogger, ExperimentWriter


class AppendSafeExperimentWriter(ExperimentWriter):
    """Preserve and append to an existing metrics file when a run resumes."""

    def _check_log_dir_exists(self) -> None:
        # Lightning's default writer removes metrics.csv whenever an explicit
        # logger version already exists. Our experiment directory is stable so
        # resumed checkpoints must retain the validation history written before
        # an interruption.
        pass

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        if self._fs.isfile(self.metrics_file_path):
            with self._fs.open(self.metrics_file_path, mode="r", newline="") as handle:
                self.metrics_keys = next(csv.reader(handle), [])


class AppendSafeCSVLogger(CSVLogger):
    """CSV logger whose fixed experiment version supports checkpoint resume."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._experiment = AppendSafeExperimentWriter(self.log_dir)


class CommonLitTrainer(Trainer):
    def __init__(self, *args, callbacks=None, **kwargs):
        if callbacks is not None:
            callbacks = [cbs.get(c) for c in callbacks]

        workbench = os.environ.get("WORKBENCH", "./workbench")
        experiment = os.environ.get("EXPERIMENT_NAME", "experiment")
        logger = AppendSafeCSVLogger(
            save_dir=os.path.join(workbench, "logs"),
            name=experiment,
            version="",
        )

        super(CommonLitTrainer, self).__init__(
            *args, callbacks=callbacks, logger=logger, **kwargs
        )

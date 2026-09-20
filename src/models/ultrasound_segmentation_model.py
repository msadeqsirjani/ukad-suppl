import torch

from ._seg_metrics import binary_seg_metrics
from .base_model import CommonLitModel


class UltrasoundSegmentationModel(CommonLitModel):
    def __init__(self, *args, **kwargs):
        super(UltrasoundSegmentationModel, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        if hasattr(self._model, "regularization_loss"):
            loss = loss + self._model.regularization_loss()
        if hasattr(self._model, "deep_supervision_loss"):
            max_epochs = max(int(self.trainer.max_epochs) - 1, 1)
            progress = float(self.current_epoch) / max_epochs
            aux_loss = self._model.deep_supervision_loss(y, self._criterion, progress=progress)
            loss = loss + aux_loss
            logs["deep_supervision_loss"] = aux_loss.detach()
        if getattr(self._model, "boundary_refinement", False):
            boundary_loss = self._model.boundary_supervision_loss(y)
            loss = loss + boundary_loss
            logs["boundary_loss"] = boundary_loss.detach()
        logs["loss"] = loss.detach()
        logs = {"train/" + k: v for k, v in logs.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)

        with torch.no_grad():
            seg = binary_seg_metrics(y_pred, y)

        logs = {"val/" + k: v for k, v in logs.items()}
        for k, v in seg.items():
            logs[f"val/{k}"] = v
        logs["step"] = self.current_epoch
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

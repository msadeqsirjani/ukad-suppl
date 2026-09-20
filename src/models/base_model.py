from . import nets
from ..nn import losses, optimizers
from src.utils.serialization_utils import create_object
from src.utils.torch_utils import split_loss_logs

from lightning import LightningModule


class CommonLitModel(LightningModule):
    def __init__(
        self,
        network,
        criterion=None,
        optimizer=None,
        scheduler="optimizer_default",
    ):
        super(CommonLitModel, self).__init__()

        self._model = nets.get(network)
        self._parameter_groups = []
        self._scheduler = None
        if criterion is not None:
            self._criterion = losses.get(criterion)
        if optimizer is not None:
            self._optimizer = optimizers.get(optimizer, partial=True)
            optimizer_metadata = optimizer if isinstance(optimizer, dict) else {}
            self._parameter_groups = optimizer_metadata.get("parameter_groups", [])
            if scheduler == "optimizer_default":
                self._scheduler = optimizer_metadata.get("lr_scheduler")
            else:
                self._scheduler = scheduler

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            try:
                y_pred = self._model(**x, **kwargs)
            except Exception:
                y_pred = self._model(x, **kwargs)
        elif isinstance(x, (list, tuple)):
            try:
                y_pred = self._model(*x, **kwargs)
            except Exception:
                y_pred = self._model(x, **kwargs)
        else:
            y_pred = self._model(x, **kwargs)

        return y_pred

    def compute_loss(self, y_pred, y):
        if isinstance(y, dict):
            try:
                value = self._criterion(y_pred, **y)
            except Exception:
                value = self._criterion(y_pred, y)
        elif isinstance(y, (list, tuple)):
            try:
                value = self._criterion(y_pred, *y)
            except Exception:
                value = self._criterion(y_pred, y)
        else:
            value = self._criterion(y_pred, y)

        return split_loss_logs(value)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        named_parameters = list(self.named_parameters())
        parameter_groups = []
        assigned = set()
        for rule in self._parameter_groups:
            match_all = [str(token).lower() for token in rule.get("match_all", [])]
            match_any = [str(token).lower() for token in rule.get("match_any", [])]
            selected = []
            for name, parameter in named_parameters:
                lowered = name.lower()
                if id(parameter) in assigned:
                    continue
                if match_all and not all(token in lowered for token in match_all):
                    continue
                if match_any and not any(token in lowered for token in match_any):
                    continue
                selected.append(parameter)
                assigned.add(id(parameter))
            if not selected:
                raise ValueError(f"optimizer parameter rule matched nothing: {rule}")
            options = {
                key: value
                for key, value in rule.items()
                if key not in {"match_all", "match_any", "name"}
            }
            parameter_groups.append({"params": selected, **options})

        remaining = [
            parameter
            for _, parameter in named_parameters
            if id(parameter) not in assigned
        ]
        if remaining:
            parameter_groups.append({"params": remaining})
        optimizer = self._optimizer(params=parameter_groups or self.parameters())
        if self._scheduler is None:
            return optimizer
        scheduler = create_object(self._scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

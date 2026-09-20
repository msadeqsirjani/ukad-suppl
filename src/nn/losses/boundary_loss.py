import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F


class BoundaryLoss(nn.Module):
    @staticmethod
    def _distance_map(mask):
        mask_np = mask.detach().cpu().numpy().astype(bool)
        dist = np.zeros(mask_np.shape, dtype=np.float32)
        for idx in np.ndindex(mask_np.shape[:-2]):
            fg = mask_np[idx]
            if fg.any() and not fg.all():
                dist[idx] = distance_transform_edt(~fg) - distance_transform_edt(fg)
            elif fg.all():
                dist[idx] = -distance_transform_edt(fg)
            else:
                dist[idx] = distance_transform_edt(~fg)
        return dist

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1) if inputs.shape[1] > 1 else torch.sigmoid(inputs)
        dist = self._distance_map(targets)
        dist = torch.from_numpy(dist).to(device=inputs.device, dtype=inputs.dtype)
        diagonal = (inputs.shape[-2] ** 2 + inputs.shape[-1] ** 2) ** 0.5
        return (dist / diagonal * probs).mean()

import torch

SMOOTH = 1e-5


def binary_seg_metrics(logits, target, threshold=0.5, eps=SMOOTH):
    pred = torch.sigmoid(logits) > threshold
    tgt = target > 0.5

    intersection = torch.logical_and(pred, tgt).sum(dtype=torch.float64)
    union = torch.logical_or(pred, tgt).sum(dtype=torch.float64)
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * iou) / (iou + 1.0)

    return {"iou": iou.item(), "dice_score": dice.item()}


def seg_indicators(logits, target, threshold=0.5):
    from medpy.metric.binary import dc, hd, hd95, jc, precision, recall, specificity

    pred = (torch.sigmoid(logits) > threshold).cpu().numpy()
    tgt = (target > 0.5).cpu().numpy()

    metrics = {
        "iou": jc,
        "dice_score": dc,
        "hd": hd,
        "hd95": hd95,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
    }
    return {name: float(function(pred, tgt)) for name, function in metrics.items()}

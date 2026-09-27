"""Vessel-geometry measures against expert annotations.

Ref: Sec. 4.10 (Dice overlap and the 95th-percentile Hausdorff distance), Table 1, Table SB1.
"""

from __future__ import annotations

import torch
from torch import Tensor

DICE_EPSILON: float = 1e-8


def dice_overlap(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Overlap of the binarised vessel mask with the expert annotation.

    Ref: Sec. 4.10.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    predicted = (prediction > threshold).to(torch.float64)
    reference = (target > 0.5).to(torch.float64)
    intersection = float((predicted * reference).sum().item())
    denominator = float(predicted.sum().item() + reference.sum().item())
    if denominator <= 0.0:
        return 1.0
    return 2.0 * intersection / (denominator + DICE_EPSILON)


def _surface_points(mask: Tensor) -> Tensor:
    binarised = (mask > 0.5).to(torch.float64)
    if binarised.dim() != 2:
        raise ValueError("a single binary mask is required")
    eroded = torch.nn.functional.max_pool2d((-binarised).unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    eroded = (-eroded).squeeze(0).squeeze(0)
    surface = binarised * (1.0 - eroded)
    points = torch.nonzero(surface > 0.5, as_tuple=False)
    if points.numel() == 0:
        points = torch.nonzero(binarised > 0.5, as_tuple=False)
    return points.to(torch.float64)


def percentile_hausdorff(prediction: Tensor, target: Tensor, percentile: float = 95.0) -> float:
    """Robust directed Hausdorff distance at the requested percentile.

    Ref: Sec. 4.10.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    if prediction.dim() != 2:
        raise ValueError("Hausdorff distance is computed on a single frame")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must lie in (0, 100]")
    predicted = _surface_points(prediction)
    reference = _surface_points(target)
    if predicted.numel() == 0 or reference.numel() == 0:
        raise ValueError("empty vessel surface")
    forward = torch.cdist(predicted, reference).min(dim=1).values
    backward = torch.cdist(reference, predicted).min(dim=1).values
    directed = torch.maximum(torch.quantile(forward, percentile / 100.0), torch.quantile(backward, percentile / 100.0))
    return float(directed.item())


def geometry_report(prediction: Tensor, target: Tensor) -> dict[str, float]:
    frames = prediction.shape[-1] if prediction.dim() == 3 else 1
    if prediction.dim() == 3:
        dice = sum(dice_overlap(prediction[:, :, index], target[:, :, index]) for index in range(frames)) / frames
        hausdorff = (
            sum(percentile_hausdorff(prediction[:, :, index], target[:, :, index]) for index in range(frames)) / frames
        )
    else:
        dice = dice_overlap(prediction, target)
        hausdorff = percentile_hausdorff(prediction, target)
    return {"dice": dice, "hausdorff95": hausdorff}


def dice_by_stratum(prediction: Tensor, target: Tensor, labels: list[str]) -> dict[str, float]:
    """Dice resolved by anatomy-complexity stratum.

    Ref: Table SB1 (pooled column and the three stratum columns).
    """
    if prediction.shape[0] != len(labels):
        raise ValueError("one stratum label is required per procedure")
    buckets: dict[str, list[float]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(dice_overlap(prediction[index], target[index]))
    return {label: sum(values) / len(values) for label, values in buckets.items()}

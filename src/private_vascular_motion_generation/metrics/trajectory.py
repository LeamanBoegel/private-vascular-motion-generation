"""Motion-agreement measures for the generated catheter trajectory.

Ref: Sec. 4.10 (ADE, FDE, DTW, smoothness, path-length efficiency), Table 1.
"""

from __future__ import annotations

import torch
from torch import Tensor

PATH_EFFICIENCY_ARTIFACT_CEILING: float = 0.92
CENTRELINE_TOLERANCE_FRACTION: float = 0.08


def average_displacement_error(generated: Tensor, recorded: Tensor) -> float:
    """Mean tip displacement error in millimetres.

    Ref: Sec. 4.10, Table 1.
    """
    if generated.shape != recorded.shape:
        raise ValueError("generated and recorded trajectories must align")
    if generated.dim() != 2 or generated.shape[1] != 3:
        raise ValueError("trajectories must be frames x 3")
    return float(torch.linalg.norm(generated - recorded, dim=1).mean().item())


def final_displacement_error(generated: Tensor, recorded: Tensor) -> float:
    if generated.shape != recorded.shape:
        raise ValueError("generated and recorded trajectories must align")
    if generated.dim() != 2:
        raise ValueError("trajectories must be frames x coordinates")
    return float(torch.linalg.norm(generated[-1] - recorded[-1]).item())


def dynamic_time_warping(generated: Tensor, recorded: Tensor, band: int | None = None) -> float:
    """Normalised dynamic-time-warping distance between the two trajectories.

    Ref: Sec. 4.10, Table 1.
    """
    if generated.shape != recorded.shape:
        raise ValueError("generated and recorded trajectories must align")
    if generated.dim() != 2:
        raise ValueError("trajectories must be frames x coordinates")
    length = int(generated.shape[0])
    if length == 0:
        raise ValueError("empty trajectory")
    cost = torch.cdist(generated.to(torch.float64), recorded.to(torch.float64))
    width = band if band is not None else length
    accumulated = torch.full((length + 1, length + 1), float("inf"), dtype=torch.float64)
    accumulated[0, 0] = 0.0
    for row in range(1, length + 1):
        low = max(1, row - width)
        high = min(length, row + width)
        for column in range(low, high + 1):
            previous = torch.minimum(
                torch.minimum(accumulated[row - 1, column], accumulated[row, column - 1]),
                accumulated[row - 1, column - 1],
            )
            accumulated[row, column] = cost[row - 1, column - 1] + previous
    return float((accumulated[length, length] / length).item())


def trajectory_smoothness(trajectory: Tensor) -> float:
    """Mean magnitude of the discrete second difference of the tip path.

    Ref: Sec. 4.10.
    """
    if trajectory.dim() != 2:
        raise ValueError("trajectory must be frames x coordinates")
    if trajectory.shape[0] < 3:
        return 0.0
    second = trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
    return float(torch.linalg.norm(second, dim=1).mean().item())


def path_length(trajectory: Tensor) -> float:
    if trajectory.dim() != 2:
        raise ValueError("trajectory must be frames x coordinates")
    if trajectory.shape[0] < 2:
        return 0.0
    steps = trajectory[1:] - trajectory[:-1]
    return float(torch.linalg.norm(steps, dim=1).sum().item())


def path_length_efficiency(centreline: Tensor, trajectory: Tensor) -> float:
    """Ratio of centreline length to travelled length; above the ceiling it is an artifact.

    Ref: Sec. 4.10.
    """
    travelled = path_length(trajectory)
    if travelled <= 0.0:
        raise ValueError("travelled length must be positive")
    return float(path_length(centreline) / travelled)


def path_efficiency_is_artifact(efficiency: float) -> bool:
    """Values above 0.92 indicate tracking or centreline-extraction rather than a route.

    Ref: Sec. 4.10.
    """
    return efficiency > PATH_EFFICIENCY_ARTIFACT_CEILING


def deviation_from_centreline(trajectory: Tensor, centreline: Tensor) -> float:
    """Relative distance of the followed path from the clinical centreline."""
    reference_length = path_length(centreline)
    if reference_length <= 0.0:
        raise ValueError("centreline length must be positive")
    offsets = torch.cdist(trajectory.to(torch.float64), centreline.to(torch.float64)).min(dim=1).values
    return float((offsets.mean() / reference_length).item())


def motion_report(generated: Tensor, recorded: Tensor, centreline: Tensor) -> dict[str, float]:
    return {
        "ade_mm": average_displacement_error(generated, recorded),
        "fde_mm": final_displacement_error(generated, recorded),
        "dtw": dynamic_time_warping(generated, recorded),
        "smoothness": trajectory_smoothness(generated),
        "path_efficiency": path_length_efficiency(centreline, generated),
        "centreline_deviation": deviation_from_centreline(generated, centreline),
    }

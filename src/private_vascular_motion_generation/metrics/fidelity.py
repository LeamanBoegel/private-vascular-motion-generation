"""Image-fidelity measures for the generated angiographic frames.

Ref: Sec. 4.10 (FID, KID, SSIM), Table 1.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _flatten(features: Tensor) -> Tensor:
    if features.dim() == 1:
        return features.reshape(1, -1)
    return features.reshape(features.shape[0], -1)


def frechet_distance(features_real: Tensor, features_generated: Tensor) -> float:
    """Frechet inception distance between two feature clouds.

    Ref: Sec. 4.10, Table 1.
    """
    real = _flatten(features_real).to(torch.float64)
    generated = _flatten(features_generated).to(torch.float64)
    if real.shape[1] != generated.shape[1]:
        raise ValueError("feature widths must agree")
    if real.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("FID needs at least two samples per cloud")
    mean_real = real.mean(dim=0)
    mean_generated = generated.mean(dim=0)
    centred_real = real - mean_real
    centred_generated = generated - mean_generated
    covariance_real = centred_real.transpose(0, 1) @ centred_real / (real.shape[0] - 1)
    covariance_generated = centred_generated.transpose(0, 1) @ centred_generated / (generated.shape[0] - 1)
    difference = mean_real - mean_generated
    mean_term = float((difference**2).sum().item())
    product = covariance_real @ covariance_generated
    values = torch.linalg.eigvalsh((product + product.transpose(0, 1)) / 2.0)
    sqrt_trace = float(values.clamp_min(0.0).sqrt().sum().item())
    trace_term = float(torch.trace(covariance_real).item() + torch.trace(covariance_generated).item())
    return float(max(mean_term + trace_term - 2.0 * sqrt_trace, 0.0))


def polynomial_mmd(
    features_real: Tensor, features_generated: Tensor, degree: int = 3, gamma: float | None = None
) -> float:
    """Kernel inception distance as an unbiased polynomial-kernel MMD estimate.

    Ref: Sec. 4.10, Table 1.
    """
    real = _flatten(features_real).to(torch.float64)
    generated = _flatten(features_generated).to(torch.float64)
    if real.shape[1] != generated.shape[1]:
        raise ValueError("feature widths must agree")
    if real.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("KID needs at least two samples per cloud")
    scale = gamma if gamma is not None else 1.0 / real.shape[1]

    def kernel(left: Tensor, right: Tensor) -> Tensor:
        return (scale * (left @ right.transpose(0, 1)) + 1.0) ** degree

    k_rr = kernel(real, real)
    k_gg = kernel(generated, generated)
    k_rg = kernel(real, generated)
    n = real.shape[0]
    m = generated.shape[0]
    diag_rr = torch.diagonal(k_rr).sum()
    diag_gg = torch.diagonal(k_gg).sum()
    cross = k_rg.sum()
    unbiased = (k_rr.sum() - diag_rr) / (n * (n - 1)) + (k_gg.sum() - diag_gg) / (m * (m - 1)) - 2.0 * cross / (n * m)
    return float(unbiased.item())


def _to_images(tensor: Tensor) -> Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError("expected a single image or a batch of images")
    return tensor


def _gaussian_kernel(window: int, sigma: float) -> Tensor:
    if window < 1 or window % 2 == 0:
        raise ValueError("SSIM window must be a positive odd integer")
    coordinates = torch.arange(window, dtype=torch.float64) - (window - 1) / 2.0
    values = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    return values / values.sum()


def structural_similarity(image: Tensor, reference: Tensor, window: int = 7, sigma: float = 1.5) -> float:
    """Mean SSIM with a separable Gaussian window.

    Ref: Sec. 4.10, Table 1.
    """
    left = _to_images(image).to(torch.float64)
    right = _to_images(reference).to(torch.float64)
    if left.shape != right.shape:
        raise ValueError("SSIM requires matched shapes")
    if min(left.shape[-2], left.shape[-1]) < window:
        raise ValueError("window exceeds the image extent")
    kernel = _gaussian_kernel(window, sigma)
    horizontal = kernel.view(1, 1, 1, window)
    vertical = kernel.view(1, 1, window, 1)
    padding = window // 2
    data_range = float((left.max() - left.min()).item())
    if data_range <= 0.0:
        raise ValueError("degenerate intensity range")

    def blur(tensor: Tensor) -> Tensor:
        filtered = torch.nn.functional.conv2d(
            torch.nn.functional.pad(tensor, (padding, padding, 0, 0), mode="reflect"),
            horizontal.expand(tensor.shape[1], 1, 1, window),
            groups=tensor.shape[1],
        )
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(filtered, (0, 0, padding, padding), mode="reflect"),
            vertical.expand(tensor.shape[1], 1, window, 1),
            groups=tensor.shape[1],
        )

    mu_left = blur(left)
    mu_right = blur(right)
    sigma_left = blur(left * left) - mu_left * mu_left
    sigma_right = blur(right * right) - mu_right * mu_right
    sigma_cross = blur(left * right) - mu_left * mu_right
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_left**2 + mu_right**2 + c1) * (sigma_left + sigma_right + c2)
    return float((numerator / denominator).mean().item())


def fidelity_report(
    real_features: Tensor, generated_features: Tensor, image: Tensor, reference: Tensor
) -> dict[str, float]:
    return {
        "fid": frechet_distance(real_features, generated_features),
        "kid": polynomial_mmd(real_features, generated_features),
        "ssim": structural_similarity(image, reference),
    }

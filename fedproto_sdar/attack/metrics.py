"""
Evaluation metrics for SDAR attack quality.
"""

import torch
import numpy as np


def compute_mse(x, x_recon):
    """
    Compute per-image Mean Squared Error.

    Args:
        x: (batch, C, H, W) original images
        x_recon: (batch, C, H, W) reconstructed images

    Returns:
        float — average MSE across the batch
    """
    mse = torch.mean((x - x_recon) ** 2, dim=[1, 2, 3])  # per-image MSE
    return mse.mean().item()


def compute_ssim(x, x_recon, window_size=11):
    """
    Compute Structural Similarity Index (SSIM) between original and
    reconstructed images. Simplified implementation.

    Args:
        x: (batch, C, H, W) original images
        x_recon: (batch, C, H, W) reconstructed images
        window_size: size of the Gaussian window

    Returns:
        float — average SSIM across the batch
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Convert to float
    x = x.float()
    x_recon = x_recon.float()

    # Compute means using average pooling
    mu_x = F.avg_pool2d(x, window_size, stride=1,
                        padding=window_size // 2)
    mu_y = F.avg_pool2d(x_recon, window_size, stride=1,
                        padding=window_size // 2)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1,
                               padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(x_recon ** 2, window_size, stride=1,
                               padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * x_recon, window_size, stride=1,
                             padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean().item()


# Import F here to avoid circular import issues
import torch.nn.functional as F


def evaluate_attack(x_real, x_recon):
    """
    Evaluate attack quality with both MSE and SSIM.

    Args:
        x_real: (batch, C, H, W) real client images
        x_recon: (batch, C, H, W) reconstructed images

    Returns:
        dict with 'mse' and 'ssim' keys
    """
    mse = compute_mse(x_real, x_recon)
    ssim = compute_ssim(x_real, x_recon)
    return {'mse': mse, 'ssim': ssim}

"""
Evaluation metrics for SDAR attack quality.

Metrics:
    - MSE  (Mean Squared Error):  lower is better, 0 = perfect
    - PSNR (Peak Signal-to-Noise Ratio):  higher is better, ∞ = perfect
    - SSIM (Structural Similarity Index):  higher is better, 1 = perfect
    - Downstream classification accuracy:  higher = attack leaked more class info
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


# ──────────────────────────────────────────────────────────
# MSE
# ──────────────────────────────────────────────────────────

def compute_mse(x, x_recon):
    """
    Compute per-image Mean Squared Error, then average.

    Args:
        x: (B, C, H, W) original images
        x_recon: (B, C, H, W) reconstructed images

    Returns:
        float — average MSE across the batch
    """
    mse = torch.mean((x - x_recon) ** 2, dim=[1, 2, 3])  # per-image
    return mse.mean().item()


# ──────────────────────────────────────────────────────────
# PSNR  (derived from MSE)
# ──────────────────────────────────────────────────────────

def compute_psnr(x, x_recon, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        x: (B, C, H, W) original images  (values in [0, max_val])
        x_recon: (B, C, H, W) reconstructed images
        max_val: maximum pixel value (1.0 for normalised tensors)

    Returns:
        float — average PSNR in dB across the batch
    """
    mse_per_img = torch.mean((x - x_recon) ** 2, dim=[1, 2, 3])
    # Avoid log(0) by clamping
    mse_per_img = torch.clamp(mse_per_img, min=1e-10)
    psnr_per_img = 10.0 * torch.log10(max_val ** 2 / mse_per_img)
    return psnr_per_img.mean().item()


# ──────────────────────────────────────────────────────────
# SSIM  (channel-wise, Gaussian-weighted)
# ──────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size, sigma):
    """Create a 1-D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_kernel_2d(size, sigma, channels):
    """Create a 2-D Gaussian kernel for use with F.conv2d."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # outer product
    k2d = k2d.expand(channels, 1, size, size).contiguous()
    return k2d


def compute_ssim(x, x_recon, window_size=7, sigma=1.5):
    """
    Compute Structural Similarity Index (SSIM).

    Uses a Gaussian-weighted window (more robust than avg_pool for small images).
    Window size is set to 7 (instead of 11) because CIFAR-10 is only 32×32.

    Args:
        x: (B, C, H, W) original images
        x_recon: (B, C, H, W) reconstructed images
        window_size: size of the Gaussian window (should be odd)
        sigma: standard deviation of the Gaussian

    Returns:
        float — average SSIM across the batch, in [-1, 1]
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    x = x.float()
    x_recon = x_recon.float()

    channels = x.size(1)
    kernel = _gaussian_kernel_2d(window_size, sigma, channels).to(x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(x_recon, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(x_recon ** 2, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * x_recon, kernel, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # Average per image, then average across batch
    return ssim_map.mean(dim=[1, 2, 3]).mean().item()


# ──────────────────────────────────────────────────────────
# Downstream Classifier Accuracy
# ──────────────────────────────────────────────────────────

def compute_downstream_accuracy(x_recon, labels, classifier, device='cpu'):
    """
    Feed reconstructed images through an independent pre-trained classifier
    and check how often it predicts the correct class.

    If the reconstructed image of class "frog" is classified as "frog",
    it proves the prototype leaked enough visual information about that class.

    Args:
        x_recon: (B, C, H, W) reconstructed images
        labels: (B,) true class labels
        classifier: a pre-trained nn.Module with .eval() method
        device: 'cuda' or 'cpu'

    Returns:
        float — top-1 accuracy in [0, 1]
    """
    classifier.eval()
    x_recon = x_recon.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = classifier(x_recon)
        # Handle models that return (logits, features)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1)

    correct = (preds == labels).float().sum().item()
    return correct / len(labels)


# ──────────────────────────────────────────────────────────
# Per-Class MSE  (useful for tables in papers)
# ──────────────────────────────────────────────────────────

def compute_per_class_mse(reconstructions, real_class_means):
    """
    Compute MSE between each reconstructed class image and the
    real per-class mean image from the client's training data.

    Args:
        reconstructions: dict {label: (C, H, W) tensor} from attacker
        real_class_means: dict {label: (C, H, W) tensor} ground-truth averages

    Returns:
        dict {label: float MSE}
    """
    results = {}
    for label in reconstructions:
        if label in real_class_means:
            recon = reconstructions[label].float()
            real = real_class_means[label].float()
            results[label] = torch.mean((recon - real) ** 2).item()
    return results


# ──────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────

def evaluate_attack(x_real, x_recon):
    """
    Evaluate attack quality with MSE, PSNR, and SSIM.

    Args:
        x_real: (B, C, H, W) real client images (or class-mean images)
        x_recon: (B, C, H, W) reconstructed images

    Returns:
        dict with 'mse', 'psnr', 'ssim' keys
    """
    mse = compute_mse(x_real, x_recon)
    psnr = compute_psnr(x_real, x_recon)
    ssim = compute_ssim(x_real, x_recon)
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}


def compute_real_class_means(dataset, idxs, num_classes, device='cpu',
                              norm_mean=(0.4914, 0.4822, 0.4465),
                              norm_std=(0.2023, 0.1994, 0.2010)):
    """
    Compute the per-class mean image from a client's training data.
    This is the "ground truth" to compare the attacker's reconstructions against.

    Since FedProto prototypes represent class-level averages, the fairest
    comparison is against the mean image of each class (not individual images).

    Images are denormalized from the training transform back to [0,1] before
    averaging, since the decoder outputs [0,1] values.

    Args:
        dataset: the base training dataset
        idxs: list of dataset indices belonging to this client
        num_classes: total number of classes
        device: 'cuda' or 'cpu'
        norm_mean: dataset normalization mean (tuple)
        norm_std: dataset normalization std (tuple)

    Returns:
        dict {label: (C, H, W) mean image tensor, values in [0,1]}
    """
    from collections import defaultdict
    class_images = defaultdict(list)

    # Build denormalization tensors
    mean_t = torch.tensor(norm_mean).view(-1, 1, 1)  # (C, 1, 1)
    std_t = torch.tensor(norm_std).view(-1, 1, 1)    # (C, 1, 1)

    for idx in idxs:
        img, label = dataset[idx]
        if isinstance(img, torch.Tensor):
            # Denormalize: original = normalized * std + mean
            img = img * std_t + mean_t
        else:
            img = torch.tensor(img) * std_t + mean_t
        class_images[label].append(img)

    class_means = {}
    for label, imgs in class_images.items():
        stacked = torch.stack(imgs).float()
        mean_img = stacked.mean(dim=0)
        # Clamp to [0,1] for fair visual comparison
        mean_img = torch.clamp(mean_img, 0.0, 1.0)
        class_means[label] = mean_img

    return class_means

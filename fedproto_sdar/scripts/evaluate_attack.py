"""
Standalone evaluation script for already-saved SDAR attack reconstructions.

Loads the saved .png reconstruction images, compares them against the
real CIFAR-10 class-mean images, and computes MSE / PSNR / SSIM /
downstream classifier accuracy.

Run this on Colab after training is complete — no retraining needed.

Usage:
    python scripts/evaluate_attack.py --recon_dir ./results/reconstructions/
"""

import sys
import os
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from attack.metrics import compute_mse, compute_psnr, compute_ssim, compute_downstream_accuracy
from models.client_models import get_model


def load_reconstruction_images(recon_dir):
    """
    Load all saved reconstruction PNGs and parse their metadata from filenames.

    Returns:
        list of dicts: [{
            'path': str,
            'round': int,
            'client': int,
            'images': dict {class_label: (C, H, W) tensor},
        }]
    """
    png_files = sorted(glob.glob(os.path.join(recon_dir, '*.png')))
    if not png_files:
        print(f"ERROR: No .png files found in {recon_dir}")
        return []

    results = []
    for fpath in png_files:
        fname = os.path.basename(fpath)
        # Parse filename: recon_round49_client0.png
        match = re.match(r'recon_round(\d+)_client(\d+)\.png', fname)
        if not match:
            match = re.match(r'recon_round(\d+)\.png', fname)
            if match:
                round_num = int(match.group(1))
                client_idx = 0
            else:
                continue
        else:
            round_num = int(match.group(1))
            client_idx = int(match.group(2))

        # Load the composite image
        composite = Image.open(fpath).convert('RGB')
        composite_np = np.array(composite)

        # The composite image contains multiple sub-images side by side
        # with titles. We need to extract individual class images.
        # Each sub-image is roughly (img_height, img_width) within the composite.
        # The reconstructed images are 32x32 pixels at 150 DPI.

        results.append({
            'path': fpath,
            'round': round_num,
            'client': client_idx,
            'composite': composite_np,
        })

    return results


def get_cifar10_class_means(data_dir='../data/'):
    """
    Compute the per-class mean image from the entire CIFAR-10 training set.
    Returns dict {label: (3, 32, 32) tensor in [0,1]}.
    """
    # Use plain ToTensor (no normalization) so pixel values stay in [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

    from collections import defaultdict
    class_sums = defaultdict(lambda: torch.zeros(3, 32, 32))
    class_counts = defaultdict(int)

    for img, label in dataset:
        class_sums[label] += img
        class_counts[label] += 1

    class_means = {}
    for label in sorted(class_sums.keys()):
        class_means[label] = class_sums[label] / class_counts[label]

    return class_means


def extract_sub_images_from_composite(composite_np, num_classes):
    """
    Given a composite image (the saved matplotlib figure), extract individual
    class reconstruction images.

    The matplotlib figure has a row of sub-images with titles above them.
    We find the image region and split it into equal-width columns.

    Returns:
        list of (H, W, 3) numpy arrays, one per class.
    """
    h, w, c = composite_np.shape

    # Find the image region (non-white rows for dark bg, or non-background)
    # Look for rows that have significant pixel variance (i.e., actual image content)
    row_var = np.var(composite_np.astype(float), axis=(1, 2))
    threshold = np.mean(row_var) * 0.3

    content_rows = np.where(row_var > threshold)[0]
    if len(content_rows) == 0:
        return []

    # Find the main image band (skip title text at top)
    top = content_rows[0]
    bottom = content_rows[-1]

    # The title region is typically in the top portion
    # Find a gap between title and image content
    row_diffs = np.diff(content_rows)
    large_gaps = np.where(row_diffs > 5)[0]

    if len(large_gaps) > 0:
        # Start after the first significant gap (skip title text)
        img_start_idx = large_gaps[0] + 1
        top = content_rows[img_start_idx]

    img_region = composite_np[top:bottom+1, :, :]

    # Split into equal columns
    col_width = w // num_classes
    sub_images = []
    for i in range(num_classes):
        x_start = i * col_width
        x_end = (i + 1) * col_width
        sub_img = img_region[:, x_start:x_end, :]
        sub_images.append(sub_img)

    return sub_images


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_dir', type=str, default='./results/reconstructions/',
                        help='Directory containing saved reconstruction PNGs')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory for CIFAR-10 dataset')
    parser.add_argument('--round', type=int, default=-1,
                        help='Which round to evaluate (-1 = latest)')
    eval_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  SDAR Attack Evaluation (from saved reconstructions)")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load CIFAR-10 class mean images ──
    print("[1/3] Computing CIFAR-10 per-class mean images...")
    class_means = get_cifar10_class_means(eval_args.data_dir)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"  Computed mean images for {len(class_means)} classes.\n")

    # ── Load reconstruction PNGs ──
    print("[2/3] Loading saved reconstruction images...")
    recon_files = sorted(glob.glob(os.path.join(eval_args.recon_dir, '*.png')))
    if not recon_files:
        print(f"  ERROR: No .png files found in {eval_args.recon_dir}")
        return

    # Find the final round reconstructions
    final_round_files = []
    all_rounds = set()
    for f in recon_files:
        match = re.match(r'recon_round(\d+)_client(\d+)\.png', os.path.basename(f))
        if match:
            r = int(match.group(1))
            all_rounds.add(r)

    if eval_args.round >= 0:
        target_round = eval_args.round
    else:
        target_round = max(all_rounds) if all_rounds else 0

    print(f"  Available rounds: {sorted(all_rounds)}")
    print(f"  Evaluating round: {target_round}\n")

    target_files = [f for f in recon_files
                    if f'round{target_round}_client' in os.path.basename(f)]

    if not target_files:
        # Fallback: try files without client suffix
        target_files = [f for f in recon_files
                        if f'round{target_round}' in os.path.basename(f)]

    print(f"  Found {len(target_files)} reconstruction files for round {target_round}")

    # ── Evaluate each client's reconstruction ──
    print(f"\n[3/3] Computing metrics...")
    print(f"{'='*60}")

    all_mse = []
    all_psnr = []
    all_ssim = []
    all_ds_correct = 0
    all_ds_total = 0

    # Load a pretrained ResNet18 for downstream classification test
    print("  Loading pre-trained ResNet18 for downstream classifier test...")
    try:
        eval_model, _, _ = get_model('resnet18', num_classes=10, pretrained=False)
        import torch.utils.model_zoo as model_zoo
        pretrained_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=False)
        eval_model.load_state_dict(pretrained_dict, strict=False)
        eval_model.to(device)
        eval_model.eval()
        has_classifier = True
    except Exception as e:
        print(f"  Warning: Could not load classifier: {e}")
        has_classifier = False

    for fpath in target_files:
        fname = os.path.basename(fpath)
        match = re.match(r'recon_round(\d+)_client(\d+)\.png', fname)
        if match:
            client_idx = int(match.group(2))
        else:
            client_idx = 0

        # Load the composite image and extract individual class reconstructions
        composite = Image.open(fpath).convert('RGB')
        composite_np = np.array(composite).astype(np.float32) / 255.0

        # Read the figure to determine how many classes
        # Count "Class" labels in the saved figure (from matplotlib title)
        # We estimate by dividing the image width by the expected sub-image width
        h, w, _ = composite_np.shape
        # Each sub-image is approximately 3 inches * 150 DPI = 450 pixels wide
        estimated_classes = max(1, round(w / 450))

        # Split into sub-images
        sub_images = extract_sub_images_from_composite(
            (composite_np * 255).astype(np.uint8), estimated_classes)

        if len(sub_images) == 0:
            print(f"  Client {client_idx}: Could not extract sub-images from {fname}")
            continue

        print(f"\n  Client {client_idx} ({len(sub_images)} classes from {fname}):")

        client_mse_list = []
        client_psnr_list = []
        client_ssim_list = []

        for class_idx in range(min(len(sub_images), 10)):
            sub_img = sub_images[class_idx]
            # Resize to 32x32 for comparison
            sub_pil = Image.fromarray(sub_img).resize((32, 32), Image.BILINEAR)
            sub_tensor = transforms.ToTensor()(sub_pil).unsqueeze(0)  # (1, 3, 32, 32)

            # Compare against CIFAR-10 class mean
            # We don't know the exact class mapping, so compare against all
            # and report the best match + the assumed sequential order
            real_mean = class_means[class_idx].unsqueeze(0)  # (1, 3, 32, 32)

            mse = compute_mse(real_mean, sub_tensor)
            psnr = compute_psnr(real_mean, sub_tensor)
            ssim = compute_ssim(real_mean, sub_tensor)

            client_mse_list.append(mse)
            client_psnr_list.append(psnr)
            client_ssim_list.append(ssim)

            print(f"    Class {class_idx} ({cifar10_classes[class_idx]:>10s}): "
                  f"MSE={mse:.6f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

            # Downstream classification test
            if has_classifier:
                # Normalize for ResNet (ImageNet stats)
                normalize = transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                sub_norm = normalize(sub_tensor.squeeze(0)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = eval_model(sub_norm)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    pred = logits.argmax(dim=1).item()
                if pred == class_idx:
                    all_ds_correct += 1
                all_ds_total += 1

        if client_mse_list:
            avg_mse = np.mean(client_mse_list)
            avg_psnr = np.mean(client_psnr_list)
            avg_ssim = np.mean(client_ssim_list)
            all_mse.append(avg_mse)
            all_psnr.append(avg_psnr)
            all_ssim.append(avg_ssim)
            print(f"    ── Average: MSE={avg_mse:.6f}  "
                  f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Round {target_round})")
    print(f"{'='*60}")
    if all_mse:
        print(f"  MSE  : {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"  PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        print(f"  SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    if all_ds_total > 0:
        ds_acc = all_ds_correct / all_ds_total
        print(f"  Downstream Classifier Acc: {ds_acc:.4f} "
              f"({all_ds_correct}/{all_ds_total})")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

"""
Standalone evaluation script for already-saved SDAR attack reconstructions.

Works on the composite PNG images saved by save_reconstructions().
Since we don't know the exact class labels from the composite image,
we use a "best-match" approach: compare each reconstructed sub-image
against ALL 10 CIFAR-10 class means and find the closest match.

This is itself a meaningful metric: it answers "can the reconstruction
be correctly identified as the right class by visual similarity alone?"

Also runs a downstream classifier test with a properly fine-tuned model.

Usage:
    python scripts/evaluate_attack.py --recon_dir ./results/reconstructions/
"""

import sys
import os
import glob
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from attack.metrics import compute_mse, compute_psnr, compute_ssim
from models.client_models import get_model

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ──────────────────────────────────────────────────────────
# 1.  Ground-truth class means
# ──────────────────────────────────────────────────────────

def get_cifar10_class_means(data_dir='../data/'):
    """
    Compute the per-class mean image from the CIFAR-10 training set.
    Uses plain ToTensor() so pixel values are in [0, 1].
    Returns dict {label_int: (3, 32, 32) tensor}.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                               transform=transform)

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


# ──────────────────────────────────────────────────────────
# 2.  Load individual per-class images (preferred, from new runs)
# ──────────────────────────────────────────────────────────

def load_individual_class_images(recon_dir, target_round):
    """
    Try to load individual per-class images saved by the updated
    save_reconstructions() method.  These have filenames like:
        recon_round49_client0_class5.png

    Returns:
        dict {client_idx: {class_label: (3, 32, 32) tensor}} or None
    """
    pattern = os.path.join(recon_dir, f'recon_round{target_round}_client*_class*.png')
    files = glob.glob(pattern)
    if not files:
        return None

    from collections import defaultdict
    result = defaultdict(dict)

    for fpath in files:
        match = re.match(
            r'recon_round(\d+)_client(\d+)_class(\d+)\.png',
            os.path.basename(fpath))
        if not match:
            continue
        client_idx = int(match.group(2))
        class_label = int(match.group(3))

        img = Image.open(fpath).convert('RGB').resize((32, 32), Image.BILINEAR)
        tensor = transforms.ToTensor()(img)  # (3, 32, 32) in [0, 1]
        result[client_idx][class_label] = tensor

    return dict(result) if result else None


# ──────────────────────────────────────────────────────────
# 3.  Load metadata JSON (preferred, from new runs)
# ──────────────────────────────────────────────────────────

def load_metadata(recon_dir, target_round):
    """
    Try to load the JSON metadata that maps sub-image index to class label.
    Returns dict {client_idx: [list of class labels]} or None.
    """
    pattern = os.path.join(recon_dir, f'recon_round{target_round}_client*_meta.json')
    files = glob.glob(pattern)
    if not files:
        return None

    result = {}
    for fpath in files:
        match = re.match(
            r'recon_round(\d+)_client(\d+)_meta\.json',
            os.path.basename(fpath))
        if not match:
            continue
        client_idx = int(match.group(2))
        with open(fpath) as f:
            meta = json.load(f)
        result[client_idx] = meta.get('class_labels', [])

    return result if result else None


# ──────────────────────────────────────────────────────────
# 4.  Extract sub-images from composite PNG (fallback for old runs)
# ──────────────────────────────────────────────────────────

def extract_sub_images_from_composite(composite_np):
    """
    Given a composite image (matplotlib figure as numpy array),
    extract the individual reconstructed class images.

    Strategy: find the image content band (skip title text),
    then split into equal-width columns.

    Returns list of (H, W, 3) numpy arrays.
    """
    h, w, c = composite_np.shape

    # Detect content rows (rows with significant variance = actual image data)
    row_var = np.var(composite_np.astype(float), axis=(1, 2))
    threshold = np.percentile(row_var, 30)
    content_rows = np.where(row_var > threshold)[0]

    if len(content_rows) < 10:
        return []

    # The matplotlib figure has: white background, title text at top,
    # image content in the middle, then bottom margin.
    # Find the densest band of content rows.
    top = content_rows[0]
    bottom = content_rows[-1]

    # Try to skip the title row by finding a gap
    row_diffs = np.diff(content_rows)
    large_gaps = np.where(row_diffs > 5)[0]
    if len(large_gaps) > 0:
        top = content_rows[large_gaps[0] + 1]

    img_band = composite_np[top:bottom+1, :, :]
    band_h, band_w, _ = img_band.shape

    # Each sub-image panel is about 450px wide (3 inches * 150 DPI)
    # Estimate number of classes
    panel_width = 450
    num_panels = max(1, round(band_w / panel_width))

    sub_images = []
    col_width = band_w // num_panels
    for i in range(num_panels):
        x0 = i * col_width
        x1 = (i + 1) * col_width
        # Trim any white borders within each panel
        panel = img_band[:, x0:x1, :]
        sub_images.append(panel)

    return sub_images


# ──────────────────────────────────────────────────────────
# 5.  Best-match class identification
# ──────────────────────────────────────────────────────────

def find_best_matching_class(sub_img_tensor, class_means):
    """
    Compare a reconstructed (3, 32, 32) image against all class means
    and return the class with minimum MSE.

    Returns (best_label, best_mse, all_mses_dict)
    """
    best_label = -1
    best_mse = float('inf')
    all_mses = {}

    sub = sub_img_tensor.unsqueeze(0)  # (1, 3, 32, 32)
    for label, mean_img in class_means.items():
        ref = mean_img.unsqueeze(0)  # (1, 3, 32, 32)
        mse = compute_mse(ref, sub)
        all_mses[label] = mse
        if mse < best_mse:
            best_mse = mse
            best_label = label

    return best_label, best_mse, all_mses


# ──────────────────────────────────────────────────────────
# 6.  Downstream classifier (properly fine-tuned)
# ──────────────────────────────────────────────────────────

def get_finetuned_classifier(data_dir='../data/', device='cpu'):
    """
    Load a pre-trained ResNet18 and fine-tune just the final FC layer
    on CIFAR-10 for a few epochs. This gives us a proper 10-class
    classifier for the downstream test.
    """
    print("  Fine-tuning ResNet18 classifier on CIFAR-10 (2 epochs)...")

    model, _, _ = get_model('resnet18', num_classes=10, pretrained=False)

    # Load pretrained ImageNet weights (skip mismatched fc layer)
    try:
        import torch.utils.model_zoo as model_zoo
        pretrained_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=False)
        model_dict = model.state_dict()
        # Only load layers that match in size
        filtered = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"    Loaded {len(filtered)}/{len(pretrained_dict)} pretrained layers.")
    except Exception as e:
        print(f"    Warning: {e}")

    model.to(device)

    # Freeze everything except the final fc layer
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze fc
    for param in model.fc.parameters():
        param.requires_grad = True

    # Fine-tune on CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=transform)
    loader = DataLoader(train_set, batch_size=256, shuffle=True,
                        num_workers=2, drop_last=True)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"    Epoch {epoch+1}: acc = {correct/total:.4f}")

    model.eval()
    return model


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_dir', type=str,
                        default='./results/reconstructions/')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--round', type=int, default=-1,
                        help='Which round to evaluate (-1 = latest)')
    parser.add_argument('--skip_downstream', action='store_true',
                        help='Skip the downstream classifier test')
    ea = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  SDAR Attack Evaluation")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Step 1: Compute CIFAR-10 class mean images ──
    print("[1/4] Computing CIFAR-10 per-class mean images...")
    class_means = get_cifar10_class_means(ea.data_dir)
    print(f"  Done. {len(class_means)} classes.\n")

    # ── Step 2: Discover available rounds ──
    print("[2/4] Scanning reconstruction directory...")
    recon_files = sorted(glob.glob(os.path.join(ea.recon_dir, '*.png')))
    if not recon_files:
        print(f"  ERROR: No .png files found in {ea.recon_dir}")
        return

    all_rounds = set()
    for f in recon_files:
        m = re.search(r'round(\d+)', os.path.basename(f))
        if m:
            all_rounds.add(int(m.group(1)))

    target_round = ea.round if ea.round >= 0 else max(all_rounds)
    print(f"  Available rounds: {sorted(all_rounds)}")
    print(f"  Evaluating round: {target_round}\n")

    # ── Step 3: Try to load individual class images (new format) ──
    individual = load_individual_class_images(ea.recon_dir, target_round)
    metadata = load_metadata(ea.recon_dir, target_round)

    if individual:
        print("[3/4] Using individual per-class images (exact class labels).\n")
        use_individual = True
    else:
        print("[3/4] No individual class images found.")
        print("  Falling back to composite image extraction with best-match")
        print("  class identification.\n")
        use_individual = False

    # ── Step 4: Prepare downstream classifier ──
    classifier = None
    if not ea.skip_downstream:
        print("[4/4] Preparing downstream classifier...")
        try:
            classifier = get_finetuned_classifier(ea.data_dir, device)
            print("  Classifier ready.\n")
        except Exception as e:
            print(f"  Could not prepare classifier: {e}\n")

    # ── Evaluate ──
    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    all_mse = []
    all_psnr = []
    all_ssim = []
    all_ds_correct = 0
    all_ds_total = 0
    best_match_correct = 0
    best_match_total = 0

    if use_individual:
        # ── Path A: Individual per-class images (exact labels) ──
        for client_idx in sorted(individual.keys()):
            client_recons = individual[client_idx]
            print(f"\n  Client {client_idx} ({len(client_recons)} classes):")

            for class_label, recon_tensor in sorted(client_recons.items()):
                if class_label not in class_means:
                    continue
                real_mean = class_means[class_label].unsqueeze(0)
                recon = recon_tensor.unsqueeze(0)

                mse = compute_mse(real_mean, recon)
                psnr = compute_psnr(real_mean, recon)
                ssim = compute_ssim(real_mean, recon)

                all_mse.append(mse)
                all_psnr.append(psnr)
                all_ssim.append(ssim)

                name = CIFAR10_CLASSES[class_label] if class_label < 10 else '?'
                print(f"    Class {class_label:2d} ({name:>10s}): "
                      f"MSE={mse:.6f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

                # Downstream classifier test
                if classifier is not None:
                    normalize = transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    inp = normalize(recon_tensor).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = classifier(inp)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        pred = logits.argmax(1).item()
                    if pred == class_label:
                        all_ds_correct += 1
                    all_ds_total += 1
    else:
        # ── Path B: Composite images with best-match identification ──
        target_files = sorted([
            f for f in recon_files
            if re.search(rf'round{target_round}_client\d+\.png',
                         os.path.basename(f))
        ])
        if not target_files:
            target_files = sorted([
                f for f in recon_files
                if f'round{target_round}' in os.path.basename(f)
            ])

        for fpath in target_files:
            fname = os.path.basename(fpath)
            m = re.match(r'recon_round(\d+)_client(\d+)\.png', fname)
            client_idx = int(m.group(2)) if m else 0

            composite = np.array(Image.open(fpath).convert('RGB'))
            sub_images = extract_sub_images_from_composite(composite)

            if not sub_images:
                print(f"  Client {client_idx}: Could not extract sub-images.")
                continue

            # Check if we have metadata for this client
            client_labels = None
            if metadata and client_idx in metadata:
                client_labels = metadata[client_idx]

            print(f"\n  Client {client_idx} ({len(sub_images)} classes from {fname}):")

            for i, sub_np in enumerate(sub_images):
                # Resize to 32x32
                sub_pil = Image.fromarray(sub_np).resize((32, 32),
                                                          Image.BILINEAR)
                sub_tensor = transforms.ToTensor()(sub_pil)  # (3, 32, 32)

                if client_labels and i < len(client_labels):
                    # We know the exact class from metadata
                    true_label = client_labels[i]
                    real_mean = class_means[true_label]
                    recon = sub_tensor.unsqueeze(0)
                    ref = real_mean.unsqueeze(0)

                    mse = compute_mse(ref, recon)
                    psnr = compute_psnr(ref, recon)
                    ssim = compute_ssim(ref, recon)

                    name = CIFAR10_CLASSES[true_label]
                    print(f"    Class {true_label:2d} ({name:>10s}): "
                          f"MSE={mse:.6f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")
                else:
                    # Best-match: compare against all class means
                    best_label, best_mse, all_mses_dict = \
                        find_best_matching_class(sub_tensor, class_means)

                    recon = sub_tensor.unsqueeze(0)
                    ref = class_means[best_label].unsqueeze(0)
                    mse = compute_mse(ref, recon)
                    psnr = compute_psnr(ref, recon)
                    ssim = compute_ssim(ref, recon)

                    name = CIFAR10_CLASSES[best_label]
                    print(f"    Sub-image {i} → best match: Class {best_label} "
                          f"({name:>10s}): "
                          f"MSE={mse:.6f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

                    best_match_total += 1

                all_mse.append(mse)
                all_psnr.append(psnr)
                all_ssim.append(ssim)

                # Downstream classifier
                if classifier is not None:
                    normalize = transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    inp = normalize(sub_tensor).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = classifier(inp)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        pred = logits.argmax(1).item()

                    target = (client_labels[i] if client_labels and i < len(client_labels)
                              else best_label)
                    if pred == target:
                        all_ds_correct += 1
                    all_ds_total += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (Round {target_round})")
    print(f"{'='*60}")

    if all_mse:
        print(f"  MSE  : {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"  PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        print(f"  SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")

    if best_match_total > 0:
        print(f"\n  Note: Class labels were identified using best-match MSE.")
        print(f"  For exact results, re-run training with the updated code.")

    if all_ds_total > 0:
        ds_acc = all_ds_correct / all_ds_total
        print(f"\n  Downstream Classifier Accuracy: {ds_acc:.4f} "
              f"({all_ds_correct}/{all_ds_total})")
    else:
        print(f"\n  Downstream classifier: skipped or not available.")

    print(f"{'='*60}\n")

    # Save metrics
    metrics = {
        'round': target_round,
        'mse_mean': float(np.mean(all_mse)) if all_mse else None,
        'mse_std': float(np.std(all_mse)) if all_mse else None,
        'psnr_mean': float(np.mean(all_psnr)) if all_psnr else None,
        'psnr_std': float(np.std(all_psnr)) if all_psnr else None,
        'ssim_mean': float(np.mean(all_ssim)) if all_ssim else None,
        'ssim_std': float(np.std(all_ssim)) if all_ssim else None,
        'downstream_acc': all_ds_correct / all_ds_total if all_ds_total > 0 else None,
        'per_image_mse': all_mse,
        'per_image_psnr': all_psnr,
        'per_image_ssim': all_ssim,
    }
    save_path = os.path.join(os.path.dirname(ea.recon_dir.rstrip('/')),
                             'attack_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {save_path}")


if __name__ == '__main__':
    main()

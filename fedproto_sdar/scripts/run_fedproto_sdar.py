"""
FedProto training with SDAR attack.

The server runs the SDAR attack at each round:
    1. Normal FedProto: clients train locally, send prototypes
    2. Server aggregates prototypes (standard FedProto)
    3. Server trains SDAR attack models (simulator + decoder + discriminators)
    4. Server reconstructs class-representative images from client prototypes
    5. Server sends global prototypes back to clients

Usage:
    python run_fedproto_sdar.py --dataset cifar10 --model resnet18 \
        --num_users 20 --rounds 50 --ways 5 --shots 100 --ld 0.1 \
        --local_bs 32 --attack --attack_epochs 5 --lambda1 0.02 --lambda2 1e-5
"""

import sys
import os
import copy
import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_args
from data.data_loader import get_dataset
from models.client_models import get_model
from fedproto.client import FedProtoClient, FedProtoLocalTest
from fedproto.proto_utils import agg_func
from fedproto.server import FedProtoServer
from attack.sdar_attacker import SDARAttackerFedProto
from attack.metrics import (evaluate_attack, compute_per_class_mse,
                             compute_real_class_means, compute_downstream_accuracy,
                             denormalize)


def main():
    args = get_args()
    args.attack = True  # Force attack mode

    # ── Setup ──
    args.device = 'cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  FedProto + SDAR Attack")
    print(f"  Dataset: {args.dataset} | Model: {args.model}")
    print(f"  Users: {args.num_users} | Rounds: {args.rounds}")
    print(f"  Ways: {args.ways} | Shots: {args.shots} | ld: {args.ld}")
    print(f"  Attack: λ1={args.lambda1} λ2={args.lambda2} "
          f"epochs={args.attack_epochs}")
    print(f"  Conditional: {args.conditional}")
    print(f"  No Proto Avg (ablation): {args.no_proto_avg}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # ── Generate non-IID config per user ──
    n_list = np.random.randint(
        max(2, args.ways - args.stdev),
        min(args.num_classes, args.ways + args.stdev + 1),
        args.num_users)
    k_list = np.random.randint(
        args.shots - args.stdev + 1,
        args.shots + args.stdev + 1,
        args.num_users)

    # ── Load dataset ──
    (train_dataset, test_dataset, user_groups, user_groups_lt,
     classes_list, server_aux_idxs) = get_dataset(args, n_list, k_list)

    # ── Build local models ──
    local_model_list = []
    for i in range(args.num_users):
        model, proto_dim, proto_spatial = get_model(
            args.model, num_classes=args.num_classes, pretrained=False)

        # Load pretrained ImageNet weights for ResNet18
        if args.model == 'resnet18':
            try:
                import torch.utils.model_zoo as model_zoo
                pretrained_dict = model_zoo.load_url(
                    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    progress=False)
                model_dict = model.state_dict()
                for key in pretrained_dict.keys():
                    if key.startswith('fc.') or key.startswith('conv1') or key.startswith('bn1'):
                        pretrained_dict[key] = model_dict[key]
                model.load_state_dict(pretrained_dict)
            except Exception as e:
                print(f"  Warning: Could not load pretrained weights: {e}")

        model.to(args.device)
        model.train()
        local_model_list.append(model)

    # ── Initialize SDAR attacker ──
    attacker = SDARAttackerFedProto(
        args=args,
        server_aux_dataset=train_dataset,
        server_aux_idxs=server_aux_idxs,
        device=args.device
    )
    print(f"\n[SDAR] Attacker initialized with {len(server_aux_idxs)} "
          f"auxiliary samples")

    # ── Initialize server with attacker ──
    server = FedProtoServer(args, attacker=attacker)

    # ── Create results directories ──
    os.makedirs(args.results_dir, exist_ok=True)
    recon_dir = os.path.join(args.results_dir, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)

    # ── Training loop ──
    train_loss_history = []
    attack_log_history = []
    start_time = time.time()

    for round_num in tqdm(range(args.rounds), desc="FedProto+SDAR"):
        local_weights = []
        local_losses = []
        local_protos = {}      # averaged protos (for FedProto FL)
        raw_protos = {}        # individual protos (for attacker, if --no_proto_avg)

        print(f'\n | Global Training Round : {round_num + 1} |')

        global_protos = server.get_global_protos()

        # ── Client local training ──
        for idx in range(args.num_users):
            client = FedProtoClient(
                args=args, dataset=train_dataset, idxs=user_groups[idx])

            w, loss, acc, protos = client.local_train(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round_num)

            # If ablation mode: keep the raw individual protos for the attacker
            if args.no_proto_avg:
                raw_protos[idx] = copy.deepcopy(protos)

            # Always average for FedProto FL (keeps accuracy identical)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

        # Update local models
        for idx in range(args.num_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[idx], strict=True)
            local_model_list[idx] = local_model

        # ── Server: aggregate + attack ──
        # Pass raw_protos to attacker when in ablation mode;
        # FedProto aggregation always uses averaged local_protos.
        global_protos, attack_results = server.receive_and_aggregate(
            local_protos, round_num,
            raw_protos=raw_protos if args.no_proto_avg else None)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss_history.append(loss_avg)

        # ── Save attack reconstructions periodically ──
        if attack_results and 'reconstructions' in attack_results:
            if (round_num + 1) % 10 == 0 or round_num == 0:
                # Save first client's reconstructions
                first_client = list(attack_results['reconstructions'].keys())[0]
                recons = attack_results['reconstructions'][first_client]
                attacker.save_reconstructions(
                    recons, recon_dir, round_num,
                    client_idx=first_client)

        if 'train_log' in attack_results:
            attack_log_history.append(attack_results['train_log'])

        print(f'  Round {round_num + 1} | Avg Loss: {loss_avg:.4f}')

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed:.1f}s ({elapsed/args.rounds:.1f}s/round)")
    print(f"{'='*60}")

    # ── Test ──
    print("\n[Testing local models...]")
    acc_list = []
    for idx in range(args.num_users):
        local_test = FedProtoLocalTest(
            args=args, dataset=test_dataset, idxs=user_groups_lt[idx])
        loss, acc = local_test.test(args, idx, classes_list[idx],
                                     local_model_list[idx])
        acc_list.append(acc)

    print(f"\n  Mean test accuracy: {np.mean(acc_list):.4f} "
          f"± {np.std(acc_list):.4f}")

    # ── Final attack reconstruction + evaluation ──
    print(f"\n{'='*60}")
    print(f"  ATTACK EVALUATION")
    print(f"{'='*60}")

    eval_clients = min(5, args.num_users)
    all_mse = []
    all_psnr = []
    all_ssim = []
    all_per_class_mse = {}
    all_downstream_acc = []

    if args.no_proto_avg:
        # ── ABLATION MODE: compare individual reconstructions vs individual source images ──
        print("\n  [ABLATION] Evaluating with individual (non-averaged) prototypes")
        print("  Evaluating Client 0 only, 10 images per class...")

        eval_client_idx = 0
        client_raw = raw_protos.get(eval_client_idx, {})
        n_per_class = 10  # evaluate 10 random images per class

        ablation_recon_all = []
        ablation_real_all = []
        ablation_labels_all = []

        # Get the real images for this client (denormalized to [0,1])
        from collections import defaultdict
        client_idxs = list(user_groups[eval_client_idx])
        real_images_by_class = defaultdict(list)
        for didx in client_idxs:
            img, lbl = train_dataset[didx]
            if isinstance(img, torch.Tensor):
                img = denormalize(img, dataset=args.dataset)
            real_images_by_class[lbl].append(img)

        for label, proto_list in client_raw.items():
            if not isinstance(proto_list, list):
                proto_list = [proto_list]
            real_imgs = real_images_by_class.get(label, [])

            # Take min(n_per_class, available) samples
            n_eval = min(n_per_class, len(proto_list), len(real_imgs))
            if n_eval == 0:
                continue

            # Sample random indices
            sample_indices = np.random.choice(
                min(len(proto_list), len(real_imgs)), n_eval, replace=False)

            for si in sample_indices:
                proto_tensor = proto_list[si]
                if isinstance(proto_tensor, torch.Tensor):
                    proto_tensor = proto_tensor.detach()
                ablation_recon_all.append(proto_tensor)
                ablation_real_all.append(real_imgs[si])
                ablation_labels_all.append(label)

        if len(ablation_recon_all) > 0:
            # Stack prototypes and reconstruct them all at once
            proto_batch = torch.stack(ablation_recon_all)
            label_batch = torch.tensor(ablation_labels_all, dtype=torch.long)
            recon_batch = attacker.attack_batch(proto_batch, label_batch)

            # Stack the real images
            real_batch = torch.stack(ablation_real_all)

            # Compute metrics
            metrics = evaluate_attack(real_batch, recon_batch)
            all_mse.append(metrics['mse'])
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])

            print(f"\n  Client {eval_client_idx} ({len(set(ablation_labels_all))} classes, "
                  f"{len(ablation_recon_all)} individual images):")
            print(f"    MSE  = {metrics['mse']:.6f}")
            print(f"    PSNR = {metrics['psnr']:.2f} dB")
            print(f"    SSIM = {metrics['ssim']:.4f}")

            # Per-class breakdown
            for lbl in sorted(set(ablation_labels_all)):
                mask = [i for i, l in enumerate(ablation_labels_all) if l == lbl]
                class_recon = recon_batch[mask]
                class_real = real_batch[mask]
                class_mse = torch.mean((class_recon - class_real) ** 2).item()
                all_per_class_mse.setdefault(eval_client_idx, {})[lbl] = class_mse
                print(f"      Class {lbl}: MSE = {class_mse:.6f} ({len(mask)} images)")

            # Save a composite image of some reconstructions vs originals
            n_show = min(10, len(ablation_recon_all))
            fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
            for i in range(n_show):
                axes[0, i].imshow(real_batch[i].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[0, i].set_title(f'Real c{ablation_labels_all[i]}')
                axes[0, i].axis('off')
                axes[1, i].imshow(recon_batch[i].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[1, i].set_title(f'Recon c{ablation_labels_all[i]}')
                axes[1, i].axis('off')
            plt.suptitle('Ablation: Individual Proto Reconstructions vs Real')
            plt.tight_layout()
            save_fig_path = os.path.join(recon_dir, 'ablation_individual_comparison.png')
            plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  Saved comparison figure to {save_fig_path}")
        else:
            print("  No individual prototypes available for evaluation.")

    else:
        # ── ORIGINAL MODE: compare class-averaged reconstructions vs class means ──
        for idx in range(eval_clients):
            # Reconstruct
            recons = attacker.attack(local_protos[idx], client_idx=idx)
            attacker.save_reconstructions(recons, recon_dir, args.rounds - 1,
                                           client_idx=idx)

            # Compute ground-truth class mean images for this client
            real_class_means = compute_real_class_means(
                train_dataset, user_groups[idx], args.num_classes, device=args.device)

            # Compute per-class MSE
            per_class_mse = compute_per_class_mse(recons, real_class_means)
            all_per_class_mse[idx] = per_class_mse

            # Build batched tensors for MSE / PSNR / SSIM
            shared_labels = sorted(set(recons.keys()) & set(real_class_means.keys()))
            if len(shared_labels) > 0:
                recon_batch = torch.stack([recons[l] for l in shared_labels])
                real_batch = torch.stack([real_class_means[l] for l in shared_labels])

                metrics = evaluate_attack(real_batch, recon_batch)
                all_mse.append(metrics['mse'])
                all_psnr.append(metrics['psnr'])
                all_ssim.append(metrics['ssim'])

                print(f"\n  Client {idx} ({len(shared_labels)} classes):")
                print(f"    MSE  = {metrics['mse']:.6f}")
                print(f"    PSNR = {metrics['psnr']:.2f} dB")
                print(f"    SSIM = {metrics['ssim']:.4f}")
                for l in shared_labels:
                    print(f"      Class {l}: MSE = {per_class_mse.get(l, 'N/A'):.6f}")

    # ── Downstream classifier accuracy ──
    print(f"\n  --- Downstream Classifier Test ---")
    try:
        # Build a ResNet18 with 10 outputs and fine-tune the fc layer on CIFAR-10
        eval_model, _, _ = get_model(args.model, num_classes=args.num_classes, pretrained=False)
        if args.model == 'resnet18':
            try:
                import torch.utils.model_zoo as model_zoo
                pretrained_dict = model_zoo.load_url(
                    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    progress=False)
                model_dict = eval_model.state_dict()
                # Only load layers that match in size (skip fc layer)
                filtered = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered)
                eval_model.load_state_dict(model_dict)
                print(f"  Loaded {len(filtered)}/{len(pretrained_dict)} pretrained layers.")
            except Exception:
                pass

        eval_model.to(args.device)

        # Freeze all layers except fc
        for param in eval_model.parameters():
            param.requires_grad = False
        for param in eval_model.fc.parameters():
            param.requires_grad = True

        # Fine-tune fc on CIFAR-10 for 2 epochs
        print("  Fine-tuning classifier on CIFAR-10 (2 epochs)...")
        from torch.utils.data import DataLoader
        ft_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        ft_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                       transform=ft_transform)
        ft_loader = DataLoader(ft_dataset, batch_size=256, shuffle=True,
                               num_workers=2, drop_last=True)
        ft_optimizer = torch.optim.Adam(eval_model.fc.parameters(), lr=0.001)
        ft_criterion = torch.nn.CrossEntropyLoss()

        eval_model.train()
        for epoch in range(2):
            correct = 0
            total = 0
            for x, y in ft_loader:
                x, y = x.to(args.device), y.to(args.device)
                out = eval_model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = ft_criterion(out, y)
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            print(f"    Epoch {epoch+1}: acc = {correct/total:.4f}")

        eval_model.eval()

        # Collect all reconstructions and their labels
        all_recon_imgs = []
        all_recon_labels = []
        for idx in range(eval_clients):
            recons = attacker.attack(local_protos[idx], client_idx=idx)
            for label, img in recons.items():
                all_recon_imgs.append(img)
                all_recon_labels.append(label)

        if len(all_recon_imgs) > 0:
            recon_batch = torch.stack(all_recon_imgs)
            label_batch = torch.tensor(all_recon_labels, dtype=torch.long)

            ds_acc = compute_downstream_accuracy(
                recon_batch, label_batch, eval_model, device=args.device)
            all_downstream_acc.append(ds_acc)
            print(f"  Downstream accuracy: {ds_acc:.4f} "
                  f"({int(ds_acc * len(label_batch))}/{len(label_batch)} correct)")
        else:
            print("  No reconstructions available for downstream test.")
    except Exception as e:
        print(f"  Downstream classifier test skipped: {e}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  FedProto Test Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"  ──────────────────────────────────")
    print(f"  Attack Metrics (avg over {eval_clients} clients):")
    if all_mse:
        print(f"    MSE  : {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"    PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        print(f"    SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    if all_downstream_acc:
        print(f"    Downstream Acc : {np.mean(all_downstream_acc):.4f}")
    print(f"{'='*60}\n")

    # ── Save results ──
    np.save(os.path.join(args.results_dir, 'sdar_loss.npy'),
            np.array(train_loss_history))
    np.save(os.path.join(args.results_dir, 'sdar_acc.npy'),
            np.array(acc_list))

    # Save attack metrics
    attack_metrics = {
        'mse': all_mse,
        'psnr': all_psnr,
        'ssim': all_ssim,
        'downstream_acc': all_downstream_acc,
        'per_class_mse': {str(k): v for k, v in all_per_class_mse.items()},
    }
    np.save(os.path.join(args.results_dir, 'attack_metrics.npy'),
            attack_metrics, allow_pickle=True)

    print(f"  Results saved to {args.results_dir}")


if __name__ == '__main__':
    main()

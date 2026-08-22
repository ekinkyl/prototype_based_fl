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
from pathlib import Path
from tqdm import tqdm

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
        local_protos = {}

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
        global_protos, attack_results = server.receive_and_aggregate(
            local_protos, round_num)

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

    # ── Final attack evaluation ──
    print("\n[Final attack reconstruction for all clients...]")
    for idx in range(min(5, args.num_users)):  # First 5 clients
        recons = attacker.attack(local_protos[idx], client_idx=idx)
        attacker.save_reconstructions(recons, recon_dir, args.rounds - 1,
                                       client_idx=idx)
        print(f"  Client {idx}: reconstructed {len(recons)} classes")

    # ── Save results ──
    np.save(os.path.join(args.results_dir, 'sdar_loss.npy'),
            np.array(train_loss_history))
    np.save(os.path.join(args.results_dir, 'sdar_acc.npy'),
            np.array(acc_list))
    print(f"\n  Results saved to {args.results_dir}")


if __name__ == '__main__':
    main()

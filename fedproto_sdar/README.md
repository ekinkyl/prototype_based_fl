# SDAR Attack on FedProto

Implementation of the **SDAR (Split-learning Data Reconstruction via Adversarial Regularization)** attack adapted for the **FedProto** prototype-based federated learning framework.

## Overview

This project adapts the SDAR passive inference attack — originally designed for Split Learning — to work against FedProto, where clients share per-class prototype embeddings instead of intermediate activations.

**Key difference from original SDAR:** In FedProto, there is **no server model**. The server only receives aggregated per-class prototypes. The attack trains a simulator (full model copy), decoder, and discriminators on the server's auxiliary data to reconstruct class-representative images from the received prototypes.

## Project Structure

```
fedproto/
├── config.py                  # Hyperparameters (CLI arguments)
├── data/data_loader.py        # CIFAR-10 loading + non-IID partitioning
├── models/
│   ├── client_models.py       # ResNet18 client model
│   └── attacker_models.py     # Decoder + discriminators
├── fedproto/
│   ├── proto_utils.py         # Prototype aggregation
│   ├── client.py              # FedProto local training
│   └── server.py              # Server + attack orchestration
├── attack/
│   ├── sdar_attacker.py       # Core SDAR attack engine
│   └── metrics.py             # MSE, SSIM evaluation
└── scripts/
    ├── run_fedproto_baseline.py  # FedProto without attack
    └── run_fedproto_sdar.py      # FedProto with SDAR attack
```

## Requirements

```
pip install torch torchvision numpy matplotlib tensorboardX tqdm scikit-learn
```

## Usage

### Baseline FedProto (no attack)

```bash
cd scripts
python run_fedproto_baseline.py --dataset cifar10 --model resnet18 \
    --num_users 20 --rounds 50 --ways 5 --shots 100 --ld 0.1 --local_bs 32
```

### FedProto with SDAR Attack

```bash
cd scripts
python run_fedproto_sdar.py --dataset cifar10 --model resnet18 \
    --num_users 20 --rounds 50 --ways 5 --shots 100 --ld 0.1 --local_bs 32 \
    --attack --attack_epochs 5 --lambda1 0.02 --lambda2 1e-5
```

## Attack Components

| Component | Role |
|-----------|------|
| **Simulator** | Full copy of client model, trained on server's auxiliary data |
| **Decoder** | Maps prototype embedding → reconstructed image |
| **Simulator Discriminator** | Ensures simulator prototypes match client prototype distribution |
| **Decoder Discriminator** | Ensures decoded images look realistic |

## References

- **SDAR:** Zhu et al., "Passive Inference Attacks on Split Learning via Adversarial Regularization", NDSS 2025
- **FedProto:** Tan et al., "FedProto: Federated Prototype Learning across Heterogeneous Clients", AAAI 2022

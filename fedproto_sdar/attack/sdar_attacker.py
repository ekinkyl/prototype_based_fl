"""
SDAR Attacker adapted for FedProto.

This is the core attack engine. The server (attacker) trains:
    1. Simulator (e): full model copy, trained on aux data
    2. Decoder (d): prototype → image
    3. Simulator discriminator (e_dis): real vs fake protos
    4. Decoder discriminator (d_dis): real vs fake images

There is NO server model (g) — unlike SplitNN, FedProto has no model split.
The simulator is self-contained: its own classifier head provides the
classification loss signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
from matplotlib import pyplot as plt

import sys
from pathlib import Path
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from models.client_models import get_model
from models.attacker_models import Decoder, SimulatorDiscriminator, DecoderDiscriminator
from data.data_loader import DatasetSplit


class PrototypeBank:
    """
    Stores client prototypes received across rounds.
    Provides sampling for discriminator training.
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.protos = []   # list of (proto_tensor, label_int)

    def add(self, proto, label):
        """Add a single (proto, label) pair."""
        self.protos.append((proto.detach().cpu(), label))
        # Evict oldest if exceeding max size
        if len(self.protos) > self.max_size:
            self.protos = self.protos[-self.max_size:]

    def add_batch(self, client_protos_dict):
        """
        Add prototypes received from clients.

        Args:
            client_protos_dict: dict {client_idx: {label: proto_or_list}}
                proto_or_list can be:
                  - a single Tensor (averaged mode)
                  - a list of Tensors (no-averaging / individual mode)
        """
        for client_idx, protos in client_protos_dict.items():
            for label, proto in protos.items():
                if isinstance(proto, torch.Tensor):
                    self.add(proto, label)
                elif isinstance(proto, list):
                    # Add ALL prototypes in the list
                    for p in proto:
                        if isinstance(p, torch.Tensor):
                            self.add(p, label)

    def sample(self, batch_size, device='cpu'):
        """
        Sample a batch of (protos, labels) from the bank.

        Returns:
            protos: (batch_size, proto_dim, ...) tensor
            labels: (batch_size,) tensor of ints
        """
        if len(self.protos) == 0:
            return None, None

        n = min(batch_size, len(self.protos))
        indices = np.random.choice(len(self.protos), n, replace=(n > len(self.protos)))

        protos = torch.stack([self.protos[i][0] for i in indices]).to(device)
        labels = torch.tensor([self.protos[i][1] for i in indices],
                              dtype=torch.long).to(device)
        return protos, labels

    def __len__(self):
        return len(self.protos)


class SDARAttackerFedProto:
    """
    SDAR attack engine adapted for FedProto.

    Architecture:
        - Simulator: full copy of client model (ResNet18), trained on server aux data
        - Decoder: maps prototype → image (conditional on label)
        - e_dis: discriminator distinguishing real client protos from simulator protos
        - d_dis: discriminator distinguishing real aux images from decoded images

    The simulator is self-contained — its own classifier head provides
    classification loss. There is NO separate server model.
    """

    # Dataset normalization constants (for denormalizing before decoder/discriminator)
    NORM_PARAMS = {
        'cifar10':  {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
        'cifar100': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
        'mnist':    {'mean': (0.1307,),                'std': (0.3081,)},
    }

    def __init__(self, args, server_aux_dataset, server_aux_idxs, device='cpu'):
        self.args = args
        self.device = device
        self.conditional = args.conditional
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

        # Store normalization params for denormalization
        norm = self.NORM_PARAMS.get(args.dataset, self.NORM_PARAMS['cifar10'])
        self.norm_mean = torch.tensor(norm['mean']).view(1, -1, 1, 1)
        self.norm_std = torch.tensor(norm['std']).view(1, -1, 1, 1)

        # ── Server auxiliary data loader ──
        self.aux_loader = DataLoader(
            DatasetSplit(server_aux_dataset, server_aux_idxs),
            batch_size=args.attack_batch_size,
            shuffle=True,
            drop_last=True
        )

        # ── Proto bank for discriminator training ──
        self.proto_bank = PrototypeBank(max_size=10000)

        # ── Initialize attack models ──
        self._init_models(args)

        # ── Logging ──
        self.log = {
            'sim_cls_loss': [],
            'sim_gen_loss': [],
            'sim_total_loss': [],
            'e_dis_loss': [],
            'dec_mse_loss': [],
            'dec_gen_loss': [],
            'dec_total_loss': [],
            'd_dis_loss': [],
            'attack_mse': [],
        }

    def _init_models(self, args):
        """Initialize all attack models and optimizers."""

        # Simulator: full copy of client model, or a surrogate if diff_simulator is true
        sim_model_name = 'cnn_surrogate' if getattr(args, 'diff_simulator', False) else args.model
        self.simulator, self.proto_dim, self.proto_spatial = get_model(
            sim_model_name, num_classes=args.num_classes, pretrained=False)
        self.simulator.to(self.device)

        # Decoder: prototype → image
        self.decoder = Decoder(
            proto_dim=self.proto_dim,
            num_classes=args.num_classes,
            img_channels=3 if args.dataset in ['cifar10', 'cifar100'] else 1,
            img_size=32,
            conditional=self.conditional
        ).to(self.device)

        # Simulator discriminator (optional)
        if self.lambda1 > 0:
            self.e_dis = SimulatorDiscriminator(
                proto_dim=self.proto_dim,
                num_classes=args.num_classes,
                conditional=self.conditional
            ).to(self.device)
        else:
            self.e_dis = None

        # Decoder discriminator (optional)
        if self.lambda2 > 0:
            self.d_dis = DecoderDiscriminator(
                img_channels=3 if args.dataset in ['cifar10', 'cifar100'] else 1,
                img_size=32,
                num_classes=args.num_classes,
                conditional=self.conditional
            ).to(self.device)
        else:
            self.d_dis = None

        # ── Optimizers ──
        self.sim_optimizer = torch.optim.Adam(
            self.simulator.parameters(), lr=args.attack_lr)
        self.dec_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=args.attack_lr * 0.5)

        if self.e_dis is not None:
            self.e_dis_optimizer = torch.optim.Adam(
                self.e_dis.parameters(), lr=args.attack_lr * self.lambda1)

        if self.d_dis is not None:
            self.d_dis_optimizer = torch.optim.Adam(
                self.d_dis.parameters(), lr=args.attack_lr * self.lambda2)

        # Loss functions
        self.criterion_cls = nn.NLLLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_mse = nn.MSELoss()

    def train_attack_round(self, local_protos, round_num):
        """
        Run one round of SDAR attack training.

        Called by the server after receiving client prototypes.

        Args:
            local_protos: dict {client_idx: {label: proto_tensor}}
                Per-class aggregated prototypes from each client
            round_num: current FedProto round number

        Returns:
            round_log: dict with average losses for this round
        """
        # Add received prototypes to the proto bank
        self.proto_bank.add_batch(local_protos)

        if len(self.proto_bank) < self.args.attack_batch_size // 2:
            print(f"  [SDAR] Proto bank too small ({len(self.proto_bank)}), "
                  f"skipping attack training this round.")
            return {}

        self.simulator.train()
        self.decoder.train()
        if self.e_dis is not None:
            self.e_dis.train()
        if self.d_dis is not None:
            self.d_dis.train()

        round_losses = {k: [] for k in [
            'sim_cls', 'sim_gen', 'sim_total',
            'e_dis', 'dec_mse', 'dec_gen', 'dec_total', 'd_dis'
        ]}

        for epoch in range(self.args.attack_epochs):
            for batch_idx, (x_aux, y_aux) in enumerate(self.aux_loader):
                x_aux = x_aux.to(self.device)
                y_aux = y_aux.to(self.device)

                # ─── A. TRAIN SIMULATOR ───
                sim_losses = self._train_simulator_step(x_aux, y_aux)
                round_losses['sim_cls'].append(sim_losses['cls'])
                round_losses['sim_gen'].append(sim_losses['gen'])
                round_losses['sim_total'].append(sim_losses['total'])

                # ─── B. TRAIN SIMULATOR DISCRIMINATOR ───
                if self.e_dis is not None:
                    e_dis_loss = self._train_e_dis_step(x_aux, y_aux)
                    round_losses['e_dis'].append(e_dis_loss)

                # ─── C. TRAIN DECODER ───
                dec_losses = self._train_decoder_step(x_aux, y_aux)
                round_losses['dec_mse'].append(dec_losses['mse'])
                round_losses['dec_gen'].append(dec_losses['gen'])
                round_losses['dec_total'].append(dec_losses['total'])

                # ─── D. TRAIN DECODER DISCRIMINATOR ───
                if self.d_dis is not None:
                    d_dis_loss = self._train_d_dis_step(x_aux, y_aux)
                    round_losses['d_dis'].append(d_dis_loss)

        # Compute averages
        round_log = {}
        for k, v in round_losses.items():
            if len(v) > 0:
                avg = sum(v) / len(v)
                round_log[k] = avg
                self.log[f'{"sim_cls_loss" if k == "sim_cls" else k + "_loss" if not k.endswith("_loss") else k}'] = \
                    self.log.get(k, [])

        # Log summary
        sim_cls = round_log.get('sim_cls', 0)
        dec_mse = round_log.get('dec_mse', 0)
        print(f"  [SDAR Round {round_num}] sim_cls={sim_cls:.4f}, "
              f"dec_mse={dec_mse:.4f}, "
              f"proto_bank_size={len(self.proto_bank)}")

        return round_log

    def _train_simulator_step(self, x_aux, y_aux):
        """Train simulator: classification loss + GAN loss."""
        self.sim_optimizer.zero_grad()

        # Forward through simulator
        logits_sim, proto_sim = self.simulator(x_aux)

        # Classification loss (simulator's own classifier provides signal)
        loss_cls = self.criterion_cls(logits_sim, y_aux)

        # GAN loss (fool the simulator discriminator)
        loss_gen = torch.tensor(0.0, device=self.device)
        if self.e_dis is not None and self.lambda1 > 0:
            # Flatten proto for discriminator
            proto_flat = proto_sim.view(proto_sim.size(0), -1)
            if self.conditional:
                e_dis_fake = self.e_dis(proto_flat, y_aux)
            else:
                e_dis_fake = self.e_dis(proto_flat)
            # Generator wants discriminator to think these are real
            real_labels = torch.ones_like(e_dis_fake)
            loss_gen = self.criterion_bce(e_dis_fake, real_labels)

        loss_total = loss_cls + self.lambda1 * loss_gen
        loss_total.backward()
        self.sim_optimizer.step()

        return {
            'cls': loss_cls.item(),
            'gen': loss_gen.item(),
            'total': loss_total.item()
        }

    def _train_e_dis_step(self, x_aux, y_aux):
        """Train simulator discriminator: distinguish real vs fake protos."""
        self.e_dis_optimizer.zero_grad()

        # Real prototypes from proto bank
        proto_real, labels_real = self.proto_bank.sample(
            x_aux.size(0), device=self.device)
        if proto_real is None:
            return 0.0

        proto_real_flat = proto_real.view(proto_real.size(0), -1)

        # Fake prototypes from simulator
        with torch.no_grad():
            _, proto_sim = self.simulator(x_aux)
        proto_sim_flat = proto_sim.view(proto_sim.size(0), -1)

        # Discriminator predictions
        if self.conditional:
            d_real = self.e_dis(proto_real_flat, labels_real)
            d_fake = self.e_dis(proto_sim_flat, y_aux)
        else:
            d_real = self.e_dis(proto_real_flat)
            d_fake = self.e_dis(proto_sim_flat)

        # Loss: real → 1, fake → 0
        loss_real = self.criterion_bce(d_real, torch.ones_like(d_real))
        loss_fake = self.criterion_bce(d_fake, torch.zeros_like(d_fake))
        loss = loss_real + loss_fake

        loss.backward()
        self.e_dis_optimizer.step()

        return loss.item()

    def _denormalize(self, x):
        """
        Reverse the dataset normalization: x_original = x_normalized * std + mean.
        Converts normalized images (range ~[-2.4, 2.7]) back to [0, 1].
        """
        mean = self.norm_mean.to(x.device)
        std = self.norm_std.to(x.device)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    def _train_decoder_step(self, x_aux, y_aux):
        """Train decoder: reconstruct aux images from simulator protos."""
        self.dec_optimizer.zero_grad()

        # Simulator processes normalized images (correct — matches client model)
        with torch.no_grad():
            _, proto_sim = self.simulator(x_aux)
        proto_sim_flat = proto_sim.view(proto_sim.size(0), -1)

        # Decode
        if self.conditional:
            x_recon = self.decoder(proto_sim_flat, y_aux)
        else:
            x_recon = self.decoder(proto_sim_flat)

        # Denormalize x_aux so both sides are in [0, 1]
        # (decoder outputs sigmoid → [0,1], so target must also be [0,1])
        x_aux_denorm = self._denormalize(x_aux)

        # MSE reconstruction loss (both in [0, 1])
        loss_mse = self.criterion_mse(x_recon, x_aux_denorm)

        # GAN loss (fool decoder discriminator)
        loss_gen = torch.tensor(0.0, device=self.device)
        if self.d_dis is not None and self.lambda2 > 0:
            if self.conditional:
                d_dis_fake = self.d_dis(x_recon, y_aux)
            else:
                d_dis_fake = self.d_dis(x_recon)
            real_labels = torch.ones_like(d_dis_fake)
            loss_gen = self.criterion_bce(d_dis_fake, real_labels)

        loss_total = loss_mse + self.lambda2 * loss_gen
        loss_total.backward()
        self.dec_optimizer.step()

        return {
            'mse': loss_mse.item(),
            'gen': loss_gen.item(),
            'total': loss_total.item()
        }

    def _train_d_dis_step(self, x_aux, y_aux):
        """Train decoder discriminator: distinguish real vs decoded images."""
        self.d_dis_optimizer.zero_grad()

        # Denormalize real images to [0,1] (same space as decoder output)
        x_aux_denorm = self._denormalize(x_aux)

        # Fake images = decoder output (already [0,1])
        with torch.no_grad():
            _, proto_sim = self.simulator(x_aux)
            proto_sim_flat = proto_sim.view(proto_sim.size(0), -1)
            if self.conditional:
                x_recon = self.decoder(proto_sim_flat, y_aux)
            else:
                x_recon = self.decoder(proto_sim_flat)

        # Discriminator sees both real and fake in [0,1]
        if self.conditional:
            d_real = self.d_dis(x_aux_denorm, y_aux)
            d_fake = self.d_dis(x_recon, y_aux)
        else:
            d_real = self.d_dis(x_aux_denorm)
            d_fake = self.d_dis(x_recon)

        loss_real = self.criterion_bce(d_real, torch.ones_like(d_real))
        loss_fake = self.criterion_bce(d_fake, torch.zeros_like(d_fake))
        loss = loss_real + loss_fake

        loss.backward()
        self.d_dis_optimizer.step()

        return loss.item()

    def attack(self, client_protos, client_idx=None):
        """
        Reconstruct images from received client prototypes.

        Args:
            client_protos: dict {label: proto_tensor}
                Per-class prototypes from a single client.
            client_idx: optional client index (for logging)

        Returns:
            reconstructions: dict {label: reconstructed_image_tensor}
                Each image is (3, 32, 32) in [0, 1].
        """
        self.decoder.eval()
        reconstructions = {}

        with torch.no_grad():
            for label, proto in client_protos.items():
                if isinstance(proto, list):
                    proto = proto[0]
                proto = proto.to(self.device)

                # Ensure correct shape
                if proto.dim() == 3:
                    proto = proto.unsqueeze(0)  # add batch dim
                if proto.dim() == 4:
                    proto = proto.view(proto.size(0), -1)
                if proto.dim() == 1:
                    proto = proto.unsqueeze(0)

                label_tensor = torch.tensor([label], dtype=torch.long,
                                             device=self.device)
                if self.conditional:
                    x_recon = self.decoder(proto, label_tensor)
                else:
                    x_recon = self.decoder(proto)

                reconstructions[label] = x_recon.squeeze(0).cpu()

        self.decoder.train()
        return reconstructions

    def attack_batch(self, protos_batch, labels_batch):
        """
        Efficiently reconstruct images from a batch of prototypes.

        Args:
            protos_batch: (N, 512, 1, 1) or (N, 512) tensor of prototypes
            labels_batch: (N,) tensor of integer class labels

        Returns:
            (N, 3, 32, 32) tensor of reconstructed images in [0, 1]
        """
        self.decoder.eval()
        with torch.no_grad():
            protos_batch = protos_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            protos_flat = protos_batch.view(protos_batch.size(0), -1)
            if self.conditional:
                x_recon = self.decoder(protos_flat, labels_batch)
            else:
                x_recon = self.decoder(protos_flat)
        self.decoder.train()
        return x_recon.cpu()

    def save_reconstructions(self, reconstructions, save_path, round_num,
                              client_idx=None):
        """
        Save reconstructed images as:
        1. A composite matplotlib figure (for visual inspection)
        2. Individual per-class PNG images (for metric evaluation)
        3. A JSON metadata file mapping class labels
        """
        import json

        n = len(reconstructions)
        if n == 0:
            return

        suffix = f'_client{client_idx}' if client_idx is not None else ''
        sorted_items = sorted(reconstructions.items())

        # ── 1. Composite figure ──
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
        if n == 1:
            axes = [axes]

        class_labels = []
        for i, (label, img) in enumerate(sorted_items):
            class_labels.append(int(label))
            # img is (3, 32, 32) or (1, 28, 28)
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)
                axes[i].imshow(img_np, cmap='gray')
            else:
                axes[i].imshow(img_np)
            axes[i].set_title(f'Class {label}')
            axes[i].set(xticks=[], yticks=[])

        fig_path = os.path.join(save_path, f'recon_round{round_num}{suffix}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 2. Individual per-class images ──
        for label, img in sorted_items:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img_np)
            ind_path = os.path.join(
                save_path,
                f'recon_round{round_num}{suffix}_class{label}.png')
            pil_img.save(ind_path)

        # ── 3. Metadata JSON ──
        meta = {
            'round': round_num,
            'client_idx': client_idx,
            'class_labels': class_labels,
            'num_classes': n,
        }
        meta_path = os.path.join(
            save_path, f'recon_round{round_num}{suffix}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  [SDAR] Saved reconstructions to {fig_path}")

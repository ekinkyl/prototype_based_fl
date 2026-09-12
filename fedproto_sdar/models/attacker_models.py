"""
Attacker models for SDAR attack on FedProto.
Ported from SDAR's TensorFlow attacker_models.py → PyTorch.

Contains:
    - Decoder: prototype → reconstructed image (conditional on label)
    - SimulatorDiscriminator: distinguishes real client protos from simulator protos
    - DecoderDiscriminator: distinguishes real images from decoded images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Generative decoder: maps a prototype embedding back to an image.

    For ResNet18 on CIFAR-10:
        Input:  prototype (batch, 512) + label (batch,)
        Output: reconstructed image (batch, 3, 32, 32)

    Architecture (conditional):
        1. Flatten prototype (512,1,1) → (512,)
        2. Embed label → (embed_dim,)
        3. Concatenate → (512 + embed_dim,)
        4. Linear → (256*4*4,) → reshape to (256, 4, 4)
        5. ConvTranspose2d stack: (256,4,4) → (128,8,8) → (64,16,16) → (3,32,32)
    """

    def __init__(self, proto_dim=512, num_classes=10, img_channels=3,
                 img_size=32, embed_dim=50, conditional=True):
        super(Decoder, self).__init__()
        self.proto_dim = proto_dim
        self.conditional = conditional
        self.img_size = img_size
        self.img_channels = img_channels

        if conditional:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
            input_dim = proto_dim + embed_dim
        else:
            input_dim = proto_dim

        # Project to spatial representation
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)

        # Upsample: (256,4,4) → (128,8,8) → (64,16,16)
        # Convolutional layers (using Upsample + Conv2d to prevent checkerboard artifacts)
        self.deconv1_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1_conv = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2_conv = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3_conv = nn.Conv2d(64, img_channels, 3, stride=1, padding=1)
        # No BN on final layer, use Sigmoid to output [0, 1]

    def forward(self, proto, labels=None):
        """
        Args:
            proto: (batch, 512, 1, 1) or (batch, 512) — prototype
            labels: (batch,) — integer class labels (required if conditional)

        Returns:
            (batch, 3, 32, 32) — reconstructed image in [0, 1]
        """
        # Flatten prototype if it has spatial dims
        if proto.dim() == 4:
            proto = proto.view(proto.size(0), -1)  # (batch, 512)

        if self.conditional:
            assert labels is not None, "Labels required for conditional decoder"
            label_emb = self.label_embedding(labels)  # (batch, embed_dim)
            x = torch.cat([proto, label_emb], dim=1)  # (batch, 512+embed_dim)
        else:
            x = proto

        x = F.relu(self.fc(x))
        x = x.view(-1, 256, 4, 4)  # (batch, 256, 4, 4)

        x = F.relu(self.bn1(self.deconv1_conv(self.deconv1_up(x))))  # (batch, 128, 8, 8)
        x = F.relu(self.bn2(self.deconv2_conv(self.deconv2_up(x))))  # (batch, 64, 16, 16)
        x = torch.sigmoid(self.deconv3_conv(self.deconv3_up(x)))     # (batch, 3, 32, 32)

        return x


class SimulatorDiscriminator(nn.Module):
    """
    MLP-based discriminator for prototype embeddings.
    Distinguishes real client prototypes from simulator-generated prototypes.

    Since prototypes from ResNet18 are effectively 512-dim vectors (spatial 1x1),
    we use a fully-connected architecture rather than convolutional.

    Input:  prototype (batch, 512) + optional label embedding
    Output: real/fake logit (batch, 1)
    """

    def __init__(self, proto_dim=512, num_classes=10, embed_dim=50,
                 conditional=True):
        super(SimulatorDiscriminator, self).__init__()
        self.conditional = conditional

        if conditional:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
            input_dim = proto_dim + embed_dim
        else:
            input_dim = proto_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1)  # output logit (no sigmoid — use BCEWithLogitsLoss)
        )

    def forward(self, proto, labels=None):
        """
        Args:
            proto: (batch, 512, 1, 1) or (batch, 512)
            labels: (batch,) integer labels

        Returns:
            (batch, 1) — real/fake logits
        """
        if proto.dim() == 4:
            proto = proto.view(proto.size(0), -1)

        if self.conditional:
            assert labels is not None
            label_emb = self.label_embedding(labels)
            x = torch.cat([proto, label_emb], dim=1)
        else:
            x = proto

        return self.net(x)


class ConvSimulatorDiscriminator(nn.Module):
    """
    Conv-based discriminator for spatial smashed data.
    Matches the original SDAR's make_simulator_discriminator architecture.

    In the original SDAR Split Learning paper, the simulator discriminator
    operates on the spatial intermediate features (e.g., 16×32×32 or 64×8×8)
    using convolutional layers. This is critical for properly aligning the
    spatial distribution of the simulator's smashed data with the client's
    real smashed data.

    Input:  smashed data (batch, in_channels, H, W) + optional label embedding
    Output: real/fake logit (batch, 1)
    """

    def __init__(self, in_channels=64, spatial_size=8, num_classes=10,
                 embed_dim=50, conditional=True):
        super(ConvSimulatorDiscriminator, self).__init__()
        self.conditional = conditional
        self.spatial_size = spatial_size

        if conditional:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
            self.label_fc = nn.Linear(embed_dim, spatial_size * spatial_size)
            conv_in_channels = in_channels + 1  # smashed + label channel
        else:
            conv_in_channels = in_channels

        # Convolutional layers (ported from SDAR's make_simulator_discriminator)
        # For 8×8 input: 8→4→2→1
        self.conv1 = nn.Conv2d(conv_in_channels, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Calculate flattened size after convolutions
        # 8→8→4→2→1 for spatial_size=8
        final_spatial = spatial_size // 8 if spatial_size >= 8 else 1
        if final_spatial < 1:
            final_spatial = 1

        self.fc = nn.Linear(256 * final_spatial * final_spatial, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, smashed, labels=None):
        """
        Args:
            smashed: (batch, C, H, W) — spatial smashed data (e.g., 64×8×8)
            labels: (batch,) integer labels

        Returns:
            (batch, 1) — real/fake logits
        """
        if self.conditional:
            assert labels is not None
            label_emb = self.label_embedding(labels)     # (batch, embed_dim)
            label_map = self.label_fc(label_emb)         # (batch, H*W)
            label_map = label_map.view(-1, 1, self.spatial_size, self.spatial_size)
            x = torch.cat([smashed, label_map], dim=1)   # (batch, C+1, H, W)
        else:
            x = smashed

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class DecoderDiscriminator(nn.Module):
    """
    Conv-based discriminator for images.
    Distinguishes real auxiliary images from decoder-reconstructed images.
    Ported from SDAR's make_decoder_discriminator (TF → PyTorch).

    Input:  image (batch, 3, 32, 32) + optional label embedding
    Output: real/fake logit (batch, 1)
    """

    def __init__(self, img_channels=3, img_size=32, num_classes=10,
                 embed_dim=50, conditional=True):
        super(DecoderDiscriminator, self).__init__()
        self.conditional = conditional
        self.img_size = img_size

        if conditional:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
            self.label_fc = nn.Linear(embed_dim, img_size * img_size)
            in_channels = img_channels + 1  # image + label channel
        else:
            in_channels = img_channels

        # Convolutional layers (ported from SDAR)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Classifier head
        # After 3 stride-2 convs on 32x32: 32→16→8→4
        self.fc = nn.Linear(256 * 4 * 4, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, img, labels=None):
        """
        Args:
            img: (batch, 3, 32, 32) — image
            labels: (batch,) integer labels

        Returns:
            (batch, 1) — real/fake logits
        """
        if self.conditional:
            assert labels is not None
            label_emb = self.label_embedding(labels)     # (batch, embed_dim)
            label_map = self.label_fc(label_emb)         # (batch, H*W)
            label_map = label_map.view(-1, 1, self.img_size, self.img_size)
            x = torch.cat([img, label_map], dim=1)       # (batch, 4, 32, 32)
        else:
            x = img

        x = F.leaky_relu(self.conv1(x), 0.2)             # (batch, 64, 32, 32)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)   # (batch, 128, 16, 16)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)   # (batch, 128, 8, 8)
        x = F.leaky_relu(self.conv4(x), 0.2)             # (batch, 256, 4, 4)

        x = x.view(x.size(0), -1)                        # (batch, 256*4*4)
        x = self.dropout(x)
        x = self.fc(x)                                   # (batch, 1)

        return x


class HybridDecoder(nn.Module):
    """
    Hybrid decoder for the smashed-data experiment.
    Takes BOTH a 1D prototype (512-dim) AND 3D smashed data (64×8×8) as inputs.

    Architecture:
        1. Prototype (512,) → Linear → reshape to (proto_spatial_ch, 8, 8)
        2. Concatenate with smashed_data (64, 8, 8) along channel dim
           → combined: (proto_spatial_ch + 64, 8, 8)
        3. Optional label embedding → expand to (1, 8, 8) and concatenate
        4. ConvTranspose2d stack: → (128, 16, 16) → (64, 32, 32) → (3, 32, 32)
    """

    def __init__(self, proto_dim=512, smashed_channels=64, smashed_spatial=8,
                 num_classes=10, img_channels=3, img_size=32,
                 embed_dim=50, conditional=True):
        super(HybridDecoder, self).__init__()
        self.proto_dim = proto_dim
        self.conditional = conditional
        self.smashed_spatial = smashed_spatial

        # Project prototype to spatial feature map
        proto_spatial_ch = 64  # channels for prototype spatial map
        self.proto_fc = nn.Linear(proto_dim, proto_spatial_ch * smashed_spatial * smashed_spatial)
        self.proto_spatial_ch = proto_spatial_ch

        # Total input channels = proto_spatial + smashed + (optional label)
        in_channels = proto_spatial_ch + smashed_channels
        if conditional:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
            self.label_fc = nn.Linear(embed_dim, smashed_spatial * smashed_spatial)
            in_channels += 1  # label channel

        # Decoder: (in_channels, 8, 8) → (128, 16, 16) → (64, 32, 32) → (3, 32, 32)
        # Using Upsample + Conv2d instead of ConvTranspose2d to prevent checkerboard artifacts
        self.deconv1_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1_conv = nn.Conv2d(in_channels, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2_conv = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # deconv3 maintains spatial dimension for HybridDecoder
        self.deconv3_conv = nn.Conv2d(64, img_channels, 3, stride=1, padding=1)
        # Sigmoid output → [0, 1]

    def forward(self, proto, smashed, labels=None):
        """
        Args:
            proto: (batch, 512) — prototype vector
            smashed: (batch, 64, 8, 8) — intermediate features from client model
            labels: (batch,) — integer class labels (required if conditional)

        Returns:
            (batch, 3, 32, 32) — reconstructed image in [0, 1]
        """
        batch_size = proto.size(0)

        # Flatten prototype if needed
        if proto.dim() == 4:
            proto = proto.view(proto.size(0), -1)

        # Project prototype to spatial map
        proto_spatial = F.relu(self.proto_fc(proto))
        proto_spatial = proto_spatial.view(
            batch_size, self.proto_spatial_ch,
            self.smashed_spatial, self.smashed_spatial
        )  # (batch, 64, 8, 8)

        # Concatenate prototype spatial map with smashed data
        x = torch.cat([proto_spatial, smashed], dim=1)  # (batch, 128, 8, 8)

        # Add label channel if conditional
        if self.conditional:
            assert labels is not None, "Labels required for conditional decoder"
            label_emb = self.label_embedding(labels)  # (batch, embed_dim)
            label_map = self.label_fc(label_emb)  # (batch, 64)
            label_map = label_map.view(
                batch_size, 1, self.smashed_spatial, self.smashed_spatial
            )  # (batch, 1, 8, 8)
            x = torch.cat([x, label_map], dim=1)  # (batch, 129, 8, 8)

        # Decode
        x = F.relu(self.bn1(self.deconv1_conv(self.deconv1_up(x))))   # (batch, 128, 16, 16)
        x = F.relu(self.bn2(self.deconv2_conv(self.deconv2_up(x))))   # (batch, 64, 32, 32)
        x = torch.sigmoid(self.deconv3_conv(x))      # (batch, 3, 32, 32)

        return x

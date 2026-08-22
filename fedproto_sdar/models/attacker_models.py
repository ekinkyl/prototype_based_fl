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

        # Upsample: (256,4,4) → (128,8,8) → (64,16,16) → (3,32,32)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, img_channels, 4, stride=2,
                                           padding=1)
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

        x = F.relu(self.bn1(self.deconv1(x)))  # (batch, 128, 8, 8)
        x = F.relu(self.bn2(self.deconv2(x)))  # (batch, 64, 16, 16)
        x = torch.sigmoid(self.deconv3(x))     # (batch, 3, 32, 32)

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

from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .ResNet import ResNet10, ResNet10NoNorm


# =====================================================================
# ResNet10 wrappers for MNIST (1-channel, 28x28)
# Feature dim = 512.
# =====================================================================

class ResNet10Mod(Model):
    """ResNet10 wrapper for MNIST. Feature dim = 512."""
    def __init__(self, config):
        super().__init__(config)
        if config.get('no_norm', False):
            self.backbone = ResNet10NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet10(num_classes=config['num_classes'])
        self.prototype = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False)
        self.backbone.linear = None

    def forward(self, x):
        feature_embedding = self.backbone(x)
        logits = self.prototype(feature_embedding)
        return logits

    def get_embedding(self, x):
        feature_embedding = self.backbone(x)
        logits = self.prototype(feature_embedding)
        return feature_embedding, logits


class ResNet10ModNH(Model):
    """ResNet10 with Normalized Head for FedNH on MNIST. Feature dim = 512."""
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        if config.get('no_norm', False):
            self.backbone = ResNet10NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet10(num_classes=config['num_classes'])
        temp = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.backbone.linear = None
        self.scaling = torch.nn.Parameter(torch.tensor([20.0]))
        self.activation = None

    def forward(self, x):
        feature_embedding = self.backbone(x)
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        self.activation = self.backbone.activation
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits



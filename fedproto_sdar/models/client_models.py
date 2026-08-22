"""
Client model architectures for FedProto.
Ported from FedProto's lib/models/resnet.py and lib/models/models.py.

All models return (log_probs, prototype) from forward().
The prototype is the attack surface — it's what gets sent to the server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# ══════════════════════════════════════════════════════════════════════
# ResNet18 for CIFAR-10 (primary target)
# ══════════════════════════════════════════════════════════════════════

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet for CIFAR-10 (32x32 input).

    forward() returns:
        log_probs: (batch, num_classes) — log-softmax predictions
        prototype: (batch, 512, 1, 1) — layer4 output, sent to server in FedProto
    """

    def __init__(self, block, layers, num_classes=10,
                 stride=None, zero_init_residual=False):
        super(ResNet, self).__init__()
        if stride is None:
            stride = [2, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride[0],
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)        # prototype: (batch, 512, 1, 1)

        x = self.avgpool(x1)       # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = self.fc(x)             # (batch, num_classes)

        return F.log_softmax(x, dim=1), x1


def resnet18(num_classes=10, stride=None, pretrained=False):
    """Constructs a ResNet-18 model for CIFAR-10."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                   stride=stride)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        # Only load weights that match (skip fc, conv1, bn1 which differ)
        for key in pretrained_dict.keys():
            if key.startswith('fc.') or key.startswith('conv1') or key.startswith('bn1'):
                pretrained_dict[key] = model_dict[key]
        model.load_state_dict(pretrained_dict)
    return model


# ══════════════════════════════════════════════════════════════════════
# CNN models (for future MNIST support)
# ══════════════════════════════════════════════════════════════════════

class CNNMnist(nn.Module):
    """
    CNN for MNIST. Prototype is the fc1 output (50-dim vector).
    """

    def __init__(self, num_channels=1, out_channels=20, num_classes=10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320 / 20 * out_channels), 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x1 = F.relu(self.fc1(x))       # prototype: (batch, 50)
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class CNNCifar(nn.Module):
    """
    CNN for CIFAR. Prototype is the fc0 output (120-dim vector).
    """

    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc0(x))       # prototype: (batch, 120)
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class CNNSurrogate(nn.Module):
    """
    A surrogate CNN to act as a black-box simulator when the server 
    does not know the client is using ResNet18.
    It intentionally differs in architecture but maps to a (512, 1, 1) prototype
    so that it matches the dimensional interface of the ResNet18 prototypes.
    """

    def __init__(self, num_classes=10):
        super(CNNSurrogate, self).__init__()
        # VGG-like simple structure, very different from ResNet (no skip connections)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4) # pool down to 1x1
        )
        # Prototype is exactly 512 channels, 1x1 spatial
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x1 = self.features(x)          # prototype: (batch, 512, 1, 1)
        x = x1.view(x1.size(0), -1)    # (batch, 512)
        x = self.fc(x)                 # (batch, num_classes)
        return F.log_softmax(x, dim=1), x1


# ══════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════

def get_model(model_name, num_classes=10, pretrained=False, **kwargs):
    """
    Create a client model by name.

    Returns:
        model: nn.Module with forward() → (log_probs, prototype)
        proto_dim: int — dimensionality of the prototype vector
        proto_spatial: tuple — spatial shape of prototype (H, W) or None if 1D
    """
    if model_name == 'resnet18':
        stride = kwargs.get('stride', [2, 2])
        model = resnet18(num_classes=num_classes, stride=stride,
                         pretrained=pretrained)
        # Prototype is (512, 1, 1) — essentially 512-dim
        return model, 512, (1, 1)

    elif model_name == 'cnn_mnist':
        out_channels = kwargs.get('out_channels', 20)
        num_channels = kwargs.get('num_channels', 1)
        model = CNNMnist(num_channels=num_channels,
                         out_channels=out_channels,
                         num_classes=num_classes)
        return model, 50, None

    elif model_name == 'cnn_cifar':
        model = CNNCifar(num_classes=num_classes)
        return model, 120, None
        
    elif model_name == 'cnn_surrogate':
        model = CNNSurrogate(num_classes=num_classes)
        return model, 512, (1, 1)

    else:
        raise ValueError(f"Unknown model: {model_name}")

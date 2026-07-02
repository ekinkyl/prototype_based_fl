import torch
import random
import torchvision.transforms as transforms
import numpy as np

def add_gaussian_noise(tensor, severity=None):
    if severity is None:
        severity = random.randint(1, 3)
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(tensor) * c
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0, 1)


def random_crop(tensor, output_size=None):
    _, _, h, w = tensor.shape
    if output_size is None:
        output_size = (h, w)
    transform = transforms.Compose([
        transforms.RandomCrop(h, padding=4)
    ])
    return transform(tensor)

def random_flip(tensor):
    if random.random() > 0.5:
        tensor = torch.flip(tensor, [2])  # 水平翻转
    if random.random() <= 0.5:
        tensor = torch.flip(tensor, [3])  # 垂直翻转
    return tensor


def random_rotation(tensor, degrees=15):
    transform = transforms.RandomRotation(degrees)
    return transform(tensor)

def weak_augmentation(tensor, output_size=None):
    _, _, h, w = tensor.shape
    if output_size is None:
        output_size = (h, w)
    transform= transforms.Compose(
            [transforms.RandomResizedCrop((32,32), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # vision.RandomGrayscale(),
            # vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return transform(tensor)

def strong_augmentation(tensor, output_size=None):
    _, _, h, w = tensor.shape
    if output_size is None:
        output_size = (h, w)
    transform= transforms.Compose(
            [transforms.RandomResizedCrop((32,32), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.RandomRotation(degrees=45),
            transforms.RandomVerticalFlip()
            # vision.RandomGrayscale(),
            # vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return transform(tensor)


def color_jitter(tensor, brightness=(0.4), contrast=(0.4), saturation=(0.4), hue=(0.2)):
    transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return transform(tensor)



def gaussian_blur(tensor, kernel_size=(5, 5), sigma=(0.1, 2.0)):
    transform = transforms.GaussianBlur(kernel_size, sigma)
    return transform(tensor)

def horizontal_flip(tensor, p=0.5):
    transform = transforms.RandomHorizontalFlip(p=p)
    return transform(tensor)

def random_erasing(tensor, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)
    return transform(tensor)


def random_scaling(tensor, scale_range=(0.8, 1.2)):
    scale_factor = random.uniform(*scale_range)
    size = (int(tensor.shape[2] * scale_factor), int(tensor.shape[3] * scale_factor))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(tensor.shape[2:])  # Assuming tensor shape is (N, C, H, W)
    ])
    return transform(tensor)


def add_uniform_noise(tensor, severity=None):
    if severity is None:
        severity = random.randint(1, 3)
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.rand_like(tensor) * c
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0, 1)


def random_data_augmentation(tensor):
    augmentation_functions = [
        # add_gaussian_noise,
        random_crop,
        random_flip,
        random_rotation,
        # color_jitter,
        # gaussian_blur,
        # random_erasing,
        # random_scaling,
        # add_uniform_noise
    ]

    augmentation_function = random.choice(augmentation_functions)

    # print(f"Applying {augmentation_function.__name__}")
    return augmentation_function(tensor)


def add_gaussian_noise(tensor, severity=None):
    # import pdb;pdb.set_trace()
    if severity is None:
        severity = random.randint(1, 3)  # 在 1 到 5 之间随机选择一个 severity 值
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    # import pdb;pdb.set_trace()
    noise = torch.randn_like(tensor) * c
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0, 1)  # 确保张量的值在 [0, 1] 范围内


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x)
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets
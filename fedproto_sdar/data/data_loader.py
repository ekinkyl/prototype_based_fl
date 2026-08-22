"""
Data loader and partitioning for FedProto.
Handles CIFAR-10 loading, Non-IID client partitioning (n-way, k-shot),
and reserves remaining data for the server's auxiliary dataset (used by SDAR attack).
"""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

class DatasetSplit(Dataset):
    """Wrapper to slice a subset of a dataset by indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(args, n_list, k_list):
    """
    Load dataset and partition among clients non-IID.
    Remaining training data is reserved for the server.

    Returns:
        train_dataset: the base training dataset
        test_dataset: the base testing dataset
        user_groups: dict of train indices assigned to each client
        user_groups_lt: dict of test indices assigned to each client
        classes_list: list of classes assigned to each client
        server_aux_idxs: list of indices reserved for the server
    """
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Group train indices by label
    train_labels = np.array(train_dataset.targets)
    train_idx_by_label = {i: np.where(train_labels == i)[0] for i in range(args.num_classes)}
    
    # Group test indices by label
    test_labels = np.array(test_dataset.targets)
    test_idx_by_label = {i: np.where(test_labels == i)[0] for i in range(args.num_classes)}

    user_groups = {}
    user_groups_lt = {}
    classes_list = []

    # Assign non-IID subsets to clients
    for i in range(args.num_users):
        user_groups[i] = []
        user_groups_lt[i] = []
        
        # Choose n random classes for this client
        n = n_list[i]
        k = k_list[i]
        selected_classes = np.random.choice(args.num_classes, n, replace=False)
        classes_list.append(selected_classes.tolist())

        for c in selected_classes:
            # Assign k train samples
            available_train = train_idx_by_label[c]
            if len(available_train) < k:
                selected_train = available_train
            else:
                selected_train = np.random.choice(available_train, k, replace=False)
            user_groups[i].extend(selected_train)
            # Remove assigned samples from available pool
            train_idx_by_label[c] = np.setdiff1d(train_idx_by_label[c], selected_train)

            # Assign test samples
            available_test = test_idx_by_label[c]
            if len(available_test) < args.test_shots:
                selected_test = available_test
            else:
                selected_test = np.random.choice(available_test, args.test_shots, replace=False)
            user_groups_lt[i].extend(selected_test)

    # Whatever is left in train_idx_by_label becomes the server's auxiliary dataset
    server_aux_idxs = []
    for c in range(args.num_classes):
        server_aux_idxs.extend(train_idx_by_label[c])
        
    # Limit server auxiliary data to server_data_frac
    total_train = len(train_dataset)
    max_server_size = int(total_train * args.server_data_frac)
    if len(server_aux_idxs) > max_server_size:
        server_aux_idxs = np.random.choice(server_aux_idxs, max_server_size, replace=False).tolist()

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, server_aux_idxs

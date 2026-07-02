"""
Generate MNIST dataset for PFLlib-based frameworks (FedTGP).

This script follows the exact same approach and format as PFLlib's official
generate_MNIST.py. It creates per-client .npz files that FedTGP's native
data_utils.py can load without any modification.

Usage (from the dataset/ directory):
    python generate_mnist.py noniid - dir
    python generate_mnist.py iid - -
    python generate_mnist.py noniid - pat

Each .npz file stores: {'data': {'x': np.array, 'y': np.array}}
"""

import numpy as np
import os
import sys
import random
import json

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# ---- PFLlib defaults ----
random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "mnist/"

# Configurable parameters (same defaults as PFLlib)
batch_size = 10
train_ratio = 0.75  # merge train+test, then split manually
alpha = 0.1         # Dirichlet concentration parameter


def separate_data(data, num_clients, num_classes, niid=False, balance=False,
                  partition=None, class_per_client=2):
    """Partition data across clients following PFLlib's logic."""
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    least_samples = int(min(batch_size / (1 - train_ratio),
                            len(dataset_label) / num_clients / 2))

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[
                :int(np.ceil((num_clients / num_classes) * class_per_client))
            ]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per)
                               for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(
                    max(num_per / 10, least_samples / num_classes),
                    num_per,
                    num_selected_clients - 1
                ).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if len(X[client]) == 0:
                    X[client] = dataset_content[
                        idx_for_each_class[i][idx:idx + num_sample]
                    ]
                    y[client] = dataset_label[
                        idx_for_each_class[i][idx:idx + num_sample]
                    ]
                else:
                    X[client] = np.append(
                        X[client],
                        dataset_content[
                            idx_for_each_class[i][idx:idx + num_sample]
                        ],
                        axis=0
                    )
                    y[client] = np.append(
                        y[client],
                        dataset_label[
                            idx_for_each_class[i][idx:idx + num_sample]
                        ],
                        axis=0
                    )
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # Dirichlet distribution
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
                # Balance
                proportions = np.array([
                    p * (len(idx_j) < N / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            X[j] = dataset_content[idx_batch[j]]
            y[j] = dataset_label[idx_batch[j]]

    else:
        raise NotImplementedError(f"Unknown partition: {partition}")

    # Compute statistics
    for client in range(num_clients):
        labels, counts = np.unique(y[client], return_counts=True)
        statistic[client] = list(zip(labels.tolist(), counts.tolist()))

    # Remove empty clients
    del_idx = []
    for i in range(num_clients):
        if len(X[i]) == 0:
            del_idx.append(i)
    for i in sorted(del_idx, reverse=True):
        X.pop(i)
        y.pop(i)
        statistic.pop(i)

    return X, y, statistic


def split_data(X, y):
    """Split each client's data into train and test sets."""
    train_data = {'x': [], 'y': []}
    test_data = {'x': [], 'y': []}

    num_clients = len(X)
    for i in range(num_clients):
        unique_labels = np.unique(y[i])
        if len(unique_labels) < 2 or len(X[i]) < 4:
            # Too few samples to split; put all in train
            train_data['x'].append(X[i])
            train_data['y'].append(y[i])
            test_data['x'].append(X[i][:1])
            test_data['y'].append(y[i][:1])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_ratio, shuffle=True
            )
            train_data['x'].append(X_train)
            train_data['y'].append(y_train)
            test_data['x'].append(X_test)
            test_data['y'].append(y_test)

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, num_classes, statistic, niid, balance, partition):
    """Save per-client .npz files in PFLlib format."""
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # Save per-client train/test .npz files
    for idx in range(num_clients):
        train_sample = {
            'x': train_data['x'][idx].tolist(),
            'y': train_data['y'][idx].tolist(),
        }
        test_sample = {
            'x': test_data['x'][idx].tolist(),
            'y': test_data['y'][idx].tolist(),
        }
        # PFLlib format: np.savez_compressed(path, data=dict)
        np.savez_compressed(
            os.path.join(train_path, str(idx) + '.npz'),
            data=train_sample
        )
        np.savez_compressed(
            os.path.join(test_path, str(idx) + '.npz'),
            data=test_sample
        )

    print(f"Saved {num_clients} clients to {train_path} and {test_path}")


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    """Main entry point: download MNIST, partition, save."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    # Check if already generated
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if (config['num_clients'] == num_clients and
                    config['non_iid'] == niid and
                    config['partition'] == partition and
                    config['alpha'] == alpha and
                    config['batch_size'] == batch_size):
                print("\nDataset already generated.\n")
                return
        except Exception:
            pass

    # Download MNIST
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True,
        transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True,
        transform=transform
    )

    # Merge train and test data, then re-split per client
    train_data_np = trainset.data.numpy()
    test_data_np = testset.data.numpy()
    train_label_np = trainset.targets.numpy()
    test_label_np = testset.targets.numpy()

    dataset_image = np.concatenate([train_data_np, test_data_np], axis=0)
    dataset_label = np.concatenate([train_label_np, test_label_np], axis=0)

    # Reshape: (N, 28, 28) -> (N, 1, 28, 28) for Conv2d compatibility
    dataset_image = dataset_image.reshape(-1, 1, 28, 28).astype(np.float32)
    # Normalize to [-1, 1] (same as transforms.Normalize([0.5], [0.5]))
    dataset_image = (dataset_image / 255.0 - 0.5) / 0.5

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients, num_classes,
        niid=niid, balance=balance, partition=partition
    )

    train_data, test_data = split_data(X, y)

    save_file(config_path, train_path, test_path,
              train_data, test_data,
              num_clients, num_classes, statistic,
              niid=niid, balance=balance, partition=partition)

    print("MNIST dataset generation complete.")


if __name__ == "__main__":
    # Parse CLI args following PFLlib convention:
    #   python generate_mnist.py noniid - dir
    #   python generate_mnist.py iid - -
    niid = True
    balance = False
    partition = "dir"

    if len(sys.argv) > 1:
        niid = (sys.argv[1] == "noniid")
    if len(sys.argv) > 2:
        balance = (sys.argv[2] == "balance")
    if len(sys.argv) > 3:
        partition = sys.argv[3] if sys.argv[3] != "-" else "pat"

    print(f"Generating MNIST: niid={niid}, balance={balance}, "
          f"partition={partition}, alpha={alpha}, "
          f"num_clients={num_clients}")

    generate_dataset(dir_path, num_clients, niid, balance, partition)

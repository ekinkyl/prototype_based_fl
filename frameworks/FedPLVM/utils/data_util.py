import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

data_root = os.getenv('DATA_ROOT', './data/')

class GenericDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_data_digit(args):
    # This acts as a fallback for IID but we will use the noniid function mainly.
    return prepare_data_digits_noniid(args.num_clients, args)

def prepare_data_digits_noniid(num_users, args):
    """
    Standardized pure MNIST loader for FedPLVM using 1-channel, 28x28.
    It distributes MNIST data across clients using Dirichlet distribution (beta).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load standard PyTorch MNIST
    mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # We map FedPLVM's list-of-datasets requirement by just cloning the same dataset N times
    train_dataset_list = [mnist_train for _ in range(num_users)]
    test_dataset_list = [mnist_test for _ in range(num_users)]

    # Dirichlet Partitioning (Label Non-IID)
    idx_batch_train = [[] for _ in range(num_users)]
    idx_batch_test = [[] for _ in range(num_users)]
    
    user_groups = {}
    user_groups_test = {}
    
    K = args.num_classes
    
    # Extract labels
    y_train = np.array(mnist_train.targets)
    y_test = np.array(mnist_test.targets)
    
    for k in range(K):
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(args.beta if args.beta > 0 else 0.5, num_users))
        proportions = proportions / proportions.sum()
        
        # Get indices for class k
        idx_k_train = np.where(y_train == k)[0]
        idx_k_test = np.where(y_test == k)[0]
        
        # Shuffle indices
        np.random.shuffle(idx_k_train)
        np.random.shuffle(idx_k_test)
        
        # Split according to proportions
        proportions_train = (proportions * len(idx_k_train)).astype(int)
        proportions_test = (proportions * len(idx_k_test)).astype(int)
        
        start_train = 0
        start_test = 0
        
        for i in range(num_users):
            end_train = start_train + proportions_train[i]
            end_test = start_test + proportions_test[i]
            
            # For the last client, give the remainder
            if i == num_users - 1:
                end_train = len(idx_k_train)
                end_test = len(idx_k_test)
                
            idx_batch_train[i].extend(idx_k_train[start_train:end_train].tolist())
            idx_batch_test[i].extend(idx_k_test[start_test:end_test].tolist())
            
            start_train = end_train
            start_test = end_test

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

# Dummy implementations for other datasets just in case they are called
def prepare_data_office(args): pass
def prepare_data_office_noniid(num_users, args): pass
def prepare_data_domain(args): pass
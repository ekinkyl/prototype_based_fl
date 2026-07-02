import os
import argparse
import pickle
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms

def generate_data(num_clients, niid=False, balance=True, partition='dir', alpha=0.1):
    print(f"Generating MNIST: niid={niid}, balance={balance}, partition={partition}, alpha={alpha}, num_clients={num_clients}")
    
    # Load dataset
    dataset_path = os.environ.get('DATA_ROOT', '../../../shared_datasets')
    trainset = datasets.MNIST(dataset_path, train=True, download=True)
    testset = datasets.MNIST(dataset_path, train=False, download=True)
    
    train_images = trainset.data.numpy()
    train_labels = trainset.targets.numpy()
    test_images = testset.data.numpy()
    test_labels = testset.targets.numpy()
    
    num_train = len(train_labels)
    num_classes = 10
    
    if not niid:
        # IID: random shuffle and split evenly
        idxs = np.random.permutation(num_train)
        batch_idxs = np.array_split(idxs, num_clients)
        client_idxs = {i: batch_idxs[i] for i in range(num_clients)}
    else:
        # Dirichlet partitioning for Non-IID
        min_size = 0
        min_require_size = 10
        client_idxs = {i: [] for i in range(num_clients)}
        
        while min_size < min_require_size:
            client_idxs = {i: [] for i in range(num_clients)}
            for k in range(num_classes):
                idx_k = np.where(train_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(client_idxs[i]) < (num_train / num_clients) if balance else 1) for i, p in enumerate(proportions)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_k_split = np.split(idx_k, proportions)
                for i in range(num_clients):
                    client_idxs[i] += idx_k_split[i].tolist()
            min_size = min([len(client_idxs[i]) for i in range(num_clients)])
            
    # Save to FedPall format
    out_dir = os.path.join(os.path.dirname(__file__), 'MNIST')
    partitions_dir = os.path.join(out_dir, 'partitions')
    os.makedirs(partitions_dir, exist_ok=True)
    
    for i in range(num_clients):
        c_idxs = np.array(client_idxs[i])
        np.random.shuffle(c_idxs)
        c_images = train_images[c_idxs]
        c_labels = train_labels[c_idxs]
        
        # FedPall uses np.load(..., allow_pickle=True), so saving with np.save or pickle dump works.
        # But wait! The original fedpall data uses np.save with .pkl extension? 
        # Actually it's just a numpy array containing an object array: [images, labels]
        # Or a tuple. We will use pickle.dump to be safe.
        with open(os.path.join(partitions_dir, f'train_part{i}.pkl'), 'wb') as f:
            pickle.dump((c_images, c_labels), f)
            
    # Save test set
    with open(os.path.join(out_dir, 'test.pkl'), 'wb') as f:
        pickle.dump((test_images, test_labels), f)
        
    print(f"Saved {num_clients} clients to {partitions_dir} and test.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--niid', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--partition', type=str, default='dir')
    args = parser.parse_args()
    
    generate_data(
        num_clients=args.num_clients, 
        niid=args.niid, 
        partition=args.partition, 
        alpha=args.alpha
    )

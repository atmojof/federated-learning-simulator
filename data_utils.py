import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class FLDataset(Dataset):
    """
    Simple Wrapper Dataset for PyTorch.
    """
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_mnist_data():
    """
    Attempts to download and load MNIST dataset.
    If it fails (e.g., no internet connection), falls back to a high-quality synthetic classification dataset.
    """
    try:
        from torchvision import datasets, transforms
        # Transform: normalize to match standard image metrics
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # We download to a 'data/' folder inside the project
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Extract numpy arrays for custom partitioning
        train_data = train_dataset.data.numpy().astype(np.float32) / 255.0
        train_data = (train_data - 0.1307) / 0.3081
        train_data = train_data.reshape(-1, 1, 28, 28)
        train_targets = train_dataset.targets.numpy()
        
        test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
        test_data = (test_data - 0.1307) / 0.3081
        test_data = test_data.reshape(-1, 1, 28, 28)
        test_targets = test_dataset.targets.numpy()
        
        return train_data, train_targets, test_data, test_targets, "Real MNIST"
        
    except Exception as e:
        # Fallback to high-quality synthetic data
        print(f"Torchvision MNIST load failed: {e}. Generating high-quality synthetic dataset.")
        
        # Generate 6000 training samples and 1000 test samples, 784 features, 10 classes
        X, y = make_classification(
            n_samples=7000, n_features=784, n_informative=100, n_redundant=10,
            n_classes=10, n_clusters_per_class=1, random_state=42
        )
        
        # Scale to standard ranges
        X = (X - X.mean()) / X.std()
        
        train_data = X[:6000].reshape(-1, 1, 28, 28).astype(np.float32)
        train_targets = y[:6000]
        
        test_data = X[6000:].reshape(-1, 1, 28, 28).astype(np.float32)
        test_targets = y[6000:]
        
        return train_data, train_targets, test_data, test_targets, "Synthetic MNIST"


def partition_data(data, targets, num_clients, partition_type='iid', alpha=0.5):
    """
    Partitions the dataset among clients.
    
    Args:
        data: numpy array of features
        targets: numpy array of class labels
        num_clients: number of edge clients
        partition_type: 'iid' or 'noniid'
        alpha: Dirichlet concentration parameter (smaller alpha = more severe skew)
        
    Returns:
        client_data: List of dicts, each with 'data' and 'targets'
    """
    num_samples = len(targets)
    client_indices = [[] for _ in range(num_clients)]
    
    if partition_type == 'iid':
        # IID: Shuffle indices and split evenly
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for i in range(num_clients):
            client_indices[i] = list(splits[i])
            
    elif partition_type == 'noniid':
        # Non-IID: Partition using Dirichlet Distribution
        # Form matrix of shape (Classes x Clients) of sample allocations
        num_classes = len(np.unique(targets))
        
        # Store index lists per class
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        
        for c in range(num_classes):
            # Sample client proportions from Dirichlet distribution
            # e.g., if alpha=0.1, output vector is sparse (e.g. [0.9, 0.05, 0.05, 0.0])
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Map proportions to exact numbers of samples
            c_samples = len(class_indices[c])
            allocation = (proportions * c_samples).astype(int)
            
            # Rebalance allocation errors due to rounding
            diff = c_samples - sum(allocation)
            if diff > 0:
                # Add remainder to the client with highest proportion
                allocation[np.argmax(proportions)] += diff
                
            # Distribute indices of class c to clients
            np.random.shuffle(class_indices[c])
            offset = 0
            for client_idx in range(num_clients):
                cnt = allocation[client_idx]
                if cnt > 0:
                    client_indices[client_idx].extend(class_indices[c][offset : offset + cnt])
                offset += cnt
                
    # Format client sets
    client_data = []
    for i in range(num_clients):
        c_idxs = client_indices[i]
        np.random.shuffle(c_idxs) # Shuffle client local set
        
        client_data.append({
            'data': data[c_idxs],
            'targets': targets[c_idxs]
        })
        
    return client_data

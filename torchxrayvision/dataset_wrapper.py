"""
Custom dataset wrapper for GAN training that tracks dataset source labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MultiDatasetWrapper(Dataset):
    """
    Wrapper for multiple datasets that adds dataset labels.
    This allows the dataset discriminator to classify which dataset each sample comes from.
    """
    def __init__(self, datasets_list, dataset_names=None):
        """
        Args:
            datasets_list: List of torch datasets
            dataset_names: List of dataset names (optional, for logging)
        """
        self.datasets = datasets_list
        self.dataset_names = dataset_names or [f"Dataset_{i}" for i in range(len(datasets_list))]
        
        # Compute cumulative sizes for indexing
        self.cumulative_sizes = [0]
        for dataset in self.datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
        
        # Create dataset label mapping
        self.dataset_labels = []
        for i, dataset in enumerate(self.datasets):
            self.dataset_labels.extend([i] * len(dataset))
        self.dataset_labels = np.array(self.dataset_labels)
        
        print(f"MultiDatasetWrapper created with {len(self.datasets)} datasets:")
        for i, (name, size) in enumerate(zip(self.dataset_names, 
                                             np.diff(self.cumulative_sizes))):
            print(f"  {i}: {name} - {size} samples")
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'img': Image tensor
                - 'lab': Original labels (if available)
                - 'dataset_label': Which dataset this sample comes from
                - Other keys from original dataset
        """
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        
        # Get local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        # Get sample from dataset
        sample = self.datasets[dataset_idx][local_idx]
        
        # Add dataset label
        if isinstance(sample, dict):
            sample['dataset_label'] = dataset_idx
        else:
            # If sample is a tuple, convert to dict
            img, *rest = sample if isinstance(sample, tuple) else (sample,)
            sample = {
                'img': img,
                'dataset_label': dataset_idx
            }
            if rest:
                sample['lab'] = rest[0] if len(rest) == 1 else rest
        
        return sample
    
    def get_dataset_distribution(self):
        """Return the distribution of samples across datasets."""
        sizes = np.diff(self.cumulative_sizes)
        proportions = sizes / sizes.sum()
        return {name: (size, prop) 
                for name, size, prop in zip(self.dataset_names, sizes, proportions)}


class BalancedMultiDatasetSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures balanced sampling across datasets.
    Useful when datasets have very different sizes.
    """
    def __init__(self, dataset, samples_per_dataset=None):
        """
        Args:
            dataset: MultiDatasetWrapper instance
            samples_per_dataset: Number of samples per dataset per epoch.
                                If None, uses size of smallest dataset.
        """
        self.dataset = dataset
        self.num_datasets = len(dataset.datasets)
        
        # Determine samples per dataset
        if samples_per_dataset is None:
            sizes = np.diff(dataset.cumulative_sizes)
            self.samples_per_dataset = int(min(sizes))
        else:
            self.samples_per_dataset = samples_per_dataset
        
        # Create indices for each dataset
        self.dataset_indices = []
        for i in range(self.num_datasets):
            start = dataset.cumulative_sizes[i]
            end = dataset.cumulative_sizes[i + 1]
            self.dataset_indices.append(list(range(start, end)))
        
        print(f"BalancedMultiDatasetSampler: {self.samples_per_dataset} samples per dataset")
    
    def __iter__(self):
        # Sample from each dataset
        sampled_indices = []
        for dataset_inds in self.dataset_indices:
            # Random sampling with replacement if needed
            if len(dataset_inds) >= self.samples_per_dataset:
                sampled = np.random.choice(dataset_inds, 
                                          self.samples_per_dataset, 
                                          replace=False)
            else:
                sampled = np.random.choice(dataset_inds, 
                                          self.samples_per_dataset, 
                                          replace=True)
            sampled_indices.extend(sampled)
        
        # Shuffle all indices
        np.random.shuffle(sampled_indices)
        return iter(sampled_indices)
    
    def __len__(self):
        return self.samples_per_dataset * self.num_datasets


def create_multi_dataset_loader(datasets_list, dataset_names=None, 
                               batch_size=32, num_workers=4,
                               balanced=False, samples_per_dataset=None):
    """
    Convenience function to create a DataLoader for multiple datasets.
    
    Args:
        datasets_list: List of torch datasets
        dataset_names: List of dataset names
        batch_size: Batch size
        num_workers: Number of worker processes
        balanced: If True, use balanced sampling across datasets
        samples_per_dataset: Samples per dataset if balanced=True
    
    Returns:
        DataLoader, MultiDatasetWrapper
    """
    # Create wrapper
    multi_dataset = MultiDatasetWrapper(datasets_list, dataset_names)
    
    # Create sampler if balanced
    sampler = None
    shuffle = True
    if balanced:
        sampler = BalancedMultiDatasetSampler(multi_dataset, samples_per_dataset)
        shuffle = False  # Can't use shuffle with custom sampler
    
    # Create loader
    loader = torch.utils.data.DataLoader(
        multi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, multi_dataset


def get_dataset_labels_tensor(dataset):
    """
    Extract dataset labels as a tensor.
    Useful for passing to training functions.
    """
    if isinstance(dataset, MultiDatasetWrapper):
        return torch.tensor(dataset.dataset_labels)
    else:
        raise ValueError("Dataset must be MultiDatasetWrapper instance")


# Example usage
if __name__ == "__main__":
    # Example: Create dummy datasets
    import torchvision
    
    dataset1 = torchvision.datasets.FakeData(size=1000, 
                                            image_size=(1, 224, 224),
                                            transform=torchvision.transforms.ToTensor())
    dataset2 = torchvision.datasets.FakeData(size=500,
                                            image_size=(1, 224, 224), 
                                            transform=torchvision.transforms.ToTensor())
    dataset3 = torchvision.datasets.FakeData(size=2000,
                                            image_size=(1, 224, 224),
                                            transform=torchvision.transforms.ToTensor())
    
    # Create multi-dataset loader
    loader, multi_dataset = create_multi_dataset_loader(
        [dataset1, dataset2, dataset3],
        dataset_names=['NIH', 'PadChest', 'CheXpert'],
        batch_size=16,
        balanced=True,
        samples_per_dataset=500
    )
    
    # Test iteration
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Image shape: {batch['img'].shape}")
        print(f"  Dataset labels: {batch['dataset_label']}")
        
        if batch_idx >= 2:
            break
    
    # Get distribution
    print("\nDataset distribution:")
    for name, (size, prop) in multi_dataset.get_dataset_distribution().items():
        print(f"  {name}: {size} samples ({prop*100:.1f}%)")

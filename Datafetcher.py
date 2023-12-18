import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import numpy as np

class AugmentedDataset(Dataset):
    """
    A dataset wrapper for applying transformations on the fly.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

def get_datasets(n=2, drop_label_percent=20, augment=True, dataset_type = 'mnist',iid = True):
    """
    Loads the CIFAR10 dataset, splits it into n parts, applies data augmentation,
    and creates tuples of tensors for labeled and unlabeled data.

    Args:
    n (int): Number of parts to split the dataset into.
    drop_label_percent (float): Percentage of labels to drop from each part (0 to 100).
    augment (bool): Flag to apply data augmentation.

    Returns:
    list of tuples: Each tuple contains labeled and unlabeled data.
                    Labeled data is a list of tuples (data tensor, label tensor).
                    Unlabeled data has an empty tensor as it's label.
    """



    # Check dataset type and define appropriate transformations
    if dataset_type == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_type == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        raise ValueError("Invalid dataset type. Choose 'cifar' or 'mnist'.")

    # Augmentation
    augmentation_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]) if augment and dataset_type == 'cifar' else transforms.Compose([])

    # Load dataset
    if dataset_type == 'cifar':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_type == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    augmented_dataset = AugmentedDataset(dataset, transform=augmentation_transform)


    # Split dataset into n parts
    total_size = len(augmented_dataset)
    part_sizes = [total_size // n + (1 if i < total_size % n else 0) for i in range(n)]
    if iid:
        subsets = random_split(augmented_dataset, part_sizes)
    else:
        # Group data by labels
        label_to_data = [[] for _ in range(10)]  # Assuming 10 classes for CIFAR10 and MNIST
        for idx in range(len(augmented_dataset)):
            data, label = augmented_dataset[idx]
            label_to_data[label].append((data, label))

        # Distribute each label group across the subsets non-uniformly
        subsets = [[] for _ in range(n)]
        for label_group in label_to_data:
            # Randomly distribute each label group across the subsets
            for data, label in label_group:
                subset_idx = np.random.choice(n)
                subsets[subset_idx].append((data, label))

        # Convert list of data points to Subset objects
        subsets = [torch.utils.data.Subset(augmented_dataset, [idx for idx, _ in subset]) for subset in subsets]


    federated_data = []

    for subset in subsets:
        labeled_data = []
        unlabeled_data = []

        for idx in subset.indices:
            data, label = augmented_dataset[idx]
            if np.random.rand() > drop_label_percent / 100:
                labeled_data.append((data, label))
            else:
                unlabeled_data.append((data, torch.empty(1)))

        federated_data.append((labeled_data, unlabeled_data))

    return federated_data

# Example usage
# federated_datasets = split_dataset_for_federated_learning(n=10, drop_label_percent=20, augment=True)

# # Accessing the first federated dataset
# labeled_data, unlabeled_data = federated_datasets[0]



# labeled_data is a list of tuples (data tensor, label tensor)
# unlabeled_data is a list of data tensors


def get_test_set(dataset = 'cifar'):
        if(dataset=='cifar'):
            cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_transform)

        return testset

        

        
    
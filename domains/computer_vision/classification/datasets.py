"""
Image Classification Datasets

Supports popular datasets:
- ImageNet (ILSVRC)
- CIFAR-10/100
- MNIST/Fashion-MNIST
- Oxford Flowers
- Food101
- Stanford Cars
- Custom datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Callable, Tuple, Dict, Any, Literal
import os
from pathlib import Path
import numpy as np
from PIL import Image


class ClassificationDataset(Dataset):
    """Unified interface for classification datasets."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """
        Args:
            root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            target_transform: Target transformations
            download: Whether to download dataset
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Get item by index."""
        sample, target = self.samples[idx], self.targets[idx]

        if isinstance(sample, str):
            sample = Image.open(sample).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_default_transforms(
    img_size: int = 224,
    is_training: bool = True,
    augmentation: Literal["none", "basic", "autoaugment", "randaugment"] = "basic",
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get default image transformations.

    Args:
        img_size: Target image size
        is_training: Whether for training (includes augmentation)
        augmentation: Type of augmentation
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    if is_training:
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        # Add augmentation
        if augmentation == "autoaugment":
            transform_list.append(transforms.AutoAugment())
        elif augmentation == "randaugment":
            transform_list.append(transforms.RandAugment())
        elif augmentation == "basic":
            transform_list.extend([
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomRotation(15),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation/Test transforms
        resize_size = int(img_size / 0.875)  # Standard crop ratio
        transform_list = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


def get_classification_dataset(
    dataset_name: Literal[
        "imagenet", "cifar10", "cifar100", "mnist", "fashion_mnist",
        "flowers102", "food101", "stanford_cars", "custom"
    ] = "cifar10",
    root: str = "./data",
    split: Literal["train", "val", "test"] = "train",
    img_size: int = 224,
    augmentation: str = "basic",
    download: bool = True,
    **kwargs
) -> Dataset:
    """
    Factory function to create classification datasets.

    Args:
        dataset_name: Name of the dataset
        root: Root directory for dataset
        split: Dataset split
        img_size: Image size for transforms
        augmentation: Augmentation strategy
        download: Whether to download dataset

    Returns:
        Dataset instance

    Examples:
        >>> # CIFAR-10
        >>> train_dataset = get_classification_dataset('cifar10', split='train')
        >>>
        >>> # ImageNet
        >>> val_dataset = get_classification_dataset('imagenet', split='val',
        ...                                           root='/path/to/imagenet')
        >>>
        >>> # Custom dataset
        >>> custom_dataset = get_classification_dataset('custom',
        ...                                              root='/path/to/data',
        ...                                              split='train')
    """

    is_training = split == "train"
    transform = get_default_transforms(
        img_size=img_size,
        is_training=is_training,
        augmentation=augmentation if is_training else "none",
    )

    root = Path(root) / dataset_name
    root.mkdir(parents=True, exist_ok=True)

    # Create dataset based on name
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(root),
            train=(split == "train"),
            transform=transform,
            download=download
        )

    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(root),
            train=(split == "train"),
            transform=transform,
            download=download
        )

    elif dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=str(root),
            train=(split == "train"),
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            download=download
        )

    elif dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=str(root),
            train=(split == "train"),
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,))
            ]),
            download=download
        )

    elif dataset_name == "imagenet":
        # ImageNet requires manual download
        split_folder = "train" if split == "train" else "val"
        dataset = datasets.ImageFolder(
            root=str(root / split_folder),
            transform=transform
        )

    elif dataset_name == "flowers102":
        split_map = {"train": "train", "val": "val", "test": "test"}
        dataset = datasets.Flowers102(
            root=str(root),
            split=split_map[split],
            transform=transform,
            download=download
        )

    elif dataset_name == "food101":
        dataset = datasets.Food101(
            root=str(root),
            split=split if split != "val" else "train",  # Food101 only has train/test
            transform=transform,
            download=download
        )

    elif dataset_name == "stanford_cars":
        dataset = datasets.StanfordCars(
            root=str(root),
            split=split if split != "val" else "train",
            transform=transform,
            download=download
        )

    elif dataset_name == "custom":
        # Custom dataset expects ImageFolder structure
        # root/class1/img1.jpg, root/class2/img2.jpg, ...
        dataset = datasets.ImageFolder(
            root=str(root / split),
            transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for a dataset.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )


# Dataset statistics
DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "img_size": 32,
        "num_train": 50000,
        "num_test": 10000,
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"],
    },
    "cifar100": {
        "num_classes": 100,
        "img_size": 32,
        "num_train": 50000,
        "num_test": 10000,
    },
    "mnist": {
        "num_classes": 10,
        "img_size": 28,
        "num_train": 60000,
        "num_test": 10000,
    },
    "fashion_mnist": {
        "num_classes": 10,
        "img_size": 28,
        "num_train": 60000,
        "num_test": 10000,
    },
    "imagenet": {
        "num_classes": 1000,
        "img_size": 224,
        "num_train": 1281167,
        "num_val": 50000,
    },
    "flowers102": {
        "num_classes": 102,
        "img_size": 224,
        "num_train": 1020,
        "num_val": 1020,
        "num_test": 6149,
    },
    "food101": {
        "num_classes": 101,
        "img_size": 224,
        "num_train": 75750,
        "num_test": 25250,
    },
    "stanford_cars": {
        "num_classes": 196,
        "img_size": 224,
        "num_train": 8144,
        "num_test": 8041,
    },
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset."""
    return DATASET_INFO.get(dataset_name, {})

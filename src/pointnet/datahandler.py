import torch
from torch.utils.data import DataLoader

from src.pointnet.dataset import ShapeNetDataset, ModelNet10Dataset, ModelNet40Dataset

from typing import List


def get_shapenet_dataloader(
    root: str,
    npoints: int = 2500,
    classification: bool = False,
    class_choice: List = None,
    data_augmentation: bool = True,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
):

    accepted_classes = [
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
        'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
        'Skateboard', 'Table'
    ]

    if isinstance(class_choice, List):
        for cls in class_choice:
            if cls not in accepted_classes:
                raise ValueError(
                    f"'{cls}' is not a valid class. Please select from {accepted_classes}"
                )

    if is_training:
        splits = ["train", "test"]
    else:
        splits = ["test"]

    datasets = {
        x:
        ShapeNetDataset(
            root=root,
            split=x,
            class_choice=class_choice,
            classification=classification,
            data_augmentation=data_augmentation,
            npoints=npoints,
        )
        for x in splits
    }

    dataloaders = {
        x:
        DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        for x in splits
    }

    return dataloaders


def get_modelnet10_dataloader(
    root: str,
    npoints: int = 2500,
    data_augmentation: bool = True,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
):

    if is_training:
        splits = ["train", "test"]
    else:
        splits = ["test"]

    datasets = {
        x:
        ModelNet10Dataset(
            root=root,
            split=x,
            npoints=npoints,
            data_augmentation=data_augmentation,
        )
        for x in splits
    }

    dataloaders = {
        x:
        DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        for x in splits
    }

    return dataloaders


def get_modelnet40_dataloader(
    root: str,
    npoints: int = 2500,
    data_augmentation: bool = True,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
):

    if is_training:
        splits = ["train", "test"]
    else:
        splits = ["test"]

    datasets = {
        x:
        ModelNet40Dataset(
            root=root,
            split=x,
            npoints=npoints,
            data_augmentation=data_augmentation,
        )
        for x in splits
    }

    dataloaders = {
        x:
        DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        for x in splits
    }

    return dataloaders

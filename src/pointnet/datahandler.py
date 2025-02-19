import torch
from torch.utils.data import DataLoader

from dataset import ShapeNetCoreDataset, ModelNet10Dataset, ModelNet40Dataset

from typing import List


def get_shapenetcore_dataloader(
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
    """Arguments
    root                : str   :   path to directory containing subfolders for each class (e.g. D:/storage/shapenet/shapenetcore_partanno_segmentation_benchmark_v0)
    npoints             : int   :   number of randomly sampled points from original point cloud
    classification      : bool  :   'True' will return all classes with class labels, 
                                    'False' will return selected classes in class_choice with per part segmentation
    class_choice        : List  :   list of classes to include in per part sementation dataset
    data_augmentation   : bool  :   perform data augmentation according to https://arxiv.org/abs/1612.00593 (pg.6)
    batch_size          : int   :   number of samples per batch
    is_training         : bool  :   'True' will return dataloaders with ['train', 'test']
                                    'False' will return dataloaders with only ['test']
    """

    # choose per part segmentation classes from here
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
        ShapeNetCoreDataset(
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
    """Arguments
    root                : str   :   path to directory containing subfolders for each class (e.g. D:/storage/ModelNet10)
    npoints             : int   :   number of randomly sampled points from original point cloud
    data_augmentation   : bool  :   perform data augmentation according to https://arxiv.org/abs/1612.00593 (pg.6)
    batch_size          : int   :   number of samples per batch
    is_training         : bool  :   'True' will return dataloaders with ['train', 'test']
                                    'False' will return dataloaders with only ['test']
    """

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
    """Arguments
    root                : str   :   path to directory containing subfolders for each class (e.g. D:/storage/ModelNet40)
    npoints             : int   :   number of randomly sampled points from original point cloud
    data_augmentation   : bool  :   perform data augmentation according to https://arxiv.org/abs/1612.00593 (pg.6)
    batch_size          : int   :   number of samples per batch
    is_training         : bool  :   'True' will return dataloaders with ['train', 'test']
                                    'False' will return dataloaders with only ['test']
    """

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

import torch
from torch.utils.data import DataLoader

from src.pointnet.dataset import S3DISDataset, ShapeNetCoreDataset, ModelNetDataset

from typing import List


def get_shapenetcore_dataloader(
    root: str,
    npoints: int = 1024,
    classification: bool = False,
    class_choice: List = None,
    data_augmentation: bool = True,
    normal_channel: bool = False,
    batch_size: int = 32,
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
            normal_channel=normal_channel,
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
            prefetch_factor=32,
        )
        for x in splits
    }

    return dataloaders


def get_modelnet10_dataloader(
    root: str,
    npoints: int = 1024,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
    process_data: bool = False,
    use_normals: bool = True,
    use_uniform_sample: bool = True,
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

    print(use_normals)

    datasets = {
        x:
        ModelNetDataset(
            root=root,
            split=x,
            npoints=npoints,
            num_category=10,
            process_data=process_data,
            use_normals=use_normals,
            use_uniform_sample=use_uniform_sample,
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
            prefetch_factor=32,
        )
        for x in splits
    }

    return dataloaders


def get_modelnet40_dataloader(
    root: str,
    npoints: int = 1024,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
    process_data: bool = False,
    use_normals: bool = True,
    use_uniform_sample: bool = True,
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
        ModelNetDataset(
            root=root,
            split=x,
            npoints=npoints,
            num_category=40,
            process_data=process_data,
            use_normals=use_normals,
            use_uniform_sample=use_uniform_sample,
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
            prefetch_factor=32,
        )
        for x in splits
    }

    return dataloaders

def get_s3dis_dataloader(
    data_root: str,
    num_point: int = 4096,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    is_training: bool = True,
    test_area: int = 5,
    block_size: float = 1.0,
    sample_rate: float = 1.0,
    transform=None,
):
    """
    Arguments:
    data_root    : str   :  path to directory containing S3DIS dataset
    num_point    : int   :  number of sampled points per cloud
    batch_size   : int   :  number of samples per batch
    is_training  : bool  :  return train and test loaders if True, else only test
    test_area    : int   :  test area number (1-6, default=5)
    block_size   : float :  size of each spatial block for sampling
    sample_rate  : float :  sampling rate for dataset
    transform    : func  :  data transformation function (if any)
    """
    if is_training:
        splits = ["train", "test"]
    else:
        splits = ["test"]

    datasets = {
        x: S3DISDataset(
            split=x,
            data_root=data_root,
            num_point=num_point,
            test_area=test_area,
            block_size=block_size,
            sample_rate=sample_rate,
            transform=transform,
        )
        for x in splits
    }

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=shuffle if x == "train" else False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=None,
        )
        for x in splits
    }

    return dataloaders

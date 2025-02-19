import os
from pathlib import Path
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from datahandler import get_shapenetcore_dataloader, get_modelnet10_dataloader, get_modelnet40_dataloader
from model import *


def get_dataloaders(is_training=True):
    dataset = args.dataset
    if dataset not in ['shapenet', 'modelnet10', 'modelnet40']:
        raise ValueError(
            f"'{dataset}' is not a valid dataset choice. Please select from 'shapenet' | 'modelnet10' | 'modelnet40'"
        )

    num_classes = {
        'shapenet': 16,
        'modelnet10': 10,
        'modelnet40': 40,
    }

    if dataset == 'shapenet':
        dataloaders = get_shapenetcore_dataloader(
            root=args.dataset_path,
            npoints=args.num_points,
            classification=True,
            batch_size=args.batch_size,
            is_training=is_training,
        )

    elif dataset == 'modelnet10':
        dataloaders = get_modelnet10_dataloader(
            root=args.dataset_path,
            npoints=args.num_points,
            batch_size=args.batch_size,
            is_training=is_training,
        )

    elif dataset == 'modelnet40':
        dataloaders = get_modelnet40_dataloader(
            root=args.dataset_path,
            npoints=args.num_points,
            batch_size=args.batch_size,
            is_training=is_training,
        )

    if is_training:
        for split in ['train', 'test']:
            print(f"Length of {split} dataset: {len(dataloaders[split])}")
    else:
        print(f"Length of test dataset: {len(dataloaders['test'])}")

    return dataloaders, num_classes[dataset]


def main(is_training=True):
    dataloaders, num_classes = get_dataloaders()

    model = PointNet(
        num_points=args.num_points,
        num_labels=num_classes,
    )

    new_param = model.state_dict()
    new_param['main.0.main.6.bias'] = torch.eye(3, 3).view(-1)
    new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    model.load_state_dict(new_param)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_list = []
    train_accuracy_list = []

    test_loss_list = []
    test_accuracy_list = []

    model.to(DEVICE)

    criterion = nn.BCELoss()

    phases = ['train', 'test'] if is_training else ['test']

    for epoch in range(args.epochs):
        for phase in phases:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            loss_list = []
            accuracy_list = []

            for sample in tqdm(iter(dataloaders[phase])):
                input_data, labels = sample
                batch_size = labels.numpy().shape[0]
                input_data = input_data.to(DEVICE)
                labels = labels.to(DEVICE)

                model.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_data)
                    outputs = nn.Sigmoid(outputs)

                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()

                with torch.no_grad():
                    outputs[outputs > 0.5] = 1
                    outputs[outputs < 0.5] = 0
                    accuracy = (outputs == labels).sum().item() / batch_size

                loss_list.append(loss.item())
                accuracy_list.append(accuracy)

            if (phase == 'train'):
                train_loss_list.append(np.mean(loss_list))
                train_accuracy_list.append(np.mean(train_accuracy_list))
            elif (phase == 'test'):
                test_loss_list.append(np.mean(loss_list))
                test_accuracy_list.append(np.mean(accuracy_list))

# python main.py --output C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output --dataset_path D:\storage\shapenet\shapenetcore_partanno_segmentation_benchmark_v0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="number of training epochs",
    )
    parser.add_argument(
        '--num_points',
        type=int,
        default=2500,
        help="number off points selected from point cloud",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="dataset batch size",
    )
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help="output folder")
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="dataset root directory",
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='shapenet',
        help="select from shapenet | modelnet10 | modelnet40",
    )

    args = parser.parse_args()

    if not Path(args.output).exists():
        raise ValueError(
            f"{args.output} doesn't exist. Please change or create.")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", DEVICE)

    main()

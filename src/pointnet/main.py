import os
from pathlib import Path
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from src.pointnet.datahandler import get_shapenetcore_dataloader, get_modelnet10_dataloader, get_modelnet40_dataloader
from src.pointnet.model import PointNetCls, feature_transform_regularizer


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

    model = PointNetCls(k=num_classes,
                        feature_transform=args.feature_transform)

    # hyper-parameters from PointNet paper - Supplementary - C (pg.10)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loss_list = []
    train_accuracy_list = []

    test_loss_list = []
    test_accuracy_list = []

    model.to(DEVICE)

    criterion = F.cross_entropy

    phases = ['train', 'test'] if is_training else ['test']

    with open(os.path.join(args.output, 'log.csv'), 'w',
              newline='') as csvfile:
        csvfile.write(f"epoch,train_loss,train_acc,test_loss,test_acc\n")

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

                labels = labels[:, 0]
                input_data = input_data.transpose(2, 1)

                input_data = input_data.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                model.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, trans, trans_feat = model(input_data)

                    loss = criterion(outputs, labels)

                    if args.feature_transform:
                        loss += feature_transform_regularizer(
                            trans_feat) * 0.001

                    if phase == 'train':
                        loss.backward()

                        optimizer.step()

                pred_choice = outputs.data.max(1)[1]
                correct = pred_choice.eq(labels.data).cpu().sum()
                accuracy = correct.item() / float(batch_size)

                loss_list.append(loss.item())
                accuracy_list.append(accuracy)

            epoch_loss = np.mean(loss_list)
            epoch_accuracy = np.mean(accuracy_list)
            print(
                f"{epoch+1} | {phase} | loss: {epoch_loss}\taccuracy: {epoch_accuracy}\n"
            )
            if (phase == 'train'):
                train_loss_list.append(epoch_loss)
                train_accuracy_list.append(epoch_accuracy)
            elif (phase == 'test'):
                test_loss_list.append(epoch_loss)
                test_accuracy_list.append(epoch_accuracy)

        scheduler.step()

        with open(os.path.join(args.output, 'log.csv'), 'a',
                  newline='') as csvfile:
            csvfile.write(
                f"{epoch+1},{train_loss_list[epoch]},{train_accuracy_list[epoch]},{test_loss_list[epoch]},{test_accuracy_list[epoch]}\n"
            )

        torch.save(model.state_dict(),
                   '%s/pointnet_model_%d.pth' % (args.output, (epoch + 1)))


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
    parser.add_argument('--feature_transform',
                        action='store_true',
                        help="use feature transform")

    args = parser.parse_args()

    if not Path(args.output).exists():
        raise ValueError(
            f"{args.output} doesn't exist. Please change or create.")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", DEVICE)

    main()

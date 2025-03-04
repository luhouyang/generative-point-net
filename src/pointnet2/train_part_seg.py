import os
from pathlib import Path
import argparse
from typing import List

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from src.pointnet.datahandler import get_shapenetcore_dataloader
from src.pointnet2.models.part_seg import PointNet2PartSegMSG, PointNet2PartSegMSGLoss

seg_classes = {
    'Earphone': [16, 17, 18],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11],
    'Laptop': [28, 29],
    'Cap': [6, 7],
    'Skateboard': [44, 45, 46],
    'Mug': [36, 37],
    'Guitar': [19, 20, 21],
    'Bag': [4, 5],
    'Lamp': [24, 25, 26, 27],
    'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3],
    'Pistol': [38, 39, 40],
    'Chair': [12, 13, 14, 15],
    'Knife': [22, 23]
}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[
        y.cpu().data.numpy(),
    ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def get_dataloaders(is_training=True, class_choice=['Chair']):
    dataset = args.dataset
    if dataset not in ['shapenet']:
        raise ValueError(
            f"'{dataset}' is not a valid dataset choice. Please select from 'shapenet'"
        )

    num_classes = {
        'shapenet': 16,
    }

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

    if dataset == 'shapenet':
        dataloaders = get_shapenetcore_dataloader(
            root=args.dataset_path,
            npoints=args.num_points,
            classification=False,
            data_augmentation=False,
            normal_channel=args.use_normal,
            class_choice=class_choice,
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
    num_part = 50
    dataloaders, num_classes = get_dataloaders(class_choice=[
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
        'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
        'Skateboard', 'Table'
    ])

    model = PointNet2PartSegMSG(num_classes=num_part,
                                normal_channel=args.use_normal)
    model.apply(inplace_relu)

    # hyper-parameters from PointNet paper - Supplementary - C (pg.10)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loss_list = []
    train_accuracy_list = []

    test_loss_list = []
    test_accuracy_list = []

    model.to(DEVICE)

    criterion = PointNet2PartSegMSGLoss().to(DEVICE)

    phases = ['train', 'test'] if is_training else ['test']

    ### CREATE LOG FILE ###
    with open(os.path.join(args.output, 'log.csv'), 'w',
              newline='') as csvfile:
        csvfile.write(f"epoch,train_loss,train_acc,test_loss,test_acc\n")

    for epoch in range(args.epochs):

        print(f"\n--- epoch: {epoch+1} ---")

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss_list = []
            accuracy_list = []

            for sample in tqdm(iter(dataloaders[phase])):
                ### DATA MANIPULATION ###
                input_data, cls, labels = sample

                batch_size = labels.numpy().shape[0]

                labels = labels.view(-1, 1)[:, 0]
                input_data = input_data.transpose(2, 1)

                input_data = input_data.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                cls = cls.to(DEVICE, non_blocking=True)

                ### TRAIN/TEST ###

                model.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    ### CALL MODEL FORWARD ###
                    outputs, trans_feat = model(
                        input_data, to_categorical(cls, num_classes))
                    outputs = outputs.contiguous().view(-1, num_part)

                    ### LOSS FN ###
                    loss = criterion(outputs, labels, trans_feat)

                    ### BACK-PROPAGATION ###
                    if phase == 'train':
                        loss.backward()

                        optimizer.step()

                ### BATCH CALCULATE METRICS ###
                pred_choice = outputs.data.max(1)[1]
                correct = pred_choice.eq(labels.data).cpu().sum()
                accuracy = correct.item() / (float(batch_size) *
                                             args.num_points)

                loss_list.append(loss.item())
                accuracy_list.append(accuracy)

            ### EPOCH CALCULATE METRICS ###
            epoch_loss = np.mean(loss_list)
            epoch_accuracy = np.mean(accuracy_list)
            print(
                f"epoch: {epoch+1} | {phase} | loss: {epoch_loss}\taccuracy: {epoch_accuracy}\n"
            )
            if (phase == 'train'):
                train_loss_list.append(epoch_loss)
                train_accuracy_list.append(epoch_accuracy)
            elif (phase == 'test'):
                test_loss_list.append(epoch_loss)
                test_accuracy_list.append(epoch_accuracy)

        ### LR SCHEDULER UPDATES ###
        scheduler.step()

        ### PRINT EPOCH RESULTS ###

        ### SAVE RESULTS TO FILE ###
        with open(os.path.join(args.output, 'log.csv'), 'a',
                  newline='') as csvfile:
            csvfile.write(
                f"{epoch+1},{train_loss_list[epoch]},{train_accuracy_list[epoch]},{test_loss_list[epoch]},{test_accuracy_list[epoch]}\n"
            )

        ### SAVE MODEL ON CONDITION ###
        torch.save(
            model.state_dict(),
            '%s/pointnet2_part_seg_model_%d.pth' % (args.output, (epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help="number of training epochs",
    )
    parser.add_argument(
        '--num_points',
        type=int,
        default=2048,
        help="number off points selected from point cloud",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="dataset batch size",
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="output folder",
    )
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
    parser.add_argument(
        '--use_normal',
        action='store_true',
        default=False,
        help='use xyz and normals',
    )

    args = parser.parse_args()

    print(f"Running for {args.epochs} epochs")
    print(f"Sampling {args.num_points} points")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print(f"Dataset: {args.dataset}")
    print(f"Using normals: {args.use_normal}")

    if not Path(args.output).exists():
        raise ValueError(
            f"{args.output} doesn't exist. Please change or create.")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", DEVICE)

    main()

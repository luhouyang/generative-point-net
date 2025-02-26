# C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\part_seg\pointnet_part_seg_model_67.pth

import torch
import open3d as o3d
import argparse
import numpy as np
from src.pointnet.model import PointNetPartSeg
from src.pointnet.datahandler import get_shapenetcore_dataloader
import matplotlib.pyplot as plt

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

shapenet_label2id = {
    "Airplane": 0,
    "Bag": 1,
    "Cap": 2,
    "Car": 3,
    "Chair": 4,
    "Earphone": 5,
    "Guitar": 6,
    "Knife": 7,
    "Lamp": 8,
    "Laptop": 9,
    "Motorbike": 10,
    "Mug": 11,
    "Pistol": 12,
    "Rocket": 13,
    "Skateboard": 14,
    "Table": 15
}
shapenet_id2label = {v: k for k, v in shapenet_label2id.items()}


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[
        y.cpu().data.numpy(),
    ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def generate_colormap(num_classes):
    cmap = plt.get_cmap("tab10")  # Use 'tab10' for high-contrast colors
    colors = np.array([cmap(i % 10)[:3] for i in range(num_classes)])  # Get RGB colors

    # Increase color contrast by scaling values
    colors = np.clip(colors * 1.2, 0, 1)  # Brighten colors slightly
    return colors


# Create Open3D Point Cloud with Color Mapping
def create_open3d_point_cloud(points, labels, num_classes=50):
    points = points.reshape(-1, 3).to(torch.float64).cpu().numpy()
    labels.cpu().numpy().astype(np.int32)

    # Generate color mapping
    colormap = generate_colormap(num_classes)

    # Assign colors based on labels
    colors = colormap[labels].astype(np.float64)  # Shape: [2500, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign different colors

    return pcd


# Visualization Function
def visualize(point_cloud,
              labels,
              msg,
              num_classes=50,
              window_title="Point Cloud Visualization"):
    pcd = create_open3d_point_cloud(point_cloud, labels, num_classes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{window_title} | {msg}")
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


def main(args):
    device = torch.device("cpu")
    print("Using device:", device)

    num_part = 50
    num_classes = 16

    model = PointNetPartSeg(normal_channel=True, part_num=num_part)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    class_choices = [
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
        'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
        'Skateboard', 'Table'
    ]

    dataloader = get_shapenetcore_dataloader(root=args.dataset_path,
                                             npoints=args.num_points,
                                             batch_size=1,
                                             classification=False,
                                             data_augmentation=True,
                                             normal_channel=True,
                                             class_choice=class_choices,
                                             is_training=False)['test']

    with torch.no_grad():
        for i, (input_data, cls, labels) in enumerate(dataloader):
            original_data = input_data.clone()
            input_data = input_data.transpose(2, 1).to(device)
            cls = cls.to(device)

            outputs, _ = model(input_data, to_categorical(cls, num_classes))
            outputs = outputs.contiguous().view(-1, num_part)
            pred = outputs.argmax(dim=1)
            # print(outputs.shape)
            # print(pred.shape)

            actual_label = cls[0].item() if torch.is_tensor(cls) else cls[i]

            msg = f"Sample {i + 1} | Actual Label: {shapenet_id2label[actual_label]}"
            print(msg)

            # print(original_data.shape)
            visualize(original_data[0, :, 0:3], pred, msg)

            if i == args.num_samples - 1:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="Path to trained PointNet++ model")
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help="Path to ShapeNet dataset")
    parser.add_argument('--num_points',
                        type=int,
                        default=2500,
                        help="Number of points in the point cloud")
    parser.add_argument('--num_samples',
                        type=int,
                        default=5,
                        help="Number of samples to visualize")

    args = parser.parse_args()
    main(args)

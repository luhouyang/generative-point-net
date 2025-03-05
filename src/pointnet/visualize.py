import torch
import open3d as o3d
import argparse
import numpy as np
from src.pointnet2.models.classification import PointNet2ClsMSG
from src.pointnet.model import PointNetCls
from src.pointnet.datahandler import get_shapenetcore_dataloader, get_modelnet10_dataloader, get_modelnet40_dataloader

from src.pointnet.dataset import *


def create_open3d_point_cloud(points, color):
    points = points.reshape(-1, 3).to(torch.float64).cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def visualize(point_cloud, msg, window_title="Point Cloud Visualization"):
    color = [0.5, 0.5, 0.5]
    pcd = create_open3d_point_cloud(point_cloud, color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{window_title} | {msg}")
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


def main(args):
    modelnet10_id2label = {v: k for k, v in modelnet10_label2id.items()}
    modelnet40_id2label = {v: k for k, v in modelnet40_label2id.items()}
    shapenet_id2label = {v: k for k, v in shapenet_label2id.items()}

    device = torch.device("cpu")
    print("Using device:", device)

    # model = PointNetCls(k=40, normal_channel=False)
    # model = PointNetCls(k=10, normal_channel=False)
    model = PointNet2ClsMSG(num_class=40, normal_channel=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    dataloader = get_modelnet40_dataloader(process_data=True,
                                           root=args.dataset_path,
                                           npoints=args.num_points,
                                           batch_size=1,
                                           use_normals=False,
                                           is_training=False)['test']

    with torch.no_grad():
        for i, (input_data, labels) in enumerate(dataloader):
            original_data = input_data.clone()
            input_data = input_data.transpose(2, 1).to(device)

            outputs, _ = model(input_data)
            pred = outputs.argmax(dim=1).item()

            label = modelnet40_id2label[pred]
            actual_label = labels[0].item() if torch.is_tensor(
                labels) else labels[i]

            msg = f"Sample {i + 1} | Predicted Label: {label} | Actual Label: {modelnet40_id2label[actual_label]}"
            print(msg)

            visualize(original_data[0].cpu(), msg)

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
                        default=1024,
                        help="Number of points in the point cloud")
    parser.add_argument('--num_samples',
                        type=int,
                        default=5,
                        help="Number of samples to visualize")

    args = parser.parse_args()
    main(args)

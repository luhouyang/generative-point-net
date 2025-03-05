import sys

from matplotlib import pyplot as plt
import numpy as np

from src.pointnet.datahandler import *
from src.pointnet.dataset import shapenet_label2id, modelnet10_label2id, modelnet40_label2id

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


def generate_colormap(num_classes):
    cmap = plt.get_cmap("tab10")  # Use 'tab10' for high-contrast colors
    colors = np.array([cmap(i % 10)[:3]
                       for i in range(num_classes)])  # Get RGB colors

    # Increase color contrast by scaling values
    colors = np.clip(colors * 1.2, 0, 1)  # Brighten colors slightly
    return colors


# Create Open3D Point Cloud with Color Mapping
def create_open3d_point_cloud_color(points, labels, num_classes=50):
    points = points.reshape(-1, 3).to(torch.float64).cpu().numpy()
    labels = labels.cpu().numpy().astype(np.int32)

    # Generate color mapping
    colormap = generate_colormap(num_classes)

    # Assign colors based on labels
    colors = colormap[labels].astype(np.float64)  # Shape: [2500, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign different colors

    return pcd


# Visualization Function
def visualize_color(
    point_cloud,
    labels,
    msg,
    num_classes=50,
):
    pcd = create_open3d_point_cloud_color(point_cloud, labels, num_classes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=msg)
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    import open3d as o3d
    """
    @article{Zhou2018,
        author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
        title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
        journal   = {arXiv:1801.09847},
        year      = {2018},
    }
    """

    from pathlib import Path

    dataset = sys.argv[1]
    datapath = sys.argv[2]

    dataset_list = ['shapenet', 'modelnet10', 'modelnet40', 's3dis']

    def create_open3d_point_cloud(points, color):
        """
        Create an Open3D point cloud from a tensor of points.
        Args:
            points (torch.Tensor): Tensor of shape (N, 3).
            color (list): RGB color for the point cloud.
        Returns:
            o3d.geometry.PointCloud: Open3D point cloud object.
        """
        points = points.view(-1, 3)
        points = points.to(torch.float64).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd

    def visualize(point_cloud, color, window_title):
        point_cloud = create_open3d_point_cloud(point_cloud, color)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title)
        vis.add_geometry(point_cloud)

        vis.run()
        vis.destroy_window()

    if dataset not in dataset_list:
        raise ValueError(
            f"'{dataset}' is not a valid dataset choice. Please select from 'shapenet' | 'modelnet10' | 'modelnet40'"
        )

    if not Path(datapath).exists():
        raise ValueError(
            f"'{datapath}' is not a valid path. Please check the path again.")

    if dataset == 'shapenet':

        shapenet_id2label = {v: k for k, v in shapenet_label2id.items()}

        # part segmentation
        # 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop'
        # 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'

        class_choice = [
            'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
            'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
            'Skateboard', 'Table'
        ]

        dataloaders = get_shapenetcore_dataloader(
            root=datapath,
            npoints=2500,
            classification=False,
            # normal_channel=True,
            class_choice=class_choice,
            data_augmentation=False,
            batch_size=4,
            num_workers=8,
            shuffle=True,
            is_training=True,
        )

        for phase in ["train", "test"]:
            sample = iter(dataloaders[phase])

            print(f"Number of data: {dataloaders[phase].__len__()}")
            for i in range(4):
                ps, cls, seg = sample._next_data()
                ps = ps[0]
                seg = seg[0]
                print(f"Num points: {len(ps.numpy())}")
                print(ps.size(), ps.type(), seg.size(), seg.type())
                print(ps)

                # visualize(
                #     ps, [0, 0, 1], f'Part Segmentation - {phase} - ' +
                #     class_choice[int(cls.numpy()[0])])
                visualize_color(
                    ps, seg, f'Part Segmentation - {phase} - ' +
                    class_choice[int(cls.numpy()[0])])

        # # classification
        # dataloaders = get_shapenetcore_dataloader(
        #     root=datapath,
        #     npoints=2500,
        #     classification=True,
        #     # normal_channel=False,
        #     data_augmentation=True,
        #     batch_size=4,
        #     num_workers=8,
        #     shuffle=True,
        #     is_training=True,
        # )

        # for phase in ["train", "test"]:
        #     sample = iter(dataloaders[phase])

        #     print(f"Number of data: {len(dataloaders[phase])}")
        #     for i in range(2):
        #         ps, cls = sample._next_data()
        #         ps = ps[0]
        #         cls = cls[0]
        #         print(
        #             f"\nClass: {shapenet_id2label[cls.numpy()[0]]}\tNum points: {len(ps.numpy())}"
        #         )
        #         print(ps.size(), ps.type(), cls.size(), cls.type())
        #         print(ps)

        #         visualize(
        #             ps, [0, 0, 1], f'Classification - {phase} - ' +
        #             shapenet_id2label[int(cls.numpy()[0])])

    if dataset == 'modelnet10':

        modelnet10_id2label = {v: k for k, v in modelnet10_label2id.items()}

        dataloaders = get_modelnet10_dataloader(
            root=datapath,
            npoints=2048,
            batch_size=4,
            num_workers=8,
            shuffle=True,
            is_training=True,
            use_normals=True,
            use_uniform_sample=True,
            process_data=False,
        )

        for phase in ["train", "test"]:
            sample = iter(dataloaders[phase])

            print(f"Number of data: {len(dataloaders[phase])}")
            for i in range(4):
                ps, cls = sample._next_data()
                ps = ps[0]
                cls = cls[0]
                print(
                    f"\nClass: {modelnet10_id2label[int(cls.numpy())]}\tNum points: {len(ps)}"
                )
                print(ps)

                visualize(torch.Tensor(ps[:, :3]), [0, 0, 1],
                          modelnet10_id2label[int(cls.numpy())])

    if dataset == 'modelnet40':

        modelnet40_id2label = {v: k for k, v in modelnet40_label2id.items()}

        dataloaders = get_modelnet40_dataloader(
            root=datapath,
            npoints=2048,
            batch_size=4,
            num_workers=8,
            shuffle=True,
            is_training=True,
            use_normals=True,
            use_uniform_sample=True,
            process_data=False,
        )

        for phase in ["train", "test"]:
            sample = iter(dataloaders[phase])

            print(f"Number of data: {len(dataloaders[phase])}")
            for i in range(4):
                ps, cls = sample._next_data()
                ps = ps[0]
                cls = cls[0]
                print(
                    f"\nClass: {modelnet40_id2label[int(cls.numpy())]}\tNum points: {len(ps.numpy())}"
                )
                print(ps)

                visualize(torch.Tensor(ps[:, :3]), [0, 0, 1],
                          modelnet40_id2label[int(cls.numpy())])

    if dataset == 's3dis':
        dataloaders = get_s3dis_dataloader(
            data_root=datapath,
            num_point=8192,
            batch_size=4,
            num_workers=0,
            shuffle=True,
            is_training=True,
            test_area=5,
            block_size=1.0,
            sample_rate=1.0,
            transform=None,
        )

        for phase in ["train", "test"]:
            dataloader = dataloaders[phase]
            samples = iter(dataloader)

            print(f"Number of data: {len(dataloader)}")

            for i in range(8):
                batch = next(samples)

                points, labels = batch

                print(f"\nSample {i}:")
                print(f"Num points: {points[i].shape[0]}")
                print(points[i])
                print(type(points[i]))

                visualize_color(points[i][:, :3], labels[i], f"Sample {i}")

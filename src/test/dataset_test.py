import sys

from matplotlib import pyplot as plt

from src.pointnet.dataset import *

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
        pcd.colors = o3d.utility.Vector3dVector(
            colors)  # Assign different colors

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
        # get_segmentation_classes(datapath)

        shapenet_id2label = {v: k for k, v in shapenet_label2id.items()}

        # part segmentation
        # 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop'
        # 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'

        class_choice = [
            'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
            'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
            'Skateboard', 'Table'
        ]

        d = ShapeNetCoreDataset(
            root=datapath,
            npoints=2500,
            classification=False,
            normal_channel=True,
            class_choice=class_choice,
            split='train',  # train | test
            data_augmentation=False,
        )

        print(f"Number of data: {len(d)}")
        for i in range(2):
            ps, cls, seg = d[i]
            print(f"Num points: {len(ps.numpy())}")
            print(ps.size(), ps.type(), seg.size(), seg.type())
            print(ps)

            visualize(ps, [0, 0, 1],
                      'Part Segmentation - ' + class_choice[cls.numpy()[0]])

        # classification
        d = ShapeNetCoreDataset(
            root=datapath,
            npoints=2500,
            classification=True,
            normal_channel=False,
            split='train',  # train | test
            data_augmentation=True,
        )
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())

        print(f"Number of data: {len(d)}")
        for i in range(4):
            ps, cls = d[i]
            print(
                f"\nClass: {shapenet_id2label[cls.numpy()[0]]}\tNum points: {len(ps.numpy())}"
            )
            print(ps.size(), ps.type(), seg.size(), seg.type())
            print(ps)

            visualize(
                ps, [0, 0, 1],
                'Classification - ' + shapenet_id2label[d[i][1].numpy()[0]])

    if dataset == 'modelnet10':
        gen_modelnet10_id(datapath)

        modelnet10_id2label = {v: k for k, v in modelnet10_label2id.items()}

        # d = ModelNet10Dataset(
        #     root=datapath,
        #     npoints=10000,
        #     split='train',  # train | test
        #     data_augmentation=True,
        #     file_format='txt',
        # )
        d = ModelNetDataset(
            npoints=2048,
            num_category=10,
            process_data=False,
            root=datapath,
            split='train',
            use_normals=True,
            use_uniform_sample=True,
        )

        print(f"Number of data: {len(d)}")
        for i in range(8):
            sample = d.__getitem__(i)
            print(sample[1])
            print(
                f"\nClass: {modelnet10_id2label[int(sample[1])]}\tNum points: {len(sample[0])}"
            )
            print(sample[0])
            print(type(sample[0]))
            visualize(torch.Tensor(sample[0][:, :3]), [0, 0, 1],
                      modelnet10_id2label[int(sample[1])])

    if dataset == 'modelnet40':
        gen_modelnet40_id(datapath)

        modelnet40_id2label = {v: k for k, v in modelnet40_label2id.items()}

        # d = ModelNet40Dataset(
        #     root=datapath,
        #     npoints=10000,
        #     split='train',  # train | test
        #     data_augmentation=True,
        #     file_format='txt',
        # )
        d = ModelNetDataset(
            npoints=2048,
            num_category=40,
            process_data=False,
            root=datapath,
            split='train',
            use_normals=True,
            use_uniform_sample=True,
        )

        print(f"Number of data: {len(d)}")
        for i in range(8):
            sample = d.__getitem__(i)
            print(
                f"\nClass: {modelnet40_id2label[int(sample[1])]}\tNum points: {len(sample[0])}"
            )
            print(sample[0])

            visualize(torch.Tensor(sample[0][:, :3]), [0, 0, 1],
                      modelnet40_id2label[int(sample[1])])

    if dataset == 's3dis':
        d = S3DISDataset(split='train',
                         data_root=datapath,
                         num_point=4096,
                         test_area=5,
                         block_size=1.0,
                         sample_rate=1.0,
                         transform=None)

        print(f"Number of samples in dataset: {len(d)}")
        for i in range(8):
            sample = d.__getitem__(i)
            print(sample[1])  # Labels
            print(f"\nNum points: {len(sample[0])}")
            print(sample[0])  # Point cloud data
            print(type(sample[0]))
            print(sample[0].shape)

            visualize_color(torch.Tensor(sample[0][:, :3]),
                            torch.tensor(sample[1]), f"Sample {i}")

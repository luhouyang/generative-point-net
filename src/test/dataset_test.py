import sys

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

    dataset_list = ['shapenet', 'modelnet10', 'modelnet40']

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
        get_segmentation_classes(datapath)

        shapenet_id2label = {v: k for k, v in shapenet_label2id.items()}

        # part segmentation
        # 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop'
        # 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'

        class_choice = ['Chair']

        d = ShapeNetDataset(
            root=datapath,
            npoints=2500,
            classification=False,
            class_choice=class_choice,
            split='train',  # train | test
            data_augmentation=True,
        )

        print(f"Number of data: {len(d)}")
        for i in range(2):
            ps, seg = d[i]
            print(f"Num points: {len(ps.numpy())}")
            print(ps.size(), ps.type(), seg.size(), seg.type())
            print(ps)

            visualize(ps, [0, 0, 1],
                      'Part Segmentation - ' + ' '.join(class_choice))

        # classification
        d = ShapeNetDataset(
            root=datapath,
            npoints=2500,
            classification=True,
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

        d = ModelNet10Dataset(
            root=datapath,
            npoints=10000,
            split='train',  # train | test
            data_augmentation=True,
        )

        print(f"Number of data: {len(d)}")
        for i in range(8):
            sample = d[i]
            print(
                f"\nClass: {modelnet10_id2label[sample[1].numpy()[0]]}\tNum points: {len(sample[0].numpy())}"
            )
            print(sample[0])

            visualize(sample[0], [0, 0, 1],
                      modelnet10_id2label[sample[1].numpy()[0]])

    if dataset == 'modelnet40':
        gen_modelnet40_id(datapath)

        modelnet40_id2label = {v: k for k, v in modelnet40_label2id.items()}

        d = ModelNet40Dataset(
            root=datapath,
            npoints=10000,
            split='train',  # train | test
            data_augmentation=True,
        )

        print(f"Number of data: {len(d)}")
        for i in range(8):
            sample = d[i]
            print(
                f"\nClass: {modelnet40_id2label[sample[1].numpy()[0]]}\tNum points: {len(sample[0].numpy())}"
            )
            print(sample[0])

            visualize(sample[0], [0, 0, 1],
                      modelnet40_id2label[sample[1].numpy()[0]])

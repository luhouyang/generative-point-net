# dataset classes referenced from: https://github.com/fxia22/pointnet.pytorch/tree/master

from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement

modelnet10_label2id = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'raw': 7,
    'sofa': 8,
    'label': 9,
    'toilet': 10
}

modelnet40_label2id = {
    'airplane': 0,
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'bookshelf': 4,
    'bottle': 5,
    'bowl': 6,
    'car': 7,
    'chair': 8,
    'cone': 9,
    'cup': 10,
    'curtain': 11,
    'desk': 12,
    'door': 13,
    'dresser': 14,
    'flower_pot': 15,
    'glass_box': 16,
    'guitar': 17,
    'keyboard': 18,
    'lamp': 19,
    'laptop': 20,
    'mantel': 21,
    'monitor': 22,
    'night_stand': 23,
    'person': 24,
    'piano': 25,
    'plant': 26,
    'radio': 27,
    'range_hood': 28,
    'sink': 29,
    'sofa': 30,
    'stairs': 31,
    'stool': 32,
    'table': 33,
    'tent': 34,
    'toilet': 35,
    'tv_stand': 36,
    'vase': 37,
    'wardrobe': 38,
    'xbox': 39
}


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'),
                               os.path.join(dir_seg, token + '.seg')))

    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(
                    np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(
                item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet10_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'misc/modelnet10_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


def gen_modelnet40_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'misc/modelnet40_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split',
                                 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(
                    (os.path.join(self.root, category, 'points',
                                  uuid + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
                                               0)  # center
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        point_set = point_set / dist  #scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
                rotation_matrix)  # random rotation
            point_set += np.random.normal(
                0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNet10Dataset(data.Dataset):

    def __init__(
            self,
            root,
            npoints=2500,
            split='train',  # train | test
            data_augmentation=True,
            seed: int = None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.actual_split_path = split + '_ply'
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                for name in os.listdir(
                        f"{root}/{line}/{self.actual_split_path}"):
                    datafile = f"{line}/{self.actual_split_path}/{name}"
                    self.fns.append(datafile)

        self.cat = {}
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'misc/modelnet10_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        # print(self.fns)
        # print(self.cat)
        self.classes = list(self.cat.keys())

        if seed:
            np.random.seed(seed)

        indices = np.arange(len(self.fns))
        np.random.shuffle(indices)
        self.fns = np.array(self.fns)[indices]

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([
            plydata['vertex']['x'], plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
                                               0)  # center
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
                rotation_matrix)  # random rotation
            point_set += np.random.normal(
                0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)


class ModelNet40Dataset(data.Dataset):

    def __init__(
            self,
            root,
            npoints=2500,
            split='train',  # train | test
            data_augmentation=True,
            seed: int = None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.actual_split_path = split + '_ply'
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                line = line.strip()
                for name in os.listdir(
                        f"{root}/{line}/{self.actual_split_path}"):
                    datafile = f"{line}/{self.actual_split_path}/{name}"
                    self.fns.append(datafile)

        self.cat = {}
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'misc/modelnet40_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        # print(self.fns)
        # print(self.cat)
        self.classes = list(self.cat.keys())

        if seed:
            np.random.seed(seed)

        indices = np.arange(len(self.fns))
        np.random.shuffle(indices)
        self.fns = np.array(self.fns)[indices]

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([
            plydata['vertex']['x'], plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
                                               0)  # center
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
                rotation_matrix)  # random rotation
            point_set += np.random.normal(
                0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    import open3d as o3d

    dataset = sys.argv[1]
    datapath = sys.argv[2]

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

    if dataset == 'shapenet':
        get_segmentation_classes(datapath)

        d = ShapeNetDataset(root=datapath, class_choice=['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        d = ShapeNetDataset(root=datapath, classification=True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet10':
        gen_modelnet10_id(datapath)

        modelnet10_id2label = {v: k for k, v in modelnet10_label2id.items()}

        d = ModelNet10Dataset(
            root=datapath,
            npoints=2500,
            split='train',  # train | test
            data_augmentation=True,
            seed=42,
        )

        print(f"Number of data: {len(d)}")
        for i in range(16):
            sample = d[i]
            print(
                f"\nClass: {modelnet10_id2label[sample[1].numpy()[0]]}\tNum points: {len(sample[0].numpy())}"
            )
            print(sample[0])

        sample_index = 0
        point_cloud = create_open3d_point_cloud(d[sample_index][0], [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=modelnet10_id2label[d[sample_index][1].numpy()[0]])
        vis.add_geometry(point_cloud)

        vis.run()
        vis.destroy_window()

    if dataset == 'modelnet40':
        gen_modelnet40_id(datapath)

        modelnet40_id2label = {v: k for k, v in modelnet40_label2id.items()}

        d = ModelNet40Dataset(
            root=datapath,
            npoints=2500,
            split='train',  # train | test
            data_augmentation=True,
            seed=42,
        )

        print(f"Number of data: {len(d)}")
        for i in range(16):
            sample = d[i]
            print(
                f"\nClass: {modelnet40_id2label[sample[1].numpy()[0]]}\tNum points: {len(sample[0].numpy())}"
            )
            print(sample[0])

        sample_index = 0
        point_cloud = create_open3d_point_cloud(d[sample_index][0], [0, 0, 1])
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=modelnet40_id2label[d[sample_index][1].numpy()[0]])
        vis.add_geometry(point_cloud)

        vis.run()
        vis.destroy_window()

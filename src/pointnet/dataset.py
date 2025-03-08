# dataset classes referenced from: https://github.com/fxia22/pointnet.pytorch/tree/master
# editted to work with original dataset folder structure

from __future__ import print_function
import pickle
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement

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

modelnet10_label2id = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'sofa': 7,
    'table': 8,
    'toilet': 9
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


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


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
    classes = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand',
        'sofa', 'table', 'toilet'
    ]
    # with open(os.path.join(root, 'train.txt'), 'r') as f:
    #     for line in f:
    #         classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'misc/modelnet10_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


def gen_modelnet40_id(root):
    classes = [
        "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
        "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
        "mantel", "monitor", "night_stand", "person", "piano", "plant",
        "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
        "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
    ]
    # with open(os.path.join(root, 'train.txt'), 'r') as f:
    #     for line in f:
    #         classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'misc/modelnet40_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetCoreDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints=1024,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 normal_channel=False,
                 seed: int = None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.normal_channel = normal_channel
        self.classification = classification
        self.seg_classes = {}

        # with open(self.catfile, 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.cat[ls[0]] = ls[1]

        # # print(self.cat)
        # if not class_choice is None:
        #     self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # self.id2cat = {v: k for k, v in self.cat.items()}

        # self.meta = {}
        # splitfile = os.path.join(self.root, 'train_test_split',
        #                          'shuffled_{}_file_list.json'.format(split))

        # # from IPython import embed; embed()
        # filelist = json.load(open(splitfile, 'r'))
        # for item in self.cat:
        #     self.meta[item] = []

        # for file in filelist:
        #     _, category, uuid = file.split('/')
        #     if category in self.cat.values():
        #         self.meta[self.id2cat[category]].append(
        #             (os.path.join(self.root, category, 'points',
        #                           uuid + '.pts'),
        #              os.path.join(self.root, category, 'points_label',
        #                           uuid + '.seg')))

        # self.datapath = []
        # for item in self.cat:
        #     for fn in self.meta[item]:
        #         self.datapath.append((item, fn[0], fn[1]))

        # if seed:
        #     np.random.seed(seed)

        # indices = np.arange(len(self.datapath))
        # np.random.shuffle(indices)
        # self.datapath = np.array(self.datapath)[indices]

        # self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # # print(self.classes)
        # with open(
        #         os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                      'misc/num_seg_classes.txt'), 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.seg_classes[ls[0]] = int(ls[1])
        # self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # # print(self.seg_classes, self.num_seg_classes)

        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [
                    fn for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {
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

        # print(len(self.datapath))

    def __getitem__(self, index):
        # fn = self.datapath[index]
        # cls = self.classes[self.datapath[index][0]]
        # point_set = np.loadtxt(fn[1]).astype(np.float32)
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        # cls = np.array([cls]).astype(np.int32)
        data = np.loadtxt(fn[1]).astype(np.float32)
        # print(point_set.shape)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]

        # seg = np.loadtxt(fn[2]).astype(np.int64)
        seg = data[:, -1].astype(np.int64)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        # point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
        #                                        0)  # center
        # dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        # point_set[:, 0:3] = point_set[:, 0:3] / dist  # scale
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
                rotation_matrix)  # random rotation
            point_set += np.random.normal(
                0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint, ))
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints=1024,
                 use_uniform_sample=True,
                 use_normals=True,
                 num_category=40,
                 split='train',
                 process_data=False):
        self.root = root
        self.process_data = process_data
        self.npoints = npoints
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root,
                                        'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root,
                                        'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [
                line.rstrip() for line in open(
                    os.path.join(self.root, 'modelnet10_train.txt'))
            ]
            shape_ids['test'] = [
                line.rstrip() for line in open(
                    os.path.join(self.root, 'modelnet10_test.txt'))
            ]
        else:
            shape_ids['train'] = [
                line.rstrip() for line in open(
                    os.path.join(self.root, 'modelnet40_train.txt'))
            ]
            shape_ids['test'] = [
                line.rstrip() for line in open(
                    os.path.join(self.root, 'modelnet40_test.txt'))
            ]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (shape_names[i],
             os.path.join(self.root, shape_names[i], shape_ids[split][i]) +
             '.txt') for i in range(len(shape_ids[split]))
        ]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(
                root, 'modelnet%d_%s_%dpts_fps.dat' %
                (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(
                root, 'modelnet%d_%s_%dpts.dat' %
                (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(
                    'Processing data %s (only running in the first time)...' %
                    self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)),
                                  total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1],
                                           delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(
                            point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[
                index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]


# class ModelNetDataset(data.Dataset):

#     def __init__(self,
#                  root,
#                  npoints=1024,
#                  split='train',
#                  data_augmentation=True,
#                  file_format='txt',
#                  seed=None):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.file_format = file_format  # 'ply' or 'txt'
#         self.actual_split_path = f"{split}_{file_format}"
#         self.data_augmentation = data_augmentation
#         self.fns = []

#         self.classes = self.get_classes()

#         for class_name in self.classes:
#             folder_path = os.path.join(root, class_name,
#                                        self.actual_split_path)
#             if not os.path.exists(folder_path):
#                 continue
#             for name in os.listdir(folder_path):
#                 self.fns.append(
#                     f"{class_name}/{self.actual_split_path}/{name}")

#         self.cat = {}
#         with open(
#                 os.path.join(os.path.dirname(__file__),
#                              f'misc/modelnet{len(self.classes)}_id.txt'),
#                 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = int(ls[1])

#         # Shuffle dataset
#         if seed:
#             np.random.seed(seed)
#         np.random.shuffle(self.fns)

#     def get_classes(self):
#         return []  # implemented in subclass

#     def __getitem__(self, index):
#         fn = self.fns[index]
#         cls = self.cat[fn.split('/')[0]]
#         file_path = os.path.join(self.root, fn)

#         if self.file_format == 'ply':
#             with open(file_path, 'rb') as f:
#                 plydata = PlyData.read(f)
#             pts = np.vstack([
#                 plydata['vertex']['x'], plydata['vertex']['y'],
#                 plydata['vertex']['z']
#             ]).T
#         else:
#             pts = np.loadtxt(file_path)

#         point_set = self.preprocess_points(pts)
#         cls = torch.tensor(cls, dtype=torch.long)
#         return point_set, cls

#     def preprocess_points(self, pts):
#         choice = np.random.choice(len(pts), self.npoints, replace=True)
#         point_set = pts[choice, :]

#         point_set -= np.mean(point_set, axis=0)
#         dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)))
#         point_set /= dist

#         if self.data_augmentation:
#             theta = np.random.uniform(0, np.pi * 2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                         [np.sin(theta),
#                                          np.cos(theta)]])
#             point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
#             point_set += np.random.normal(0, 0.02, size=point_set.shape)

#         return torch.tensor(point_set, dtype=torch.float32)

#     def __len__(self):
#         return len(self.fns)

# class ModelNet10Dataset(ModelNetDataset):

#     def get_classes(self):
#         return [
#             'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
#             'night_stand', 'sofa', 'table', 'toilet'
#         ]

# class ModelNet40Dataset(ModelNetDataset):

#     def get_classes(self):
#         return [
#             "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle",
#             "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door",
#             "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
#             "laptop", "mantel", "monitor", "night_stand", "person", "piano",
#             "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool",
#             "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
#         ]


class S3DISDataset(data.Dataset):

    def __init__(self,
                 split='train',
                 data_root='trainval_fullarea',
                 num_point=4096,
                 test_area=5,
                 block_size=1.0,
                 sample_rate=1.0,
                 transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [
                room for room in rooms
                if not 'Area_{}'.format(test_area) in room
            ]
        else:
            rooms_split = [
                room for room in rooms if 'Area_{}'.format(test_area) in room
            ]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:,
                                       0:6], room_data[:,
                                                       6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points,
                                           axis=0)[:3], np.amax(points,
                                                                axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(
                coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(
            np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] *
                             int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs),
                                                     split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [
                self.block_size / 2.0, self.block_size / 2.0, 0
            ]
            block_max = center + [
                self.block_size / 2.0, self.block_size / 2.0, 0
            ]
            point_idxs = np.where((points[:, 0] >= block_min[0])
                                  & (points[:, 0] <= block_max[0])
                                  & (points[:, 1] >= block_min[1])
                                  & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs,
                                                   self.num_point,
                                                   replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs,
                                                   self.num_point,
                                                   replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[
            room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[
            room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[
            room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(
                current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self,
                 root,
                 block_points=4096,
                 split='test',
                 test_area=5,
                 stride=0.5,
                 block_size=1.0,
                 padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [
                d for d in os.listdir(root)
                if d.find('Area_%d' % test_area) is -1
            ]
        else:
            self.file_list = [
                d for d in os.listdir(root)
                if d.find('Area_%d' % test_area) is not -1
            ]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + '/' + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points,
                                           axis=0)[:3], np.amax(points,
                                                                axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(
                coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(
            np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points,
                                                                    axis=0)[:3]
        grid_x = int(
            np.ceil(
                float(coord_max[0] - coord_min[0] - self.block_size) /
                self.stride) + 1)
        grid_y = int(
            np.ceil(
                float(coord_max[1] - coord_min[1] - self.block_size) /
                self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array(
            []), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding)
                                      & (points[:, 0] <= e_x + self.padding)
                                      & (points[:, 1] >= s_y - self.padding)
                                      & (points[:,
                                                1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size
                                    <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs,
                                                     point_size -
                                                     point_idxs.size,
                                                     replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:,
                           0] = data_batch[:,
                                           0] - (s_x + self.block_size / 2.0)
                data_batch[:,
                           1] = data_batch[:,
                                           1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz),
                                            axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch
                                       ]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch
                                        ]) if label_room.size else label_batch
                sample_weight = np.hstack([
                    sample_weight, batch_weight
                ]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs
                                        ]) if index_room.size else point_idxs
        data_room = data_room.reshape(
            (-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

# dataset classes referenced from: https://github.com/fxia22/pointnet.pytorch/tree/master
# editted to work with original dataset folder structure

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


class ModelNetDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints=1024,
                 split='train',
                 data_augmentation=True,
                 file_format='txt',
                 seed=None):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.file_format = file_format  # 'ply' or 'txt'
        self.actual_split_path = f"{split}_{file_format}"
        self.data_augmentation = data_augmentation
        self.fns = []

        # Define class categories
        self.classes = self.get_classes()

        # Load file names
        for class_name in self.classes:
            folder_path = os.path.join(root, class_name,
                                       self.actual_split_path)
            if not os.path.exists(folder_path):
                continue
            for name in os.listdir(folder_path):
                self.fns.append(
                    f"{class_name}/{self.actual_split_path}/{name}")

        # Load class labels
        self.cat = {}
        with open(
                os.path.join(os.path.dirname(__file__),
                             f'misc/modelnet{len(self.classes)}_id.txt'),
                'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        # Shuffle dataset
        if seed:
            np.random.seed(seed)
        np.random.shuffle(self.fns)

    def get_classes(self):
        return []  # implemented in subclass

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        file_path = os.path.join(self.root, fn)

        if self.file_format == 'ply':
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
            pts = np.vstack([
                plydata['vertex']['x'], plydata['vertex']['y'],
                plydata['vertex']['z']
            ]).T
        else:
            pts = np.loadtxt(file_path)

        point_set = self.preprocess_points(pts)
        cls = torch.tensor(cls, dtype=torch.long)
        return point_set, cls

    def preprocess_points(self, pts):
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set -= np.mean(point_set, axis=0)
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)))
        point_set /= dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        return torch.tensor(point_set, dtype=torch.float32)

    def __len__(self):
        return len(self.fns)


class ModelNet10Dataset(ModelNetDataset):

    def get_classes(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]


class ModelNet40Dataset(ModelNetDataset):

    def get_classes(self):
        return [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle",
            "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door",
            "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
            "laptop", "mantel", "monitor", "night_stand", "person", "piano",
            "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool",
            "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
        ]


# class ModelNet10Dataset(data.Dataset):

#     def __init__(
#             self,
#             root,
#             npoints=1024,
#             split='train',  # train | test
#             data_augmentation=True,
#             seed: int = None):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.actual_split_path = split + '_ply'
#         self.data_augmentation = data_augmentation
#         self.fns = []

#         classes = [
#             'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
#             'night_stand', 'sofa', 'table', 'toilet'
#         ]

#         # with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
#         #     for line in f:
#         #         line = line.strip()
#         for line in classes:
#             for name in os.listdir(f"{root}/{line}/{self.actual_split_path}"):
#                 datafile = f"{line}/{self.actual_split_path}/{name}"
#                 self.fns.append(datafile)

#         self.cat = {}
#         with open(
#                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                              'misc/modelnet10_id.txt'), 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = int(ls[1])

#         # print(self.fns)
#         # print(self.cat)
#         self.classes = list(self.cat.keys())

#         if seed:
#             np.random.seed(seed)

#         indices = np.arange(len(self.fns))
#         np.random.shuffle(indices)
#         self.fns = np.array(self.fns)[indices]

#     def __getitem__(self, index):
#         fn = self.fns[index]
#         cls = self.cat[fn.split('/')[0]]
#         with open(os.path.join(self.root, fn), 'rb') as f:
#             plydata = PlyData.read(f)
#         pts = np.vstack([
#             plydata['vertex']['x'], plydata['vertex']['y'],
#             plydata['vertex']['z']
#         ]).T
#         choice = np.random.choice(len(pts), self.npoints, replace=True)
#         point_set = pts[choice, :]

#         point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
#                                                0)  # center
#         dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
#         point_set = point_set / dist  # scale

#         if self.data_augmentation:
#             theta = np.random.uniform(0, np.pi * 2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                         [np.sin(theta),
#                                          np.cos(theta)]])
#             point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
#                 rotation_matrix)  # random rotation
#             point_set += np.random.normal(
#                 0, 0.02, size=point_set.shape)  # random jitter

#         point_set = torch.from_numpy(point_set.astype(np.float32))
#         cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#         return point_set, cls

#     def __len__(self):
#         return len(self.fns)

# class ModelNet40Dataset(data.Dataset):

#     def __init__(
#             self,
#             root,
#             npoints=1024,
#             split='train',  # train | test
#             data_augmentation=True,
#             seed: int = None):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.actual_split_path = split + '_ply'
#         self.data_augmentation = data_augmentation
#         self.fns = []

#         classes = [
#             "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle",
#             "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door",
#             "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
#             "laptop", "mantel", "monitor", "night_stand", "person", "piano",
#             "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool",
#             "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
#         ]

#         # with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
#         #     for line in f:
#         for line in classes:
#             line = line.strip()
#             for name in os.listdir(f"{root}/{line}/{self.actual_split_path}"):
#                 datafile = f"{line}/{self.actual_split_path}/{name}"
#                 self.fns.append(datafile)

#         self.cat = {}
#         with open(
#                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                              'misc/modelnet40_id.txt'), 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = int(ls[1])

#         # print(self.fns)
#         # print(self.cat)
#         self.classes = list(self.cat.keys())

#         if seed:
#             np.random.seed(seed)

#         indices = np.arange(len(self.fns))
#         np.random.shuffle(indices)
#         self.fns = np.array(self.fns)[indices]

#     def __getitem__(self, index):
#         fn = self.fns[index]
#         cls = self.cat[fn.split('/')[0]]
#         with open(os.path.join(self.root, fn), 'rb') as f:
#             plydata = PlyData.read(f)
#         pts = np.vstack([
#             plydata['vertex']['x'], plydata['vertex']['y'],
#             plydata['vertex']['z']
#         ]).T
#         choice = np.random.choice(len(pts), self.npoints, replace=True)
#         point_set = pts[choice, :]

#         point_set = point_set - np.expand_dims(np.mean(point_set, axis=0),
#                                                0)  # center
#         dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
#         point_set = point_set / dist  # scale

#         if self.data_augmentation:
#             theta = np.random.uniform(0, np.pi * 2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                         [np.sin(theta),
#                                          np.cos(theta)]])
#             point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
#                 rotation_matrix)  # random rotation
#             point_set += np.random.normal(
#                 0, 0.02, size=point_set.shape)  # random jitter

#         point_set = torch.from_numpy(point_set.astype(np.float32))
#         cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#         return point_set, cls

#     def __len__(self):
#         return len(self.fns)

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from src.pointnet.references.Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

# input transform
#
# 3D version of Spatial Transformer Network (STN) [https://arxiv.org/pdf/1811.07246]
# torch.nn.Conv1d() : 1 x 1 convolution is used to implement the MLP
#
# from PointNet paper, f(.) ≈ g[h(.)]
# where h(.) is the collection of MLPs that learn to capture properties of the set
# g(.) is the symmetric function (MaxPooling) that produce new vector that is invariant to the inputs order
#
# T-net Architecture
# learns a transformation matrix, orients the input point cloud before feature extraction
class STN3d(nn.Module):

    def __init__(self, channel=6):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        # scale up, feature extraction
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # MaxPooling (symmetric function)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # scale down, to transformation matrix 9 | (3 x 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # initialize (3 x 3) identity matrix
        iden = Variable(
            torch.from_numpy(
                np.array([1, 0, 0, 0, 1, 0, 0, 0, 1
                          ]).astype(np.float32))).view(1,
                                                       9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        # output (x) added to identity matrix, ensures matrix is non-singular (has a unique solution)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


# feature transform
class STNkd(nn.Module):

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(
                np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# classification model
class PointNetCls(nn.Module):

    # def __init__(self, k=2, feature_transform=False):
    #     super(PointNetCls, self).__init__()
    #     self.feature_transform = feature_transform
    #     self.feat = PointNetfeat(global_feat=True,
    #                              feature_transform=feature_transform)
    #     self.fc1 = nn.Linear(1024, 512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc3 = nn.Linear(256, k)
    #     self.dropout = nn.Dropout(p=0.3)
    #     self.bn1 = nn.BatchNorm1d(512)
    #     self.bn2 = nn.BatchNorm1d(256)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x, trans, trans_feat = self.feat(x)
    #     x = F.relu(self.bn1(self.fc1(x)))
    #     x = F.relu(self.bn2(self.dropout(self.fc2(x))))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1), trans, trans_feat
    def __init__(self, k=40, normal_channel=True):
        super(PointNetCls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class PointNetClsLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetClsLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class PointNetPartSeg(nn.Module):
    def __init__(self, part_num=50, normal_channel=True):
        super(PointNetPartSeg, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net, trans_feat


class PointNetPartSegLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetPartSegLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

# # not used for now
# class PointNetDenseCls(nn.Module):

#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=False,
#                                  feature_transform=feature_transform)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)

#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2, 1).contiguous()
#         x = F.log_softmax(x.view(-1, self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans, trans_feat


"""
References:

@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}

Non-singular matrix reading:
3-mins reading: https://medium.com/@noshinnawarneha/singular-and-non-singular-matrix-in-machine-learning-in-3-minutes-9f9b58b51c43
making non-singular matrix: https://www.johndcook.com/blog/2012/06/13/matrix-condition-number/
"""

# if __name__ == '__main__':
#     sim_data = Variable(torch.rand(32,3,2500))
#     trans = STN3d()
#     out = trans(sim_data)
#     print('stn', out.size())
#     print('loss', feature_transform_regularizer(out))

#     sim_data_64d = Variable(torch.rand(32, 64, 2500))
#     trans = STNkd(k=64)
#     out = trans(sim_data_64d)
#     print('stn64d', out.size())
#     print('loss', feature_transform_regularizer(out))

#     pointfeat = PointNetfeat(global_feat=True)
#     out, _, _ = pointfeat(sim_data)
#     print('global feat', out.size())

#     pointfeat = PointNetfeat(global_feat=False)
#     out, _, _ = pointfeat(sim_data)
#     print('point feat', out.size())

#     cls = PointNetCls(k = 5)
#     out, _, _ = cls(sim_data)
#     print('class', out.size())

#     seg = PointNetDenseCls(k = 3)
#     out, _, _ = seg(sim_data)
#     print('seg', out.size())

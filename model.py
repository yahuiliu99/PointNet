import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_util import PointNetSetAbstractionMsg, PointNetSetAbstraction

# Origin data size: (B, N, 3)  where B=batch_size, N=num_of_points(1024)
# Model inputs size: (B, 3, N)
# input.shape : (B, N, 3) ==> (B, 3, N)

class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k
        self.mlp_layers = nn.Sequential(
            nn.Conv1d(k,64,1),  
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Conv1d(64,128,1), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1024,1), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024,512),  
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Linear(512,256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.output = nn.Linear(256,k*k)


    def forward(self, input):
        # input.shape == (B, k, N)
        batch_size = input.size(0)
        x = self.mlp_layers(input) # (B, 1024, N)

        pool = nn.MaxPool1d(x.size(-1))(x) # (B, 1024, 1)
        flat = nn.Flatten(1)(pool) # (B, 1024)

        x = self.fc_layers(flat) # (B, 256)
        x = self.output(x) # (B, k*k)

        init = torch.eye(self.k, requires_grad=True).repeat(batch_size,1,1) # (B, k, k)
        if x.is_cuda:
            init = init.cuda()
        matrix = x.view(-1, self.k, self.k) + init # (B, k, k)
        return matrix


class BasePointNet(nn.Module):
    def __init__(self, return_feature_transform=False):
        super(BasePointNet, self).__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.return_feature_transform = return_feature_transform
        self.mlp_layer1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp_layer2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, input):
        # input.shape == (B, 3, N)
        num_points = input.size(2)
        matrix_3 = self.input_transform(input) # (B, 3, 3)
        # batch matrix multiplication
        x = torch.bmm(matrix_3, input) # (B, 3, N)
        x = self.mlp_layer1(x) # (B, 64, N)

        matrix_64 = self.feature_transform(x) # (B, 64, 64)
        x = torch.bmm(matrix_64, x) # (B, 64, N)
        local_features = x
        x = self.mlp_layer2(x) # (B, 1024, N)

        x = nn.MaxPool1d(x.size(-1))(x) # (B, 1024, 1)
        global_features = nn.Flatten(1)(x) # (B, 1024)

        if self.return_feature_transform:
            output = torch.cat([local_features, global_features.repeat(1, 1, num_points)], dim=1)
            # ouput.shape == (B, 1088, N)
            return output, matrix_3, matrix_64
        
        else:
            return global_features, matrix_3, matrix_64


class ClsPointNet(nn.Module):
    def __init__(self, classes=10):
        super(ClsPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_feature_transform=False)
        self.fc_layer = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(256, classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input):
        # input.shape == (B, 3, N)
        x, matrix_3, matrix_64 = self.base_pointnet(input) 
        # x == (B, 1024), matrix_3 == (B, 3, 3), matrix_64 == (B, 64, 64)
        x = self.fc_layer(x) # (B, 256)
        output = self.pred_layer(x) # (B, classes)
        return output, matrix_3, matrix_64


class SegPointNet(nn.Module):
    def __init__(self, categories=16):
        super(SegPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_feature_transform=True)
        self.mlp_layer1 = nn.Sequential(
            nn.Conv1d(1088,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp_layer2 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,categories,1),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self, input):
        # input.shape == (B, 3, N)
        x, matrix_3, matrix_64 = self.base_pointnet(input) 
        # x == (B, 1088, N), matrix_3 == (B, 3, 3), matrix_64 == (B, 64, 64)
        x = self.mlp_layer1(x) # (B, 128, N)
        output = self.mlp_layer2(x) # (B, categories, N)
        return output, matrix_3, matrix_64


class PointNetLoss(object):
    def __call__(self, outputs, labels, m_3, m_64, alpha=0.001):
        criterion = nn.NLLLoss()
        batch_size = outputs.size(0)
        idm_3 = torch.eye(3, requires_grad=True).repeat(batch_size,1,1)
        idm_64 = torch.eye(64, requires_grad=True).repeat(batch_size,1,1)

        if outputs.is_cuda:
            idm_3 = idm_3.cuda()
            idm_64 = idm_64.cuda()
        
        diff_3 = idm_3 - torch.bmm(m_3, m_3.transpose(1,2))
        diff_64 = idm_64 - torch.bmm(m_64, m_64.transpose(1,2))
        diff = torch.mean(torch.linalg.norm(diff_3, dim=(1,2))) + torch.mean(torch.linalg.norm(diff_64, dim=(1,2)))

        return criterion(outputs, labels) + alpha * diff


class ClsPointNet2SSG(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(ClsPointNet2SSG, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class ClsPointNet2MSG(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(ClsPointNet2MSG, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
    
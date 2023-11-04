# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: models.py
@time: 2023/6/16 02:14
"""
import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = torch.nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class PreModel(torch.nn.Module):
    def __init__(self, backbone):
        super(PreModel, self).__init__()
        self.backbone = backbone
        self.fc1 = torch.nn.Linear(1000, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        features = self.backbone(x)
        x1 = torch.relu(self.fc1(features.squeeze()))
        x2 = self.fc2(x1)
        return x2, x1

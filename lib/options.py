# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: options.py
@time: 2023/6/16 01:57
"""
import torch


class args_parser():
    def __init__(self):
        # 参数设置
        self.rounds = 50  # 训练轮数
        self.num_users = 10  # 用户数量
        self.frac = 0.04  # 客户端比例
        self.train_ep = 1  # 本地训练次数
        self.local_bs = 8  # help="本地批次大小
        self.lr = 0.001  # 学习率
        self.momentum = 0.5  # SGD动量（默认：0.5）
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 模型参数
        self.model = 'cnn'  # 模型名称
        self.alg = 'fedpn'  # 算法 fedpn fedavg fedprox
        self.num_channels = 1  # 图像通道数
        self.out_channels = 20  # 图像通道数

        # 其他参数
        self.data_dir = './data/'  # 数据集目录
        self.dataset = 'mnist'  # 数据集名称 mnist femnist
        self.num_classes = 10  # 类别数量
        self.gpu = 0  # 设置特定的GPU ID以使用cuda。默认设置为使用CPU。
        self.optimizer = 'sgd'  # 优化器类型
        self.noniid = 2  # 1、2表示为l两种Non-IID。
        self.seed = 42  # 随机种子

        # 本地参数
        self.ways = 3  # 分类数
        self.shots = 100  # 数据数量
        self.train_shots_max = 110  # 数据数量上限
        self.test_shots = 15  # 测试集数据数量
        self.stdev = 2  # 分类标准差
        self.ld = 1  # fedpn loss的权重
        self.mu = 0.1 # fedprox loss的权重

        # 可视化方案
        self.vs = 'visdom' # tensorboard、visdom
        self.t_sne = 0 # 0 开启 1 关闭
        self.cam = 0 # 0 开启 1 关闭
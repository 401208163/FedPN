# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: utils.py
@time: 2023/6/17 17:32
"""
import copy
import random

import torch
import torchvision
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from lib.sampling import noniid, noniid_lt, noniid_unequal
from lib.update import DatasetSplit


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def average_weights(w):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]

        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg


def get_dataset(args):
    if args.dataset == 'mnist':
        if args.vs == 'visdom':
            # 不归一化fedavg fedprox train不动
            apply_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])
        else:
            apply_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=apply_transform)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=apply_transform)
        return train_dataset, test_dataset
    elif args.dataset == 'femnist':
        if args.vs == 'visdom':
            apply_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])
        else:
            apply_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

        train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                                          transform=apply_transform)
        test_dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                                         transform=apply_transform)
        return train_dataset, test_dataset


def get_user_groups(args, train_dataset, test_dataset):
    if args.noniid == 1:
        user_groups = noniid_unequal(train_dataset, args.num_users)
        return user_groups
    elif args.noniid == 2:
        n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1),
                                   args.num_users)
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
        user_groups, classes_list = noniid(args, train_dataset, args.num_users, n_list, k_list)
        user_groups_lt = noniid_lt(test_dataset, args.num_users, classes_list)
        classes_list_gt = classes_list
        return user_groups, user_groups_lt


def test_inference(args, model, test_dataset, vs, idxs=[]):
    criterion = torch.nn.NLLLoss().to(args.device)

    model.to(args.device)
    model.eval()

    if idxs != []:
        test_dataloader = DataLoader(DatasetSplit(test_dataset, idxs)
                                     , batch_size=args.local_bs
                                     , shuffle=True
                                     , drop_last=True
                                     )
    else:
        test_dataloader = DataLoader(test_dataset
                                     , batch_size=args.local_bs
                                     , shuffle=True
                                     , drop_last=True
                                     )

    loss, total, acc = 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        if args.vs == "visdom":
            if batch_idx % 100 == 0:
                vs.images(images, win='test_images', opts=dict(title='test_images'))
        images, labels = images.to(args.device), labels.to(args.device)

        model.zero_grad()
        outputs, protos = model(images)

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        acc += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    acc_avg = acc / total
    loss_avg = loss / total
    return acc_avg, loss_avg


def get_proto(args, model, test_dataset, idxs=[]):
    model.to(args.device)
    model.eval()

    if idxs != []:
        test_dataloader = DataLoader(DatasetSplit(test_dataset, idxs)
                                     , batch_size=args.local_bs
                                     , shuffle=True
                                     , drop_last=True
                                     )
    else:
        test_dataloader = DataLoader(test_dataset
                                     , batch_size=args.local_bs
                                     , shuffle=True
                                     , drop_last=True
                                     )

    images_array = []
    labels_array = []
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(args.device), labels.to(args.device)

        _, proto = model(images)
        images_array.append(proto.clone().detach().cpu().numpy())
        labels_array.append(labels.cpu().numpy().reshape(-1, 1))
    images_array = np.vstack(images_array)
    labels_array = np.vstack(labels_array)
    return images_array, labels_array


def vis_t_sne_lt(args, local_model_list, test_dataset, r, vs, colors, user_groups_lt=[]):
    images_array_list, labels_array_list = [], []
    for idx, model in enumerate(local_model_list):
        if args.noniid == 1:
            images_array, labels_array = get_proto(args=args, model=model, test_dataset=test_dataset)
        elif args.noniid == 2:
            images_array, labels_array = get_proto(args=args, model=model, test_dataset=test_dataset,
                                                   idxs=user_groups_lt[idx])
        images_array_list.append(images_array)
        labels_array_list.append(labels_array)
    images_array_list = np.vstack(images_array_list)
    labels_array_list = np.vstack(labels_array_list)
    tsne = TSNE()
    X = tsne.fit_transform(images_array_list)
    Y = labels_array_list.reshape((-1, 1)) + 1
    vs.scatter(
        X=X,
        Y=Y,
        opts=dict(
            markersize=5,
            markercolor=colors,
            title=f'T-SNE(Alg:{args.alg},Epoch:{r})'
        ),
    )


def vis_t_sne(args, model, test_dataset, r, vs, colors):
    images_array, labels_array = get_proto(args=args, model=model, test_dataset=test_dataset)
    images_array_list = np.vstack(images_array)
    labels_array_list = np.vstack(labels_array)
    tsne = TSNE()
    X = tsne.fit_transform(images_array_list)
    Y = labels_array_list.reshape((-1, 1)) + 1
    vs.scatter(
        X=X,
        Y=Y,
        opts=dict(
            markersize=5,
            markercolor=colors,
            title=f'T-SNE(Alg:{args.alg},Epoch:{r})'
        ),
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

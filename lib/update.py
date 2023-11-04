# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: update.py
@time: 2023/6/16 15:57
"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, train_dataset, idxs):
        self.args = args
        self.train_dataloader = DataLoader(DatasetSplit(train_dataset, idxs)
                                           , batch_size=self.args.local_bs
                                           , shuffle=True
                                           , drop_last=True
                                           )
        self.criterion = nn.NLLLoss().to(self.args.device)
        self.loss_mse = nn.MSELoss().to(self.args.device)

    def update_weights_het(self, global_protos, model, vs):
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}
        epoch_acc = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            batch_acc = []
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.train_dataloader):
                if self.args.vs == "visdom":
                    if batch_idx % 10 == 0:
                        vs.images(images, win='train_images', opts=dict(title='train_images'))
                images, labels = images.to(self.args.device), label_g.to(self.args.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss1 = self.criterion(log_probs, labels)

                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = copy.deepcopy(protos.data.clone().detach())
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = self.loss_mse(proto_new, protos)

                loss = loss1 + loss2 * self.args.ld
                loss.backward()
                optimizer.step()
                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i, :]]

                log_probs = log_probs[:, 0:self.args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                batch_acc.append(acc_val.item())
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, sum(epoch_acc) / len(epoch_acc), agg_protos_label

    def update_weights(self, model, vs):
        model.train()
        epoch_acc = []
        epoch_loss = []

        global_model = copy.deepcopy(model)

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.train_dataloader):
                if self.args.vs == "visdom":
                    if batch_idx % 100 == 0:
                        vs.images(images, win='train_images', opts=dict(title='train_images'))
                images, labels = images.to(self.args.device), labels_g.to(self.args.device)

                model.zero_grad()
                log_probs, _ = model(images)

                if self.args.alg == 'fedprox':
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss = self.criterion(log_probs, labels) + (self.args.mu / 2) * proximal_term
                else:
                    loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                batch_loss.append(loss.item())
                batch_acc.append(acc_val.item())
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
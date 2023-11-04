# -*- coding: utf-8 -*-
"""
@author: Kuang Hangdong
@software: PyCharm
@file: main.py
@time: 2023/6/16 01:54
"""
import copy
import datetime
import numpy as np
import torch
from tqdm import tqdm

from lib.models import CNN
from lib.update import LocalUpdate
from lib.options import args_parser
from lib.utils import agg_func, proto_aggregation, average_weights, test_inference, get_dataset, get_user_groups, \
    setup_seed, vis_t_sne_lt, vis_t_sne

from visdom import Visdom
from tensorboardX import SummaryWriter


def FedPN(args, train_dataset, test_dataset, local_model_list, user_groups, vs, user_groups_lt=[]):
    global_protos = []
    train_loss_list, train_loss_std_list, train_accuracy_list, train_accuracy_std_list = [], [], [], []
    test_loss_list, test_loss_std_list, test_accuracy_list, test_accuracy_std_list = [], [], [], []
    colors = np.random.randint(0, 255, (args.num_classes, 3,))
    for r in tqdm(range(args.rounds)):
        local_weights, local_train_loss, local_protos, local_train_acc = [], [], {}, []
        print(f'\n | Global Training Round : {r + 1} |\n')

        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, train_dataset=train_dataset, idxs=user_groups[idx])
            w, train_loss, train_acc, protos = local_model.update_weights_het(copy.deepcopy(global_protos),
                                                                              copy.deepcopy(local_model_list[idx]), vs)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_train_loss.append(copy.deepcopy(train_loss['total']))
            local_train_acc.append(copy.deepcopy(train_acc))
            local_protos[idx] = agg_protos

        # update global weights
        local_weights_list = local_weights

        for idx in range(args.num_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        train_loss_avg = np.mean(local_train_loss)
        train_loss_std = np.std(local_train_loss)
        train_acc_avg = np.mean(local_train_acc)
        train_acc_std = np.std(local_train_acc)
        print(
            f"train_loss_avg:{train_loss_avg:.3f}, train_loss_std:{train_loss_std:.3f};train_acc_avg:{train_acc_avg:.3f}, train_acc_std:{train_acc_std:.3f}.")
        train_loss_list.append(round(train_loss_avg, 3))
        train_loss_std_list.append(round(train_loss_std, 3))
        train_accuracy_list.append(round(train_acc_avg, 3))
        train_accuracy_std_list.append(round(train_acc_std, 3))

        for idx in range(args.num_users):
            print(f"client{idx}:train_loss:{local_train_loss[idx]:.3f},train_acc:{local_train_acc[idx]:.3f}.")

        print(' ' * 50)

        local_test_acc, local_test_loss = [], []
        for idx in range(args.num_users):
            if args.noniid == 1:
                test_acc, test_loss = test_inference(args=args, model=copy.deepcopy(local_model_list[idx]),
                                                     test_dataset=test_dataset, vs=vs)
            elif args.noniid == 2:
                test_acc, test_loss = test_inference(args=args, model=copy.deepcopy(local_model_list[idx]),
                                                     test_dataset=test_dataset, vs=vs, idxs=user_groups_lt[idx])
            local_test_acc.append(test_acc)
            local_test_loss.append(test_loss)

        test_loss_avg = np.mean(local_test_loss)
        test_loss_std = np.std(local_test_loss)
        test_acc_avg = np.mean(local_test_acc)
        test_acc_std = np.std(local_test_acc)
        print(
            f'test_loss_avg:{test_loss_avg:.3f}, test_loss_std:{test_loss_std:.3f};test_acc_avg:{test_acc_avg:.3f}, test_acc_std:{test_acc_std:.3f}')
        test_loss_list.append(round(test_loss_avg, 3))
        test_loss_std_list.append(round(test_loss_std, 3))
        test_accuracy_list.append(round(test_acc_avg, 3))
        test_accuracy_std_list.append(round(test_acc_std, 3))

        for idx in range(args.num_users):
            print(f"client{idx}:test_loss:{local_test_loss[idx]:.3f},test_acc:{local_test_acc[idx]:.3f}.")

        print(f'-' * 50)

        if args.vs == "tensorboard":
            vs.add_scalar('train_acc', train_acc_avg, r)
            vs.add_scalar('train_loss', train_loss_avg, r)
            vs.add_scalar('test_loss', test_loss_avg, r)
            vs.add_scalar('test_acc', test_acc_avg, r)
        elif args.vs == "visdom":
            vs.line([[train_loss_avg]], [r], win='train_loss', update='append')
            vs.line([[train_acc_avg]], [r], win='train_acc', update='append')
            vs.line([[test_loss_avg]], [r], win='test_loss', update='append')
            vs.line([[test_acc_avg]], [r], win='test_acc', update='append')

            # 可视化
            if r % 10 == 0:
                if args.noniid == 1:
                    vis_t_sne_lt(args=args, local_model_list=copy.deepcopy(local_model_list), test_dataset=test_dataset,
                                 r=r, vs=vs,
                                 colors=colors)
                elif args.noniid == 2:
                    vis_t_sne_lt(args=args, local_model_list=copy.deepcopy(local_model_list), test_dataset=test_dataset,
                              r=r, vs=vs,
                              colors=colors, user_groups_lt=user_groups_lt)


def FL(args, train_dataset, test_dataset, model, user_groups, vs, user_groups_lt=[]):
    train_loss_list, train_loss_std_list, train_accuracy_list, train_accuracy_std_list = [], [], [], []
    test_loss_list, test_loss_std_list, test_accuracy_list, test_accuracy_std_list = [], [], [], []
    colors = np.random.randint(0, 255, (args.num_classes, 3,))
    global_weights = model.state_dict()
    for r in tqdm(range(args.rounds)):
        local_weights, local_train_loss, local_train_acc = [], [], []

        print(f'\n | Global Training Round : {r + 1} |\n')

        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, train_dataset=train_dataset, idxs=user_groups[idx])
            w, train_loss, train_acc = local_model.update_weights(copy.deepcopy(model), vs)

            local_weights.append(copy.deepcopy(w))
            local_train_loss.append(copy.deepcopy(train_loss))
            local_train_acc.append(copy.deepcopy(train_acc))

        # weights_avg = average_weights(copy.deepcopy(local_weights))
        weights_avg = copy.deepcopy(local_weights[0])
        for k in weights_avg.keys():
            for i in range(1, len(local_weights)):
                weights_avg[k] += local_weights[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(local_weights))

        global_weights = weights_avg
        model.load_state_dict(global_weights)

        train_loss_avg = np.mean(local_train_loss)
        train_loss_std = np.std(local_train_loss)
        train_acc_avg = np.mean(local_train_acc)
        train_acc_std = np.std(local_train_acc)
        print(
            f"train_loss_avg:{train_loss_avg:.3f}, train_loss_std:{train_loss_std:.3f};train_acc_avg:{train_acc_avg:.3f}, train_acc_std:{train_acc_std:.3f}.")
        train_loss_list.append(round(train_loss_avg, 3))
        train_loss_std_list.append(round(train_loss_std, 3))
        train_accuracy_list.append(round(train_acc_avg, 3))
        train_accuracy_std_list.append(round(train_acc_std, 3))

        for idx in range(args.num_users):
            print(f"client{idx}:train_loss:{local_train_loss[idx]:.3f},train_acc:{local_train_acc[idx]:.3f}.")

        print(' ' * 50)

        local_test_acc, local_test_loss = [], []
        for idx in range(args.num_users):
            if args.noniid == 1:
                test_acc, test_loss = test_inference(args=args, model=copy.deepcopy(model),
                                                     test_dataset=test_dataset, vs=vs)
            elif args.noniid == 2:
                test_acc, test_loss = test_inference(args=args, model=copy.deepcopy(model),
                                                     test_dataset=test_dataset, vs=vs, idxs=user_groups_lt[idx])
            local_test_acc.append(test_acc)
            local_test_loss.append(test_loss)

        test_loss_avg = np.mean(local_test_loss)
        test_loss_std = np.std(local_test_loss)
        test_acc_avg = np.mean(local_test_acc)
        test_acc_std = np.std(local_test_acc)
        print(
            f'test_loss_avg:{test_loss_avg:.3f}, test_loss_std:{test_loss_std:.3f};test_acc_avg:{test_acc_avg:.3f}, test_acc_std:{test_acc_std:.3f}')
        test_loss_list.append(round(test_loss_avg, 3))
        test_loss_std_list.append(round(test_loss_std, 3))
        test_accuracy_list.append(round(test_acc_avg, 3))
        test_accuracy_std_list.append(round(test_acc_std, 3))

        for idx in range(args.num_users):
            print(f"client{idx}:test_loss:{local_test_loss[idx]:.3f},test_acc:{local_test_acc[idx]:.3f}.")

        print(f'-' * 50)

        if args.vs == "tensorboard":
            vs.add_scalar('train_acc', train_acc_avg, r)
            vs.add_scalar('train_loss', train_loss_avg, r)
            vs.add_scalar('test_loss', test_loss_avg, r)
            vs.add_scalar('test_acc', test_acc_avg, r)
        elif args.vs == "visdom":
            vs.line([[train_loss_avg]], [r], win='train_loss', update='append')
            vs.line([[train_acc_avg]], [r], win='train_acc', update='append')
            vs.line([[test_loss_avg]], [r], win='test_loss', update='append')
            vs.line([[test_acc_avg]], [r], win='test_acc', update='append')
            if r % 10 == 0:
                vis_t_sne(args=args, model=copy.deepcopy(model), test_dataset=test_dataset, r=r, vs=vs,
                          colors=colors)


if __name__ == "__main__":
    setup_seed(42)
    args = args_parser()
    train_dataset, test_dataset = get_dataset(args)

    if args.vs == "tensorboard":
        vs = SummaryWriter(
            './logs/' + args.dataset + f'_{args.alg}_' + 'noniid' + f'{args.noniid}' + f"_{args.num_users}_" + f"{datetime.datetime.now()}_")
    elif args.vs == "visdom":
        vs = Visdom()
        vs.line([[0.]], [0], win='train_loss', opts=dict(title='train_loss', legend=['train_loss']))
        vs.line([[0.]], [0], win='train_acc',
                opts=dict(title='train_acc', legend=['train_acc']))
        vs.line([[0.]], [0], win='test_loss', opts=dict(title='test_loss', legend=['test_loss']))
        vs.line([[0.]], [0], win='test_acc',
                opts=dict(title='test_acc', legend=['test_acc']))

    if args.noniid == 1:
        user_groups = get_user_groups(args, train_dataset, test_dataset)
        if args.alg == 'fedpn':
            local_model_list = []
            for i in range(args.num_users):
                local_model = CNN(args=args)
                local_model.to(args.device)
                local_model.train()
                local_model_list.append(local_model)
            FedPN(args, train_dataset, test_dataset, copy.deepcopy(local_model_list), user_groups, vs)
        elif args.alg == 'fedavg' or args.alg == 'fedprox':
            model = CNN(args=args)
            model = model.to(args.device)
            FL(args, train_dataset, test_dataset, copy.deepcopy(model), user_groups, vs)
    elif args.noniid == 2:
        user_groups, user_groups_lt = get_user_groups(args, train_dataset, test_dataset)
        if args.alg == 'fedpn':
            local_model_list = []
            for i in range(args.num_users):
                local_model = CNN(args=args)
                local_model.to(args.device)
                local_model.train()
                local_model_list.append(local_model)
            FedPN(args, train_dataset, test_dataset, copy.deepcopy(local_model_list), user_groups, vs, user_groups_lt)
        elif args.alg == 'fedavg' or args.alg == 'fedprox':
            model = CNN(args=args)
            model = model.to(args.device)
            FL(args, train_dataset, test_dataset, copy.deepcopy(model), user_groups, vs, user_groups_lt)

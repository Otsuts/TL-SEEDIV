import os
import time
import torch
import numpy as np
import random
import argparse
import torch.nn.functional as func

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def tsne_visualize(data, labels, title='scatter'):
    tsne = TSNE(n_components=2, random_state=233)
    data_2d = tsne.fit_transform(data)

    unique_labels = set(labels)

    # 定义不同类别的颜色
    colors = [plt.cm.get_cmap('jet')(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    # 绘制不同类别的数据点
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(data_2d[indices, 0], data_2d[indices, 1], c=[colors[i]], label=label)

    plt.title(title)
    plt.legend()
    plt.show()


def tsne_3d_visualize(data, labels, title='scatter'):
    tsne = TSNE(n_components=3, random_state=233)
    data_3d = tsne.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = set(labels)

    # 定义不同类别的颜色
    colors = [plt.cm.get_cmap('jet')(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    # 绘制不同类别的数据点
    for i, label in enumerate(unique_labels):
        indices = labels == label
        ax.scatter(data_3d[indices, 0], data_3d[indices, 1], data_3d[indices, 2], c=[colors[i]], label=label)

    # 添加标题和轴标签
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图像
    plt.legend()
    plt.show()


# Prepare data
def preprocess_data(train_data, train_labels, test_data, test_labels, device):
    train_data = torch.tensor(train_data).float()
    train_labels = torch.tensor(train_labels).long()
    test_data = torch.tensor(test_data).float()
    test_labels = torch.tensor(test_labels).long()

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return train_data.to(device), train_labels.to(device), test_data.to(device), test_labels.to(device)


# Training function
def train_model(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
                model, domain_adv, optimizer, lr_scheduler, epoch, device, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = func.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


# Testing function
def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = func.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


# 读取数据集和标签
def data_flow(path):
    full_train_data = []
    full_test_data = []
    for n_lab in range(1, 4):
        folder_path = f'{path}/{n_lab}'
        lab_train_data = []
        lab_test_data = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                file_path = os.path.join(root, dir)
                train_data = np.load(file_path + '/train_data.npy')
                train_label = np.load(file_path + '/train_label.npy')
                test_data = np.load(file_path + '/test_data.npy')
                test_label = np.load(file_path + '/test_label.npy')

                train_data = train_data.reshape((train_data.shape[0], -1))
                test_data = test_data.reshape((test_data.shape[0], -1))
                lab_train_data.append([train_data, train_label])
                lab_test_data.append([test_data, test_label])
        full_train_data.append(lab_train_data)
        full_test_data.append(lab_test_data)
    data = [full_train_data, full_test_data]
    return data


# 生成被试独立的数据（15）
def across_sub(data):
    across_data = np.zeros(15, dtype=list)
    full_train_data = data[0]
    full_test_data = data[1]
    sub_data = []
    sub_labels = []

    scaler = StandardScaler()
    for i in range(15):
        ses_data = []
        ses_labels = []
        for j in range(3):
            ses_dat = np.concatenate([full_train_data[j][i][0], full_test_data[j][i][0]])
            ses_label = np.concatenate([full_train_data[j][i][1], full_test_data[j][i][1]])
            scaler.fit(ses_dat)
            ses_dat = scaler.transform(ses_dat)
            ses_data.append(ses_dat)
            ses_labels.append(ses_label)

        sub_dat = np.concatenate(ses_data, axis=0)
        sub_label = np.concatenate(ses_labels, axis=0)

        sub_data.append(sub_dat)
        sub_labels.append(sub_label)

    for sub in range(15):
        train_sub = [i for i in range(15) if i != sub]
        test_sub = [sub]
        train_data = np.concatenate([sub_data[i] for i in train_sub], axis=0)
        train_labels = np.concatenate([sub_labels[i] for i in train_sub], axis=0)
        test_data = np.concatenate([sub_data[i] for i in test_sub], axis=0)
        test_labels = np.concatenate([sub_labels[i] for i in test_sub], axis=0)

        # sampling
        n_samples = train_data.shape[0]
        n_train_samples = 2560  # int(n_samples * 0.1)
        train_indices = random.sample(range(n_samples), n_train_samples)
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]

        n_samples = test_data.shape[0]
        n_test_samples = 2048   # int(n_samples * 0.5)
        test_indices = random.sample(range(n_samples), n_test_samples)
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]

        across_data[sub] = [train_data, train_labels, test_data, test_labels]

    return across_data

import os
import torch
import numpy as np
import random
import copy
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, ConcatDataset, Sampler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import higher
from torch.nn import functional

from tllib.utils.metric import accuracy


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

    return train_data, train_labels, test_data, test_labels


def train_adda(source_model, target_model, source_iter, target_iter, num_iter, criterion, domain_adv, optimizer,
               device):
    target_model.train()
    source_model.train()
    domain_adv.train()

    cls_acs = []
    domain_acs = []
    running_loss = []
    transfer_loss = []

    for i in range(num_iter):
        x_s, label_s = next(source_iter)[:2]
        x_t = next(target_iter)[:1][0]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        label_s = label_s.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            y_s, f_s = source_model(x_s)
        y_t, f_t = target_model(x_t)
        cls_loss = criterion(y_s, label_s)
        loss_transfer = domain_adv(f_s, f_t)
        optimizer.zero_grad()
        loss_transfer.backward()
        optimizer.step()

        running_loss.append(cls_loss.item())
        transfer_loss.append(loss_transfer.item())
        cls_acc = accuracy(y_s, label_s)[0].item()
        domain_acc = domain_adv.domain_discriminator_accuracy.item()
        cls_acs.append(cls_acc)
        domain_acs.append(domain_acc)

    cls_acc = np.mean(cls_acs)
    domain_acc = np.mean(domain_acs)
    running_loss = np.mean(running_loss)
    transfer_loss = np.mean(transfer_loss)
    return running_loss, transfer_loss, cls_acc, domain_acc


def pretrain_one_epoch(model, source_iter, target_loader, num_iter,
                       criterion, optimizer, device):
    '''
    pretrain model using training set and val set in source domain
    '''
    model.train()
    for i in range(num_iter):
        x_train, y_train = next(source_iter)[:2]
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    # 现在在验证集上验证准确率
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for inputs, targets in target_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100 * correct / total


# Training function

def train_plain(model, source_iter, target_iter, num_iter,
                criterion, domain_adv, optimizer, device):
    model.train()

    cls_acs = []
    domain_acs = []
    running_loss = []
    transfer_loss = []

    for i in range(num_iter):
        x_s, label_s = next(source_iter)[:2]
        x_t = next(target_iter)[:1][0]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        label_s = label_s.to(device)

        optimizer.zero_grad()

        y_s, f_s = model(x_s.float())
        y_t, f_t = model(x_t.float())
        cls_loss = criterion(y_s, label_s.long())
        tf_loss = 0
        loss = cls_loss
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        transfer_loss.append(tf_loss)
        cls_acc = accuracy(y_s, label_s)[0].item()
        domain_acc = 0
        cls_acs.append(cls_acc)
        domain_acs.append(domain_acc)

    cls_acc = np.mean(cls_acs)
    domain_acc = np.mean(domain_acs)
    running_loss = np.mean(running_loss)
    transfer_loss = np.mean(transfer_loss)
    return running_loss, transfer_loss, cls_acc, domain_acc


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


def get_low_bound(num, batch_size):
    num = num // batch_size
    return num * batch_size


def get_sub_data(data):
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

    return sub_data, sub_labels


def sampling(data, label, n, m):
    indices = random.sample(range(n), m)
    return data[indices], label[indices]


def get_dataset(sub_data, sub_labels, sub):
    train_sub = [i for i in range(15) if i != sub]
    train_data = np.concatenate([sub_data[i] for i in train_sub], axis=0)
    train_labels = np.concatenate([sub_labels[i] for i in train_sub], axis=0)
    num = train_data.shape[0]
    train_data, train_labels = sampling(train_data, train_labels, num, get_low_bound(int(num*0.2), 640))

    num = sub_data[sub].shape[0]
    test_data, test_labels = sampling(sub_data[sub], sub_labels[sub], num, get_low_bound(int(num), 512))

    train_data = torch.tensor(train_data).float()
    test_data = torch.tensor(test_data).float()
    train_labels = torch.tensor(train_labels).long()
    test_labels = torch.tensor(test_labels).long()

    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                      test_size=0.2, random_state=42)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    return [train_dataset, val_dataset, test_dataset]


def get_labeled_dataset(sub_data, sub_labels, sub):
    sub_data = [torch.tensor(sub_data[i]).float() for i in range(15)]
    sub_labels = [torch.tensor(sub_labels[i]).long() for i in range(15)]

    train_labeled_dataset = []
    test_labeled_dataset = None
    val_labeled_dataset = []

    sample_rate = 0.2
    for i in range(15):
        if i != sub:
            num = sub_data[i].shape[0]
            train_data, train_labels = sampling(sub_data[i], sub_labels[i], num, int(sample_rate*num))
            train_dataset = TensorDataset(train_data, train_labels)
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
            train_labeled_dataset.append(train_dataset)
            val_labeled_dataset.append(val_dataset)
        else:
            test_labeled_dataset = TensorDataset(sub_data[i], sub_labels[i])

    dataset = [ConcatDatasetWithDomainLabel(train_labeled_dataset),
               ConcatDataset(val_labeled_dataset), test_labeled_dataset]

    return dataset


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


class RandomDomainSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.
    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length

def random_split(x_list, labels_list, n_domains_per_batch, n_support_domains):
    assert n_support_domains < n_domains_per_batch

    support_domain_idxes = random.sample(range(n_domains_per_batch), n_support_domains)
    support_domain_list = [(x_list[idx], labels_list[idx]) for idx in range(n_domains_per_batch) if
                           idx in support_domain_idxes]
    query_domain_list = [(x_list[idx], labels_list[idx]) for idx in range(n_domains_per_batch) if
                         idx not in support_domain_idxes]

    return support_domain_list, query_domain_list

def train(train_iter, model, optimizer, n_domains, n_support, n_query, n_iter, n_inner_iter, trade_off, device):
    cls_accs = []
    losses = []

    model.train()

    for i in range(n_iter):
        x, labels, _ = next(train_iter)
        x = x.to(device)
        labels = labels.to(device)

        x_list = x.chunk(n_domains, dim=0)
        labels_list = labels.chunk(n_domains, dim=0)
        support_list, query_list = random_split(x_list, labels_list, n_domains, n_support)

        optimizer.zero_grad()

        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (inner_model, inner_optimizer):
            for _ in range(n_inner_iter):
                loss_inner = 0
                for (x_s, labels_s) in support_list:
                    y_s, _ = inner_model(x_s.float())
                    loss_inner += functional.cross_entropy(y_s, labels_s.long()) / n_support

                inner_optimizer.step(loss_inner)

            loss_outer = 0
            cls_acc = 0

            for (x_s, labels_s) in support_list:
                y_s, _ = model(x_s.float())
                loss_outer += functional.cross_entropy(y_s, labels_s.long()) / n_support

            for (x_q, labels_q) in query_list:
                y_q, _ = model(x_q.float())
                loss_outer += functional.cross_entropy(y_q, labels_q.long()) * trade_off / n_query
                cls_acc += accuracy(y_q, labels_q)[0] / n_query

        loss_outer.backward()
        optimizer.step()

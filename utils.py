import os
import torch
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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


# Training function
def train_model(model, source_iter, target_iter, num_iter,
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

        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        cls_loss = criterion(y_s, label_s)
        tf_loss = domain_adv(f_s, f_t)
        loss = cls_loss + 2*tf_loss
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        transfer_loss.append(tf_loss.item())
        cls_acc = accuracy(y_s, label_s)[0].item()
        domain_acc = domain_adv.domain_discriminator_accuracy.item()
        cls_acs.append(cls_acc)
        domain_acs.append(domain_acc)

    cls_acc = np.mean(cls_acs)
    domain_acc = np.mean(domain_acs)
    running_loss = np.mean(running_loss)
    transfer_loss = np.mean(transfer_loss)
    return running_loss, transfer_loss, cls_acc, domain_acc


# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predicted_labels = []  # Add this line to store the predicted labels

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Append the predicted labels to the list
            predicted_labels.extend(predicted.cpu().numpy())

    test_acc = 100 * correct / total
    return test_loss / len(test_loader), test_acc, predicted_labels  # Return the predicted_labels


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


# 生成被试依存的数据（3*15 = 45）
def depend_sub(data):
    depend_data = np.zeros((3, 15), dtype=list)
    full_train = data[0]
    full_test = data[1]
    scaler = StandardScaler()

    for ses in range(3):
        ses_train_data = np.concatenate([full_train[ses][sub][0] for sub in range(15)], axis=0)
        ses_test_data = np.concatenate([full_test[ses][sub][0] for sub in range(15)], axis=0)
        ses_data = np.concatenate([ses_train_data, ses_test_data])
        scaler.fit(ses_data)

        for sub in range(15):
            train_data = scaler.transform(full_train[ses][sub][0])
            train_label = full_train[ses][sub][1]
            test_data = scaler.transform(full_test[ses][sub][0])
            test_label = full_test[ses][sub][1]
            depend_data[ses][sub] = [train_data, train_label, test_data, test_label]

    return depend_data


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

        n_samples = train_data.shape[0]
        n_train_samples = get_low_bound(int(n_samples * 0.1), int(512*1.25))
        train_indices = random.sample(range(n_samples), n_train_samples)
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]

        n_samples = test_data.shape[0]
        n_test_samples = get_low_bound(int(n_samples), 512)
        test_indices = random.sample(range(n_samples), n_test_samples)
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]

        across_data[sub] = [train_data, train_labels, test_data, test_labels]

    return across_data

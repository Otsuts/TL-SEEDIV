import os
import argparse
from os import walk
import numpy as np
import torch.nn
import random
from tqdm import tqdm
from models import CNN1d, DANN
from utils import import_forder, set_seed, write_log
from dataloader import CNNDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dann', type=str)
    parser.add_argument('--dataset_path', default='./SEED-IV')
    parser.add_argument('--log_path', default='./logs')
    parser.add_argument('--scale', default=False, type=bool)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epoch_times', default=100, type=int)
    parser.add_argument('--test_iter', default=1, type=int)
    parser.add_argument('--tb_path', default='./logs/tensorboard/')
    parser.add_argument('--convolution_method', default='1d', type=str)
    return parser.parse_args()


class TrainCNN():
    def __init__(self, args):
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.args = args
        self.model = args.model
        self.precision_all = 0.0
        self.model_num = 0
        if args.model == 'plain':
            self.CNN = CNN1d().to(self.device)
        else:
            self.CNN = DANN().to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.domain_loss_func = torch.nn.NLLLoss()
        self.writer = SummaryWriter(args.tb_path)
        self.acc = 0.0
        self.train_func_dict = {
            'plain': self.train_plain,
            'dann': self.train_dann,
        }

    def train(self):
        self.train_func_dict[self.model]()

    def train_dann(self):
        write_log('Training started', self.args)
        for i in range(15):
            if self.args.model == 'plain':
                self.CNN = CNN1d().to(self.device)
            else:
                self.CNN = DANN().to(self.device)
            self.model_num += 1
            X_train, y_train, X_test, y_test = import_forder(
                i + 1, scale=self.args.scale)
            train_dataset = CNNDataset(data_path=None, label_path=None, data_numpy_array=X_train,
                                       label_numpy_array=y_train,
                                       convolution_method=self.args.convolution_method)
            test_dataset = CNNDataset(data_path=None, label_path=None, data_numpy_array=X_test,
                                      label_numpy_array=y_test,
                                      convolution_method=self.args.convolution_method)
            train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False)

            optimizer = Adam(self.CNN.parameters(), lr=self.args.learning_rate)
            best_test_acc = 0.0
            for epoch in tqdm(range(self.args.epoch_times)):
                len_dataloader = min(len(train_loader), len(test_loader))
                data_source_iter = iter(train_loader)
                data_target_iter = iter(test_loader)

                for i in range(len_dataloader):
                    p = float(i+epoch*len_dataloader) / \
                        self.args.epoch_times/len_dataloader

                    alpha = 2./(1.+np.exp(-10*p))-1

                    # training model using source data
                    data_source = data_source_iter.next()
                    s_input, s_label = data_source
                    s_input = s_input.to(self.device)

                    optimizer.zero_grad()
                    batch_size = len(s_label)
                    domain_label = torch.zeros(batch_size).long()

                    class_output, domain_output = self.CNN(
                        s_input, alpha)
                    loss_s_label = self.loss_func(
                        class_output.cpu(), s_label.long())
                    loss_s_domain = self.domain_loss_func(
                        domain_output.cpu(), domain_label)

                    # training model using target data
                    data_target = data_target_iter.next()
                    t_input, _ = data_target
                    t_input = t_input.to(self.device)
                    batch_size = len(t_input)
                    domain_label = torch.ones(batch_size).long()

                    _, domain_output = self.CNN(t_input, alpha=alpha)
                    loss_t_domain = self.domain_loss_func(
                        domain_output.cpu(), domain_label)
                    loss = loss_s_label + 2*loss_s_domain + 2*loss_t_domain
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % self.args.test_iter == 0:
                    self.CNN.eval()
                    test_loss = 0.0
                    correct = 0
                    with torch.no_grad():
                        for inputs, target in test_loader:
                            inputs = inputs.to(self.device)
                            target = target.unsqueeze(1)
                            # target是bs*1
                            outputs, _ = self.CNN(inputs, 0.0)  # bs*4
                            true_label = torch.zeros_like(target).tile(4)
                            for i, x in enumerate(target):
                                true_label[i][int(x.item())] = 1
                            loss = self.loss_func(true_label, outputs.cpu())
                            test_loss += loss.item()
                            pred = outputs.argmax(dim=-1).cpu()
                            correct += len(np.where(pred ==
                                           target.squeeze())[0])
                        precision = correct / len(test_dataset)
                        if precision > best_test_acc:
                            best_test_acc = precision
                        test_loss = test_loss / len(test_dataset)
                        # write_log(
                        #     f'test precision:{precision:.4f}, test loss:{test_loss:.4f}', self.args)
                        # self.writer.add_scalar(
                        #     f'Test/test_loss{i}', test_loss, epoch)
                        # self.writer.add_scalar(
                        #     f'Test/test_precision{i}', precision, epoch)
            self.acc += best_test_acc
            write_log(
                f'best test acc in model{self.model_num}: {best_test_acc}', self.args)
        write_log(self.acc / self.model_num, self.args)


    def train_plain(self):
        write_log('Training started', self.args)
        for i in range(15):
            self.CNN = CNN1d().to(self.device)
            self.model_num += 1
            X_train, y_train, X_test, y_test = import_forder(
                i + 1, scale=self.args.scale)
            train_dataset = CNNDataset(data_path=None, label_path=None, data_numpy_array=X_train,
                                       label_numpy_array=y_train,
                                       convolution_method=self.args.convolution_method)
            test_dataset = CNNDataset(data_path=None, label_path=None, data_numpy_array=X_test,
                                      label_numpy_array=y_test,
                                      convolution_method=self.args.convolution_method)
            train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False)

            optimizer = Adam(self.CNN.parameters(), lr=self.args.learning_rate)
            best_test_acc = 0.0
            # 开始训练
            for epoch in tqdm(range(self.args.epoch_times)):
                self.CNN.train()
                train_loss = 0.0
                correct = 0
                len_dataloader = min(len(train_loader), len(test_loader))
                data_source_iter = iter(train_loader)
                data_target_iter = iter(test_loader)

                for i in range(len_dataloader):
                    data_source = data_source_iter.next()
                    inputs, target = data_source
                    inputs = inputs.to(self.device)
                    target = target.unsqueeze(1)
                    # target是bs*1
                    optimizer.zero_grad()
                    outputs = self.CNN(inputs)  # bs*4
                    true_label = torch.zeros_like(target).tile(4)
                    for i, x in enumerate(target):
                        true_label[i][int(x.item())] = 1
                    loss = self.loss_func(true_label, outputs.cpu())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    pred = outputs.argmax(dim=-1).cpu()
                    correct += len(np.where(pred == target.squeeze())[0])
                # 每个epoch计算训练损失
                train_loss = train_loss / len(train_dataset)
                precision = correct / len(train_dataset)
                self.writer.add_scalar(
                    f'Train/train_loss{i}', train_loss, epoch)
                self.writer.add_scalar(
                    f'Train/train_presicion{i}', precision, epoch)
                # write_log(
                #     f'train precision:{precision:.4f}, train loss:{train_loss:.4f}', self.args)

                if (epoch + 1) % self.args.test_iter == 0:
                    self.CNN.eval()
                    test_loss = 0.0
                    correct = 0
                    with torch.no_grad():
                        for inputs, target in test_loader:
                            inputs = inputs.to(self.device)
                            target = target.unsqueeze(1)
                            # target是bs*1
                            outputs = self.CNN(inputs)  # bs*4
                            true_label = torch.zeros_like(target).tile(4)
                            for i, x in enumerate(target):
                                true_label[i][int(x.item())] = 1
                            loss = self.loss_func(true_label, outputs.cpu())
                            test_loss += loss.item()
                            pred = outputs.argmax(dim=-1).cpu()
                            correct += len(np.where(pred ==
                                           target.squeeze())[0])
                        precision = correct / len(test_dataset)
                        if precision > best_test_acc:
                            best_test_acc = precision
                        test_loss = test_loss / len(test_dataset)
                        # write_log(
                        #     f'test precision:{precision:.4f}, test loss:{test_loss:.4f}', self.args)
                        self.writer.add_scalar(
                            f'Test/test_loss{i}', test_loss, epoch)
                        self.writer.add_scalar(
                            f'Test/test_precision{i}', precision, epoch)
            self.acc += best_test_acc
            write_log(
                f'best test acc in model{self.model_num}: {best_test_acc}', self.args)
        write_log(self.acc / self.model_num, self.args)


def main(args):
    set_seed()
    trainer = TrainCNN(args)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)

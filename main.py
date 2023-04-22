import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import models
from utils import *
from sklearn.model_selection import train_test_split
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.classifier import Classifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    # 创建日志记录器
    logger = CompleteLogger(args.log, args.phase)

    # 1. 数据加载
    # 根据您的数据集进行数据处理和加载
    full_data = data_flow('SEED-IV')
    across_data = across_sub(full_data)
    acc_list = []
    for sub in range(15):
        train_data, train_labels, test_data, test_labels = across_data[sub]
        train_data, train_labels, test_data, test_labels = \
            preprocess_data(train_data, train_labels, test_data, test_labels, device)

        # Split Valid DataSets
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                          test_size=0.2, random_state=42)

        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        train_source_iter = ForeverDataIterator(train_loader)
        train_target_iter = ForeverDataIterator(test_loader)

        # 2. 创建模型
        # 根据任务选择合适的特征提取网络
        classifier = models.CNN1d().to(device)
        domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

        # 3. 定义损失函数和优化器
        optimizer = optim.Adam(classifier.get_parameters() + domain_discri.get_parameters()
                               , lr=args.lr, weight_decay=args.weight_decay)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                   lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
                                                       -args.lr_decay))
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)

        # 4. 训练和评估
        # 请根据您的情况编写train_model()和validate()函数
        best_acc1 = 0.
        for epoch in range(args.epochs):
            # 训练一个epoch
            train_model(train_source_iter, train_target_iter,
                        classifier, domain_adv, optimizer, lr_scheduler, epoch, device, args)

            # 在验证集上评估
            acc1 = validate(val_loader, classifier, args, device)

            # 更新最佳准确率
            if acc1 > best_acc1:
                best_acc1 = acc1
                torch.save(classifier.state_dict(), logger.get_checkpoint_path('best'))

        # 5. 测试
        classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
        test_acc1 = validate(test_loader, classifier, args, device)
        acc_list.append(test_acc1)
        print("For sub {}, test_acc = {:3.1f}".format(sub, test_acc1))

    print(f'The average acc is {np.mean(acc_list)}')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # model parameters
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
import optparse
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *
from models import *
from parameter import *
import argparse
import copy
from tllib.utils.logger import CompleteLogger
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_options():
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--log', action='store',
                            type=str, dest='log', default='logs')
    opt_parser.add_argument('--phase', action='store',
                            type=str, dest='phase', default='train')
    opt_parser.add_argument('-m', '--mode', action='store', type=str, dest='mode', default='across',
                            help='dependent or across')
    opt_parser.add_argument('-l', '--learning_rate', action='store',
                            type=float, dest='lr', default=LEARNING_RATE)
    opt_parser.add_argument('-b', '--batch_size', action='store',
                            type=float, dest='bz', default=BATCH_SIZE)
    opt_parser.add_argument('-e', '--epoch', action='store',
                            type=float, dest='epoch', default=NUM_EPOCH)
    opt_parser.add_argument('--pretrain_epochs', type=int,
                            default=PRETRAIN_EPOCH)
    opt_parser.add_argument('-w', '--weight_decay', action='store',
                            type=float, dest='wd', default=WEIGHT_DECAY)
    opt_parser.add_argument('-p', '--patience', action='store',
                            type=int, dest='pat', default=PATIENCE)
    opt_parser.add_argument('--pretrain_learning_rate', type=float, default=PRETRAIN_LEARNING_RATE)
    opts = opt_parser.parse_args()
    return opts


if __name__ == '__main__':
    args = parse_options()
    set_seed(233)

    # 定义网络参数
    DEPENDENT_NUM = 45
    ACROSS_NUM = 15
    results = []
    test_mode = args.mode
    learning_rate = args.lr
    batch_size = args.bz
    weight_decay = args.wd
    num_epoch = args.epoch
    patience = args.pat
    pretrain_learning_rate = args.pretrain_learning_rate

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = CompleteLogger(args.log, args.phase)
    full_data = data_flow('SEED-IV')

    if test_mode == 'dependent':
        data_set = depend_sub(full_data)
        num_model = DEPENDENT_NUM

    else:
        data_set = across_sub(full_data)
        num_model = ACROSS_NUM

    for num in range(num_model):
        if test_mode == 'dependent':
            train_data, train_labels, test_data, test_labels = data_set[num // 15][num % 15]
        else:
            train_data, train_labels, test_data, test_labels = data_set[num]

        # 获取数据并预处理
        raw_test_labels = test_labels
        train_data, train_labels, test_data, test_labels = \
            preprocess_data(train_data, train_labels,
                            test_data, test_labels, device)

        # Split Valid DataSets
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                          test_size=0.2, random_state=42)

        # Create Test DataLoaders
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        num_iter = len(train_dataset) // batch_size
        source_iter = ForeverDataIterator(train_loader)
        val_iter = ForeverDataIterator(val_loader)
        target_iter = ForeverDataIterator(test_loader)

        # 创建模型，优化器
        model = EEG_CNN(num_classes=4).to(device)
        optimizer_pretrain = optim.Adam(
            model.parameters(), lr=pretrain_learning_rate, weight_decay=args.wd)
        criterion_pretrain = nn.CrossEntropyLoss()

        # 预训练
        for epoch in range(args.pretrain_epochs):
            best_acc = 0.0
            val_acc = pretrain_one_epoch(model, source_iter, val_loader, num_iter,
                                         criterion_pretrain, optimizer_pretrain, device)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), logger.get_checkpoint_path('pretrain'))

        print('Pretraining process is done')

        checkpoint = torch.load(logger.get_checkpoint_path('pretrain'), )
        # model就是source_classifier,需要锁住他的参数
        model.load_state_dict(checkpoint)
        target_model = copy.deepcopy(model)
        for parameter in model.parameters():
            parameter.requires_grad = False

        # model = CNN1d().to(device)
        adapter = DomainDiscriminator(
            in_feature=model.features_dim, hidden_size=128).to(device)
        grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([{'params': target_model.parameters()}, {'params': adapter.parameters(
        ), 'lr': 5e-4}], lr=learning_rate, weight_decay=weight_decay)
        domain_adv = DomainAdversarialLoss(adapter, grl=grl).to(device)

        cls_acc, domain_acc, train_loss = 0, 0, 0
        val_acc, val_loss = 0, 0

        # Train the model
        min_val_loss = float('inf')
        count = 0
        best_acc = 0
        tf_losses = []
        tol_losses = []

        for epoch in tqdm(range(num_epoch)):
            # Train in train dataset
            train_loss, tf_loss, cls_acc, domain_acc = train_adda(model, target_model, source_iter, target_iter,
                                                                  num_iter, criterion, domain_adv, optimizer, device)
            tf_losses.append(tf_loss)
            tol_losses.append(train_loss)
            # Evaluate in valid dataset
            if (epoch + 1) % 2 == 0:
                val_loss, val_acc, _ = test_model(
                    target_model, val_loader, criterion, device)
                best_acc = max(val_acc, best_acc)
                if val_loss < min_val_loss:
                    count = 0
                    min_val_loss = val_loss
                    torch.save(target_model.state_dict(),
                               logger.get_checkpoint_path('best'))
                    torch.save(adapter.state_dict(),
                               logger.get_checkpoint_path('adapt'))
                else:
                    count += 1
                    if count >= patience:
                        break

        tqdm.write(f'In train No. {num}, Cls Acc: {cls_acc:.4f}%, Domain Acc: {domain_acc:.4f}%, '
                   f'loss: {np.mean(tf_losses):.4f}, {np.mean(tol_losses):.4f}, Val Acc: {val_acc:.4f}%')

        # Evaluate the model
        target_model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
        adapter.load_state_dict(torch.load(
            logger.get_checkpoint_path('adapt')))
        test_loss, test_acc, label_pred = test_model(
            target_model, test_loader, criterion, device)
        # tsne_3d_visualize(test_data.view(test_data.shape[0], -1), label_pred, 'CNN Prediction Labels')
        # tsne_3d_visualize(test_data.view(test_data.shape[0], -1), raw_test_labels, 'Real Labels')
        tqdm.write(
            f'The accuracy in test {num} is {test_acc:.4f}%, the loss is {test_loss:.4f}')
        results.append(test_acc)

    # Calculate the average accuracy for subject-dependent condition
    accuracy = np.mean(results)
    print(f"Subject-dependent accuracy: {accuracy:.2f}%")
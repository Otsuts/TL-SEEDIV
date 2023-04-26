import optparse
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *
from models import *
from parameter import *

from tllib.utils.logger import CompleteLogger
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.dann import DomainAdversarialLoss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('--log', action='store', type='string', dest='log', default='logs')
    opt_parser.add_option('--phase', action='store', type='string', dest='phase', default='train')
    opt_parser.add_option('-m', '--mode', action='store', type='string', dest='mode', default='across',
                          help='dependent or across')
    opt_parser.add_option('-l', '--learning_rate', action='store', type='float', dest='lr', default=LEARNING_RATE)
    opt_parser.add_option('-b', '--batch_size', action='store', type='int', dest='bz', default=BATCH_SIZE)
    opt_parser.add_option('-e', '--epoch', action='store', type='int', dest='epoch', default=NUM_EPOCH)
    opt_parser.add_option('-w', '--weight_decay', action='store', type='float', dest='wd', default=WEIGHT_DECAY)
    opt_parser.add_option('-p', '--patience', action='store', type='int', dest='pat', default=PATIENCE)
    opts, _ = opt_parser.parse_args()
    return opts


if __name__ == '__main__':
    args = parse_options()
    setup_seed(233)

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

    n_support = 2
    n_query = 2
    n_domains = n_support + n_query

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = CompleteLogger(args.log, args.phase)
    full_data = data_flow('SEED-IV')

    num_model = ACROSS_NUM
    data_set = get_labeled_dataset(full_data)

    for sub in range(num_model):

        # 创建模型，优化器
        model = EEG_CNN(num_classes=4).to(device)
        # model = CNN1d().to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam([{'params': model.parameters()}]
        #                        , lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Create Test DataLoaders
        train_dataset, val_dataset, test_dataset = data_set[sub]
        sampler = RandomDomainSampler(train_dataset, batch_size, n_domains)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_iter = len(train_dataset) // batch_size
        source_iter = ForeverDataIterator(train_loader)

        val_acc, val_loss = 0, 0

        # Train the model
        min_val_loss = float('inf')
        count = 0
        best_acc = 0
        tf_losses = []
        tol_losses = []
        n_inner_iter = 1
        trade_off = 2
        for epoch in tqdm(range(num_epoch)):
            # Train in train dataset
            train(source_iter, model, optimizer, n_domains, n_support, n_query,
                  num_iter, n_inner_iter, trade_off, device)
            # Evaluate in valid dataset
            if (epoch + 1) % 2 == 0:
                val_loss, val_acc, _ = test_model(model, val_loader, criterion, device)
                best_acc = max(val_acc, best_acc)
                if val_loss < min_val_loss:
                    count = 0
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), logger.get_checkpoint_path('best'))
                else:
                    count += 1
                    if count >= patience:
                        break

        tqdm.write(f'In train No. {sub}, Val Acc: {best_acc:.4f}%')

        # Evaluate the model
        model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
        test_loss, test_acc, label_pred = test_model(model, test_loader, criterion, device)
        # tsne_3d_visualize(test_data.view(test_data.shape[0], -1), label_pred, 'CNN Prediction Labels')
        # tsne_3d_visualize(test_data.view(test_data.shape[0], -1), raw_test_labels, 'Real Labels')
        tqdm.write(f'The accuracy in test {sub} is {test_acc:.4f}%, the loss is {test_loss:.4f}')
        results.append(test_acc)

    # Calculate the average accuracy for subject-dependent condition
    accuracy = np.mean(results)
    print(f"Subject-dependent accuracy: {accuracy:.2f}%")



import argparse
from utils import *
from models import *
from parameters import *
from methods import DANN, ADDA, MixStyle, MLDG

from tllib.utils.logger import CompleteLogger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_options():
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--model', type=str, default='dann')
    opt_parser.add_argument('--log', action='store', type=str, dest='log', default='logs')
    opt_parser.add_argument('--phase', action='store', type=str, dest='phase', default='train')
    opt_parser.add_argument('-m', '--mode', action='store', type=str, dest='mode', default='across',
                            help='dependent or across')
    opt_parser.add_argument('-l', '--learn_rate', action='store', type=str, dest='learn_rate', default=LEARNING_RATE)
    opt_parser.add_argument('-b', '--batch_size', action='store', type=str, dest='batch_size', default=BATCH_SIZE)
    opt_parser.add_argument('-e', '--num_epoch', action='store', type=str, dest='num_epoch', default=NUM_EPOCH)
    opt_parser.add_argument('-w', '--weight_decay', action='store', type=str, dest='weight_decay', default=WEIGHT_DECAY)
    opt_parser.add_argument('-p', '--patience', action='store', type=str, dest='patience', default=PATIENCE)
    opt_parser.add_argument('-n', '--num_class', action='store', type=str, dest='num_class', default=4)
    opt_parser.add_argument('-t', '--trade_off', action='store', type=str, dest='trade_off', default=2)
    opt_parser.add_argument('--pretrain_epoch', type=str, default=100)
    opt_parser.add_argument('--pretrain_lr', type=float, default=5e-4)
    opts = opt_parser.parse_args()
    return opts


if __name__ == '__main__':
    args = parse_options()
    setup_seed(233)

    # 定义网络参数
    results = []
    logger = CompleteLogger(args.log, args.phase)
    full_data = data_flow('SEED-IV')
    sub_data, sub_labels = get_sub_data(full_data)
    num_model = 15

    for sub in range(num_model):
        if args.model == 'dann' or args.model == 'adda':
            dataset = get_dataset(sub_data, sub_labels, sub)
        else:
            dataset = get_labeled_dataset(sub_data, sub_labels, sub)
        if args.model == 'dann':
            worker = DANN(args, dataset, sub, logger)
        elif args.model == 'adda':
            worker = ADDA(args, dataset, sub, logger)
        elif args.model == 'mixstyle':
            worker = MLDG(args, dataset, sub, logger)
        else:
            raise f'Model {args.model} not supported'
        test_acc = worker.work()
        results.append(test_acc)

    # Calculate the average accuracy for subject-dependent condition
    accuracy = np.mean(results)
    print(f"Subject-dependent accuracy: {accuracy:.2f}%")

import copy

from models import *
from utils import *
from tqdm import tqdm
from abc import abstractmethod
import torch.optim as optim
from torch.utils.data import DataLoader

from tllib.utils.logger import CompleteLogger
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.utils.metric import accuracy


class TrainBase:
    def __init__(self, args, dataset, sub, logger):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger

        self.model = EEG_CNN(self.args.num_class).to(self.device)
        self.sub = sub
        self.train_dataset, self.val_dataset, self.test_dataset, self.test_data = dataset
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        self.domain = False
        self.criterion = nn.CrossEntropyLoss()
        self.bz = self.args.batch_size
        self.lr = self.args.learn_rate
        self.wd = self.args.weight_decay
        self.epoch = self.args.num_epoch
        self.patience = self.args.patience

    @abstractmethod
    def train_model(self):
        pass

    def work(self):
        min_loss = float('inf')
        count, best_acc = 0, 0
        cls_acc, domain_acc = 0, 0
        losses = []

        for epoch in tqdm(range(self.epoch)):
            train_loss, cls_acc, domain_acc = self.train_model()
            losses.append(train_loss)

            if (epoch + 1) % 2 == 0:
                val_loss, val_acc, _ = self.validate(mode='val')
                best_acc = max(val_acc, best_acc)
                if val_loss < min_loss:
                    count = 0
                    min_loss = val_loss
                    torch.save(self.model.state_dict(), self.logger.get_checkpoint_path('best'))
                else:
                    count += 1
                    if count >= self.patience:
                        break

        if self.domain:
            tqdm.write(f'In train No. {self.sub}, Cls Acc: {cls_acc:.4f}%, Domain Acc: {domain_acc:.4f}%, '
                       f'loss: {np.mean(losses):.4f}, Val Acc: {best_acc:.4f}%')
        else:
            tqdm.write(f'In train No. {self.sub}, Cls Acc: {cls_acc:.4f}%, '
                       f'loss: {np.mean(losses):.4f}, Val Acc: {best_acc:.4f}%')

        self.model.load_state_dict(torch.load(self.logger.get_checkpoint_path('best')))
        test_loss, test_acc, label_pred = self.validate(mode='test')
        tsne_visualize(self.test_data, label_pred, f'TL Prediction Labels ({self.args.model})')
        # tsne_3d_visualize(test_data.view(test_data.shape[0], -1), raw_test_labels, 'Real Labels')
        tqdm.write(f'The accuracy in test {self.sub} is {test_acc:.4f}%, the loss is {test_loss:.4f}')

        return test_acc

    def validate(self, mode='val'):
        self.model.eval()
        device = self.device

        losses = 0
        correct = 0
        cnt = 0
        predicted_labels = []

        if mode == 'val':
            loader = self.val_loader
        else:
            loader = self.test_loader

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                losses += loss.item()
                _, predicted = outputs.max(1)
                cnt += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                predicted_labels.extend(predicted.cpu().numpy())

        acc = 100 * correct / cnt
        return losses / len(loader), acc, predicted_labels


class BaseLine(TrainBase):
    def __init__(self, args, dataset, sub, logger):
        super(BaseLine, self).__init__(args, dataset, sub, logger)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bz, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bz, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bz, shuffle=False)

        self.num_iter = len(self.train_dataset) // self.bz
        self.train_iter = ForeverDataIterator(self.train_loader)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def train_model(self):
        self.model.train()
        cls_acs = []
        running_loss = []

        for i in range(self.num_iter):
            x_s, label_s = next(self.train_iter)[:2]
            x_s, label_s = x_s.to(self.device), label_s.to(self.device)

            self.optimizer.zero_grad()

            y_s, _ = self.model(x_s)

            cls_loss = self.criterion(y_s, label_s)
            loss = cls_loss
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            cls_acc = accuracy(y_s, label_s)[0].item()
            cls_acs.append(cls_acc)

        cls_acc = np.mean(cls_acs)
        running_loss = np.mean(running_loss)
        return running_loss, cls_acc, 0


class DANN(TrainBase):
    def __init__(self, args, dataset, sub, logger):
        super(DANN, self).__init__(args, dataset, sub, logger)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bz, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bz, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bz, shuffle=False)

        self.domain = True
        self.trade_off = self.args.trade_off
        self.num_iter = len(self.train_dataset) // self.bz
        self.source_iter = ForeverDataIterator(self.train_loader)
        self.target_iter = ForeverDataIterator(self.test_loader)

        self.adapter = DomainDiscriminator(in_feature=self.model.features_dim,
                                           hidden_size=64, batch_norm=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.adapter.parameters(), 'lr': 5e-4}],
                                    lr=self.lr, weight_decay=self.wd)
        self.domain_adv = DomainAdversarialLoss(self.adapter).to(self.device)

    def train_model(self):
        self.model.train()
        cls_acs = []
        domain_acs = []
        running_loss = []

        for i in range(self.num_iter):
            x_s, label_s = next(self.source_iter)[:2]
            x_t = next(self.target_iter)[:1][0]
            x_s, x_t, label_s = x_s.to(self.device), x_t.to(self.device), label_s.to(self.device)

            self.optimizer.zero_grad()

            y_s, f_s = self.model(x_s)
            y_t, f_t = self.model(x_t)

            cls_loss = self.criterion(y_s, label_s)
            tf_loss = self.domain_adv(f_s, f_t)
            loss = cls_loss + self.trade_off * tf_loss
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            cls_acc = accuracy(y_s, label_s)[0].item()
            domain_acc = self.domain_adv.domain_discriminator_accuracy.item()
            cls_acs.append(cls_acc)
            domain_acs.append(domain_acc)

        cls_acc = np.mean(cls_acs)
        domain_acc = np.mean(domain_acs)
        running_loss = np.mean(running_loss)
        return running_loss, cls_acc, domain_acc


class ADDA(TrainBase):
    def __init__(self, args, dataset, sub, logger):
        super(ADDA, self).__init__(args, dataset, sub, logger)
        self.lr = 1e-4
        self.pretrain_lr = args.pretrain_lr
        self.pretrain_epoch = args.pretrain_epoch
        self.optimizer_pretrain = optim.Adam(self.model.parameters(), lr=self.pretrain_lr,
                                             weight_decay=args.weight_decay)
        self.criterion_pretrain = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bz, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bz, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bz, shuffle=False)

        self.domain = True
        self.trade_off = self.args.trade_off
        self.num_iter = len(self.train_dataset) // self.bz
        self.source_iter = ForeverDataIterator(self.train_loader)
        self.target_iter = ForeverDataIterator(self.test_loader)
        self.optimizer = None
        self.adapter = None
        self.criterion = nn.CrossEntropyLoss()

        self.domain_adv = None
        self.target_model = None
        self.pretrain_and_init()

    def pretrain_one_epoch(self):
        self.model.train()
        for i in range(self.num_iter):
            x_train, y_train = next(self.source_iter)[:2]
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            self.optimizer_pretrain.zero_grad()
            y_pred, _ = self.model(x_train)
            loss = self.criterion_pretrain(y_pred, y_train)
            loss.backward()
            self.optimizer_pretrain.step()

        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            total = 0
            correct = 0
            for inputs, target in self.val_loader:
                inputs, targets = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion_pretrain(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100 * correct / total

    def pretrain_and_init(self):
        best_pretrain_acc = 0.0
        for epoch in range(self.pretrain_epoch):
            val_acc = self.pretrain_one_epoch()
            if val_acc > best_pretrain_acc:
                best_pretrain_acc = val_acc
                torch.save(self.model.state_dict(), self.logger.get_checkpoint_path('pretrain'))

        checkpoint = torch.load(self.logger.get_checkpoint_path('pretrain'))
        self.model.load_state_dict(checkpoint)
        self.target_model = copy.deepcopy(self.model)
        for parameter in self.target_model.parameters():
            parameter.requires_grad = False

        self.adapter = DomainDiscriminator(in_feature=self.model.features_dim, hidden_size=128).to(self.device)
        grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)
        self.domain_adv = DomainAdversarialLoss(self.adapter, grl=grl).to(self.device)
        self.optimizer = optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.adapter.parameters(), 'lr': 5e-4}],
                                    lr=self.lr, weight_decay=self.wd)

    def train_model(self):

        self.model.train()
        self.target_model.train()
        self.domain_adv.train()
        cls_acs = []
        domain_acs = []
        running_loss = []
        transfer_loss = []
        for i in range(self.num_iter):
            x_s, label_s = next(self.source_iter)[:2]
            x_t = next(self.target_iter)[:1][0]

            x_s = x_s.to(self.device)
            x_t = x_t.to(self.device)
            label_s = label_s.to(self.device)

            self.optimizer.zero_grad()
            with torch.no_grad():
                y_s, f_s = self.target_model(x_s)
            y_t, f_t = self.model(x_t)
            cls_loss = self.criterion(y_s, label_s)
            loss_transfer = self.domain_adv(f_s, f_t)
            self.optimizer.zero_grad()
            loss_transfer.backward()
            self.optimizer.step()

            running_loss.append(cls_loss.item())
            transfer_loss.append(loss_transfer.item())
            cls_acc = accuracy(y_s, label_s)[0].item()
            domain_acc = self.domain_adv.domain_discriminator_accuracy.item()
            cls_acs.append(cls_acc)
            domain_acs.append(domain_acc)

        cls_acc = np.mean(cls_acs)
        domain_acc = np.mean(domain_acs)
        running_loss = np.mean(running_loss)
        transfer_loss = np.mean(transfer_loss)
        return running_loss, cls_acc, domain_acc


class MixStyle(TrainBase):
    def __init__(self, args, dataset, sub, logger):
        super(MixStyle, self).__init__(args, dataset, sub, logger)
        self.adapter = DomainDiscriminator(in_feature=self.model.features_dim, hidden_size=128).to(self.device)
        self.domain_adv = DomainAdversarialLoss(self.adapter).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([{'params': self.model.parameters()}], lr=self.lr, weight_decay=self.wd)
        self.sampler = RandomDomainSampler(self.train_dataset, self.bz, n_domains_per_batch=2)

        self.train_loader = DataLoader(self.train_dataset, self.bz, sampler=self.sampler)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bz, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bz, shuffle=False)
        self.num_iter = len(self.train_dataset) // self.bz
        self.source_iter = ForeverDataIterator(self.train_loader)
        self.target_iter = ForeverDataIterator(self.test_loader)

    def train_model(self):
        self.model.train()

        cls_acs = []
        domain_acs = []
        running_loss = []
        transfer_loss = []

        for i in range(self.num_iter):
            x_s, label_s = next(self.source_iter)[:2]
            x_t = next(self.target_iter)[:1][0]

            x_s = x_s.to(self.device)
            x_t = x_t.to(self.device)
            label_s = label_s.to(self.device)

            self.optimizer.zero_grad()

            y_s, f_s = self.model(x_s.float())
            y_t, f_t = self.model(x_t.float())
            cls_loss = self.criterion(y_s, label_s.long())
            tf_loss = 0
            loss = cls_loss
            loss.backward()
            self.optimizer.step()

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
        return running_loss, cls_acc, domain_acc


class MLDG(TrainBase):
    def __init__(self, args, dataset, sub, logger):
        super(MLDG, self).__init__(args, dataset, sub, logger)
        self.n_support = args.num_support
        self.n_domain = args.num_domain
        self.n_query = self.n_domain - self.n_support
        self.inner_iter = 2
        sampler = RandomDomainSampler(self.train_dataset, self.bz, self.n_domain)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bz, sampler=sampler, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bz, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bz, shuffle=False)

        self.trade_off = self.args.trade_off
        self.num_iter = len(self.train_dataset) // self.bz
        self.train_iter = ForeverDataIterator(self.train_loader)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def train_model(self):
        self.model.train()
        cls_acs = []
        running_loss = []

        for i in range(self.num_iter):
            x, label = next(self.train_iter)[:2]
            x, label = x.to(self.device), label.to(self.device)

            x_list = x.chunk(self.n_domain, dim=0)
            labels_list = label.chunk(self.n_domain, dim=0)
            support_list, query_list = random_split(x_list, labels_list, self.n_domain, self.n_support)

            self.optimizer.zero_grad()

            loss_outer = 0
            cls_acc = 0

            with higher.innerloop_ctx(self.model, self.optimizer, copy_initial_weights=False) \
                    as (inner_model, inner_optimizer):
                for _ in range(self.inner_iter):
                    loss_inner = 0
                    for (x_s, labels_s) in support_list:
                        y_s, _ = inner_model(x_s)
                        loss_inner += self.criterion(y_s, labels_s) / self.n_support

                    inner_optimizer.step(loss_inner)

                for (x_s, labels_s) in support_list:
                    y_s, _ = self.model(x_s)
                    loss_outer += self.criterion(y_s, labels_s) / self.n_support

                for (x_q, labels_q) in query_list:
                    y_q, _ = self.model(x_q)
                    loss_outer += self.criterion(y_q, labels_q) * self.trade_off / self.n_query
                    cls_acc += accuracy(y_q, labels_q)[0] / self.n_query

            loss_outer.backward()
            self.optimizer.step()
            cls_acs.append(cls_acc.item())
            running_loss.append(loss_outer.item())

        cls_acc = np.mean(cls_acs)
        running_loss = np.mean(running_loss)
        return running_loss, cls_acc, 0

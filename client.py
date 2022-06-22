import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler



class Client():
    def __init__(self, id, init_args: dict):
        self.id = id

        self.train_dataset = None
        self.test_dataset = None
        self.device = 0
        self.net = init_args['net']

        self.num_epochs = init_args['epoch']
        self.batch_size = init_args['batch']

        self.train_iter = None
        self.test_iter = None

        self.logger = init_args['logger']



    def load_data(self, train_dataset, test_datset):

        loader_args = dict(batch_size=self.batch_size,
                           num_workers=2, pin_memory=True)

        self.train_dataset = train_dataset
        self.test_dataset = test_datset

        self.train_iter = DataLoader(
            train_dataset, shuffle=True, **loader_args)
        self.test_iter = DataLoader(
            test_datset, shuffle=False, **loader_args)

    def loss(self, y, y_label):
        cel = nn.CrossEntropyLoss()
        return cel(y, y_label)


    def train(self):
        assert self.train_iter != None
        assert self.test_iter != None

        net = self.net.to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False)
        num_epochs = self.num_epochs
        train_iter = self.train_iter
        device = self.device
        scaler = GradScaler() 

        net.train()

        for epoch in tqdm(range(num_epochs)):
            for x, y in train_iter:
                x, y = x.to(device),  y.to(device)
                optimizer.zero_grad()

                with autocast():
                    l = self.loss(net(x), y)
                
                scaler.scale(l).backward()
        
                scaler.step(optimizer)
                scaler.update()

            if epoch % 2 == 0:
                self.evaluate(False, epoch)


    def evaluate(self, is_train: bool, epoch):
        net = self.net

        net.eval()
        cal_acc = Accuracy()
        acc = 0
        l = 0
        device = self.device

        if is_train:
            data_iter = self.train_iter
        else:
            data_iter = self.test_iter

        with torch.no_grad():
            for x, y in data_iter:
                x, y = x.to(device),  y.to(device)
                preds = net(x)
                l += self.loss(preds, y).item()
                preds_pr = nn.functional.softmax(preds, dim=1)
                # e.g. [[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]] | [0, 1, 2]
                cal_acc.update(preds_pr.cpu(), y.cpu())

        acc = cal_acc.compute()
        l /= len(data_iter)

        self.logger.info('epoch: {}, loss: {}, acc: {}'.format(epoch, l, acc))

        return l, acc

    def prediction(self, x):
        self.net.eval()
        return nn.functional.softmax(self.net(x), dim=1)

    def get_weight(self):
        if hasattr(self.net, 'module'):
            w = self.net.module.state_dict()
        else:
            w = self.net.state_dict()
        w = copy.deepcopy(w)
        return w

    def load_weight(self, weight):
        if hasattr(self.net, 'module'):
            self.net.module.load_state_dict(weight)
        else:
            self.net.load_state_dict(weight)

    def data_len(self):
        return len(self.train_dataset) + len(self.test_dataset)




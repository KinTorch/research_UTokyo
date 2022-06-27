import torch
from init import client_manager, mixed_dataset
from client import Client
from models.resnet18 import Resnet
import random
import torch.nn as nn
import os
from torchmetrics import Accuracy
import time
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pickle
import socket
from tqdm import tqdm
from logger import init_logger
import hashlib
from datetime import datetime

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]

    return devices if devices else [torch.device('cpu')]


def FedAvg(w):
    with torch.no_grad():
        w_avg = w[0]
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]

            w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def aggregate(clients):
    clients_weight = [clients[id].get_weight() for id in selected_clients]
    global_weight = FedAvg(clients_weight)
    return global_weight


def spread(clients, global_weight):
    for c in clients:
        c.load_weight(global_weight)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


def train_client(id):
    clients[id].load_weight(global_weight)
    clients[id].train()


def global_eval(net, data_iter, ge):
    def loss(y, y_label):
        cel = nn.CrossEntropyLoss()
        return cel(y, y_label)

    cal_acc = Accuracy()
    acc = 0
    l = 0

    net.to(0)
    net.eval()

    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(0),  y.to(0)
            preds = net(x)
            l += loss(preds, y).item()
            preds_pr = nn.functional.softmax(preds, dim=1)
            # e.g. [[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]] | [0, 1, 2]
            cal_acc.update(preds_pr.cpu(), y.cpu())

    acc = cal_acc.compute()
    l /= len(data_iter)

    logger.info('global epoch: {}, loss: {}, acc: {}'.format(ge ,l, acc))


def connect_client(id):

    connector, _ = sk.accept()
    logger.debug('waiting for {}'.format(id))

    data = pickle.dumps(clients[id], protocol=4)

    #print(hashlib.md5(data).hexdigest())
    data = data + 'end'.encode()
    
    connector.send(data)

    rev_data = bytes()
    
    while True:
        temp = connector.recv(10485760)
        rev_data += temp
        if rev_data.endswith('end'.encode()):
            rev_data = rev_data[:-3]
            break

    rev_data = pickle.loads(rev_data)

    clients[id].load_weight(rev_data)
    logger.debug('{} received'.format(id))

    del rev_data
    del data


def init_global_data():
    transform1 = transforms.Compose([transforms.Resize(32), transforms.Grayscale(
        num_output_channels=3), transforms.ToTensor()])
    data1 = datasets.MNIST('./datasets/mnist', train=False,
                           transform=transform1, download=False)

    data2 = datasets.SVHN(
        './datasets/svhn', split='test', transform=transforms.ToTensor(), download=False)

    eval_data = mixed_dataset(data1, data2)

    loader_args = dict(batch_size=256, num_workers=4, pin_memory=True)

    data_iter = DataLoader(eval_data, shuffle=False, **loader_args)

    return data_iter


if __name__ == '__main__':
    meta_logger = init_logger(0)
    logger = meta_logger.get_logger()
    LO = logger.get_logger()
    sk = socket.socket()
    sk.bind(('localhost', 8888))
    sk.listen()
    logger.info('start socket')
    net = Resnet()
    manager = client_manager(num_client=30,
                             net=net, epoch=5, batch=256, time = datetime.now())
    clients = manager.clients

    manager.load_dataset(['mnist'], range(0, 15), True, -1)
    data_size = clients[0].data_len()
    manager.load_dataset(['svhn'], range(15, 30), True, data_size)

    ids = [i for i in range(len(clients))]

    num_select = 30
    max_workers = 5
    global_epoch = 1
    net.apply(init_weights)
    global_weight = net.state_dict()

    tick = time.time()
    for ge in tqdm(range(global_epoch)):
        random.shuffle(ids)
        selected_clients = ids[0:num_select]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(connect_client, selected_clients)

        global_weight = aggregate(clients)

        spread(clients, global_weight)

        net.load_state_dict(global_weight)
        global_eval(net, init_global_data(), ge)

    tock = time.time()

    logger.info('{} time consumed'.format(tock-tick))

    torch.save(net.state_dict(), meta_logger.get_path())

    logger.info('model saved')



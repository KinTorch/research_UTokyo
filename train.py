from matplotlib.pyplot import connect
import torch
from init import client_manager
from client import Client
from models.resnet18 import Resnet
import random
import torch.nn as nn
import copy
from torchmetrics import Accuracy
import time
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
import pickle
import socket
from tqdm import tqdm

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


def global_eval(net, is_train = False):
    def loss(y, y_label):
        cel = nn.CrossEntropyLoss()
        return cel(y, y_label)

    cal_acc = Accuracy()
    acc = 0
    l = 0
    
    net.to(0)
    net.eval()

    if is_train:
        data = manager.dataset_train
    else:
        data = manager.dataset_test

    loader_args = dict(batch_size=256,num_workers=4, pin_memory=True)

    data_iter = DataLoader(data, shuffle=False, **loader_args)


    with torch.no_grad():
        for x,y in data_iter:
            x,y = x.to(0),  y.to(0)
            preds = net(x)
            l += loss(preds, y).item()
            preds_pr = nn.functional.softmax(preds, dim=1)
            cal_acc.update(preds_pr.cpu(), y.cpu()) #e.g. [[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]] | [0, 1, 2]

    acc = cal_acc.compute()
    l /= len(data_iter)
    
    print('loss: {}, acc: {}'.format(l, acc))

def connect_client(id):
    #print('start listen')

    connector, _ = sk.accept()
    data = pickle.dumps(clients[id])
    
    data = data + 'end'.encode()
    connector.send(data)

    rev_data = bytes()
    print('receiving client ', id)
    while True:
        temp = connector.recv(10485760)
        rev_data += temp
        if rev_data.endswith('end'.encode()):
            rev_data = rev_data[:-3]
            break
    
    rev_data = pickle.loads(rev_data)

    clients[id].load_weight(rev_data)

    del rev_data
    del data


        



if __name__ == '__main__':
    sk = socket.socket()
    sk.bind(('localhost', 8888))
    sk.listen()

    net = Resnet()
    manager = client_manager(num_client = 200, iid = True, net = net, epoch=5, batch=256)
    clients = manager.clients
    ids = [i for i in range(len(clients))]

    num_select = 30
    max_workers = 3
    global_epoch = 30
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
        global_eval(net, True)

    tock = time.time()
    print(tock - tick)

    


    print('done')




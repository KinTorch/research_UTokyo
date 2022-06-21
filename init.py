import torch
from client import Client
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import copy


class client_dataset(Dataset):
    def __init__(self, dataset, data_ids):
        self.dataset = dataset
        self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, k):
        dataset = self.dataset
        data_ids = self.data_ids
        x = dataset[data_ids[k]][0]
        y = dataset[data_ids[k]][1]

        return x, y


class client_manager():

    def __init__(self, num_client: int, iid: bool, net, epoch, batch):
        self.net = net
        self.num_epoch = epoch
        self.num_batch = batch

        self.dataset_train = datasets.CIFAR10(
            './datasets/cifar10', train=True, download=False, transform=transforms.ToTensor())
        self.dataset_test = datasets.CIFAR10(
            './datasets/cifar10', train=False, download=False, transform=transforms.ToTensor())

        self.clients = self.init_clients(num_client, iid)

    def init_clients(self, num_client, iid):
        clients = []
        for i in range(num_client):
            id = i
            net_c = copy.deepcopy(self.net)
            init_args = dict(net=net_c, epoch=self.num_epoch,
                             batch=self.num_batch)
            c = Client(id, init_args)
            clients.append(c)

        self.init_detaset(clients, iid)
        return clients

    def iid_group(self, dataset):
        data_ids = [i for i in range(len(dataset))]
        random.shuffle(data_ids)
        return data_ids

    def noniid_group(self, dataset):
        labels = [y for _, y in dataset]
        data_ids = [i for i in range(len(dataset))]
        pairs = [(label, id) for label, id in zip(labels, data_ids)]
        pairs.sort()
        data_ids = [id for _, id in pairs]
        return data_ids

    def init_client_dataset(self, dataset, data_ids):
        t = len(data_ids) // 10
        train_dataset = client_dataset(dataset, data_ids[0:-t])
        test_dataset = client_dataset(dataset, data_ids[-t:])
        return train_dataset, test_dataset

    def init_detaset(self, clients: Client, iid: bool):
        dataset_train = self.dataset_train
        dataset_test = self.dataset_test

        if iid:
            data_ids = self.iid_group(dataset_train)
        else:
            data_ids = self.noniid_group(dataset_train)

        for i, c in enumerate(clients):
            t = len(dataset_train) // len(clients)
            train_dataset, test_dataset = self.init_client_dataset(
                dataset_train, data_ids[i*t:(i+1)*t])
                
            c.load_data(train_dataset, test_dataset)

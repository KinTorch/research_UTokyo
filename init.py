import logging
import string
import torch
from client import Client
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import copy
import numpy as np


class client_dataset(Dataset):
    def __init__(self, dataset, data_ids):
        self.dataset = dataset
        self.data_ids = data_ids
        self.len = len(self.data_ids)

    def __len__(self):
        return self.len

    def __getitem__(self, k):
        if k >= self.len:
            raise StopIteration

        dataset = self.dataset
        data_ids = self.data_ids
        x = dataset[data_ids[k]][0]
        y = dataset[data_ids[k]][1]

        return x, y


class shuffled_dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_ids = [i for i in range(len(dataset))]
        random.shuffle(self.data_ids)
        self.len = len(self.dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, k):
        if k >= self.len:
            raise StopIteration

        x = self.dataset[self.data_ids[k]][0]
        y = self.dataset[self.data_ids[k]][1]

        return x, y


class mixed_dataset(Dataset):
    def __init__(self, *datasets):
        self.lens = np.zeros((len(datasets)), dtype=int)

        for i in range(len(datasets)):
            self.lens[i] = len(datasets[i])

        self.min_len = np.min(self.lens)

        self.datasets = [shuffled_dataset(dataset) for dataset in datasets]

        self.len = self.min_len * len(datasets)

    def __len__(self):
        return self.len

    def __getitem__(self, k):
        if k >= self.len:
            raise StopIteration

        for data in self.datasets:
            if k >= self.min_len:
                k -= self.min_len
            else:
                x = data[k][0]
                y = data[k][1]
                break

        return x, y


class client_manager():

    def __init__(self, num_client: int, net, epoch, batch, logger):
        self.net = net
        self.num_epoch = epoch
        self.num_batch = batch

        self.clients = self.init_clients(num_client, logger)

    def init_clients(self, num_client, logger):
        clients = []
        for i in range(num_client):
            id = i
            net_c = copy.deepcopy(self.net)
            init_args = dict(net=net_c, epoch=self.num_epoch,
                             batch=self.num_batch, logger=logger)
            c = Client(id, init_args)
            clients.append(c)

        return clients

    def iid_group(self, dataset):
        data_ids = [i for i in range(len(dataset))]
        random.shuffle(data_ids)
        return data_ids

    def noniid_group(self, dataset):
        t = dataset[0]
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

    # size: -1 avg, >0 data size per client
    def load_dataset(self, data_list, ids, iid: bool, size=-1):

        data_mixed = []

        if 'cifar10' in data_list:
            data_mixed.append(datasets.CIFAR10(
                './datasets/cifar10', train=True, download=False, transform=transforms.ToTensor()))

        if 'svhn' in data_list:
            data_mixed.append(datasets.SVHN(
                './datasets/svhn', split='train', transform=transforms.ToTensor(), download=False))

        if 'mnist' in data_list:
            transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(
                num_output_channels=3), transforms.ToTensor()])
            data_mixed.append(datasets.MNIST(
                './datasets/mnist', train=True, transform=transform, download=False))

        else:
            raise 'No dataset'

        dataset_train = mixed_dataset(*data_mixed)

        if iid:
            data_ids = self.iid_group(dataset_train)
        else:
            data_ids = self.noniid_group(dataset_train)

        if size == -1:
            t = len(dataset_train) // len(ids)
        else:
            assert size > 0
            t = size

        for i, id in enumerate(ids):

            train_dataset, test_dataset = self.init_client_dataset(
                dataset_train, data_ids[i*t:(i+1)*t])

            self.clients[id].load_data(train_dataset, test_dataset)



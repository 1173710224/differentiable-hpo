from const import BATCHSIZE
from matplotlib.pyplot import axis
from pandas.core.arrays import sparse
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torch._C import device, dtype
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt
from bayes_opt import BayesianOptimization
import pandas as pd
import torch


def bayes():
    return


def zoopt(dataset):
    dim = Dimension(
        19,
        [[16, 32], [1, 8], [1, 1], [1, 1], [16, 32],
         [1, 8], [1, 1], [1, 1], [0, 1], [1, 8],
         [1, 10], [0, 1], [1, 8], [1, 10], [40, 50],
         [30, 40], [20, 30], [10, 20], [0.0001, 0.001]],
        [False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, True]
    )
    obj = Objective(eval, dim)
    value = 0.95 if type == 'mnist' else 0.8
    solution = Opt.min(obj, Parameter(budget=20, terminal_value=value))
    return solution.get_x()


class Data():
    '''
    we load huge datasets as two torch-loader: train-loader and test-loader, so as to train them in multi-batches.
    While load other small or big datasets as
    '''

    def __init__(self) -> None:
        self.datasets = ["iris", "wine", "car", "agaricus_lepiota", "dota2"]
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        pass

    def read_partition_size(self):
        for dataset in self.datasets:
            print("{}:".format(dataset))
            train, test = eval(
                "self.load_{}()".format(dataset))
            print("train:{},test:{}".format(len(train[0]), len(test[0])))
        return

    def load_mnist(self):
        data_root_path = "data/"
        train_dataset = datasets.MNIST(root=data_root_path, train=True,
                                       transform=transforms.ToTensor(), download=True)
        # if torch.cuda.is_available():
        #     train_dataset.data = train_dataset.data.cuda()
        # else:
        #     train_dataset.data = train_dataset.data.cpu()
        # # print(train_dataset.data.device)
        test_dataset = datasets.MNIST(root=data_root_path, train=False,
                                      transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4,
                                  #   pin_memory=True
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 #  num_workers=4,
                                 #  pin_memory=True
                                 )
        return train_loader, test_loader

    def load_svhn(self):
        data_root_path = "data/SVHN/"
        train_dataset = datasets.SVHN(root=data_root_path, split="train",
                                      transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.SVHN(root=data_root_path, split="test",
                                     transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True)

        return train_loader, test_loader

    def load_iris(self):
        LabelIndex = 4
        path = "data/iris/iris.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, :-1],
                                  sp.LabelEncoder().fit_transform(
            df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)

    def load_wine(self):
        LabelIndex = 0
        path = "data/wine/wine.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, 1:],
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)

    def load_car(self):
        LabelIndex = 6
        path = "data/car/car.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, :-1]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)

    def load_agaricus_lepiota(self):
        LabelIndex = 0
        path = "data/agaricus-lepiota/agaricus-lepiota.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, 1:11]),
                                   sp.OneHotEncoder(sparse=False).fit_transform(
                                       df.values[:, 12:]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)

    def load_dota2(self):
        LabelIndex = 0
        train_path = "data/dota2/dota2Train.csv"
        test_path = "data/dota2/dota2Test.csv"
        df_train = pd.read_csv(train_path, header=None, dtype=float)
        df_test = pd.read_csv(test_path, header=None, dtype=float)
        data_train = torch.Tensor(df_train.values)
        if torch.cuda.is_available():
            data_train = data_train.cuda()
        data_test = torch.Tensor(df_test.values)
        if torch.cuda.is_available():
            data_test = data_test.cuda()
        x_train, y_train = data_train[:, 1:], data_train[:, LabelIndex].reshape(
            len(data_train))
        x_test, y_test = data_test[:, 1:], data_test[:,
                                                     LabelIndex].reshape(len(data_test))
        return (x_train, y_train), (x_test, y_test)


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            imgs, label = self.batch
            imgs = imgs.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            self.batch = [imgs, label]
            # for k in self.batch:
            #     if k != 'meta':
            #         self.batch[k] = self.batch[k].to(
            #             device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

if __name__ == "__main__":
    data = Data()
    (_, y_train), (_, y_test) = data.load_dota2()
    print(set(y_train))
    pass

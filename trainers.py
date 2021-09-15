from models import ConvMNIST, ConvSVHN, DhpoConvMNIST, DhpoConvSVHN, Dnn, DhpoDnn
from utils import Data, DataPrefetcher
import torch
from const import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, dataset, h_params, epoch=-1) -> None:
        self.epoch = epoch
        if self.epoch == -1:
            self.epoch = EPOCHS
        self.data = Data()
        self.h_params = h_params
        if dataset == MNIST:
            self.train_loader, self.test_loader = self.data.load_mnist()
            input_channel = 1
            ndim = 28
            nclass = 10
            self.model = ConvMNIST(input_channel=input_channel,
                                   ndim=ndim, nclass=nclass, h_params=h_params)
        elif dataset == SVHN:
            self.train_loader, self.test_loader = self.data.load_svhn()
            input_channel = 3
            ndim = 32
            nclass = 10
            self.model = ConvSVHN(input_channel=input_channel,
                                  ndim=ndim, nclass=nclass, h_params=h_params)

        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.reset_parameters()
        self.optimizier = torch.optim.Adam(
            self.model.parameters(), lr=0.001)
        self.loss_sequence = []
        self.model.train()
        for i in range(self.epoch):
            loss_sum = 0
            img_num = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
                img_num += len(imgs)
            avg_loss = loss_sum * 1.0/img_num
            self.loss_sequence.append(avg_loss)
            print("Epoch~{}->{}".format(i+1, avg_loss))
        return

    def val(self):
        ncorrect = 0
        nsample = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            # print(imgs.device, label.device)
            self.model.eval()
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return ncorrect/nsample

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


class DhpoTrainer():
    def __init__(self, dataset) -> None:
        self.data = Data()
        if dataset == MNIST:
            self.train_loader, self.test_loader = self.data.load_mnist()
            input_channel = 1
            ndim = 28
            nclass = 10
            self.model = DhpoConvMNIST(input_channel=input_channel,
                                       ndim=ndim, nclass=nclass)
        elif dataset == SVHN:
            self.train_loader, self.test_loader = self.data.load_svhn()
            input_channel = 3
            ndim = 32
            nclass = 10
            self.model = DhpoConvSVHN(input_channel=input_channel,
                                      ndim=ndim, nclass=nclass)
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.reset_parameters()
        self.optimizier = torch.optim.Adam(
            self.model.parameters(), lr=0.001)
        self.loss_sequence = []
        self.model.train()
        for i in range(EPOCHS):
            loss_sum = 0
            img_num = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
                img_num += len(imgs)
            self.loss_sequence.append(loss_sum * 1.0 / img_num)
            print("Epoch~{}->{}".format(i+1, loss_sum * 1.0 / img_num))
        return

    def val(self):
        ncorrect = 0
        nsample = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            self.model.eval()
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return ncorrect/nsample

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


class TrainerDense():
    def __init__(self, dataset, h_params, it_n=-1) -> None:
        self.it_n = it_n
        if self.it_n == -1:
            self.it_n = EPOCHSDENSE
        self.data = Data()
        self.h_params = h_params
        if dataset == IRIS:
            self.train_data, self.test_data = self.data.load_iris()
            ndim = 4
            nclass = 3
            self.model = Dnn(ndim=ndim, nclass=nclass, h_params=h_params)
        if dataset == WINE:
            self.train_data, self.test_data = self.data.load_wine()
            ndim = 13
            nclass = 3
            self.model = Dnn(ndim=ndim, nclass=nclass, h_params=h_params)
        if dataset == CAR:
            self.train_data, self.test_data = self.data.load_car()
            ndim = 21
            nclass = 4
            self.model = Dnn(ndim=ndim, nclass=nclass, h_params=h_params)
        if dataset == AGARICUS:
            self.train_data, self.test_data = self.data.load_agaricus_lepiota()
            ndim = 112
            nclass = 2
            self.model = Dnn(ndim=ndim, nclass=nclass, h_params=h_params)
        if dataset == DOTA2:
            self.train_data, self.test_data = self.data.load_dota2()
            ndim = 116
            nclass = 2
            self.model = Dnn(ndim=ndim, nclass=nclass, h_params=h_params)

        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.reset_parameters()
        self.optimizier = torch.optim.Adam(
            self.model.parameters(), lr=0.001)
        self.loss_sequence = []
        self.model.train()
        for i in range(self.it_n):
            x, y = self.train_data
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            self.optimizier.zero_grad()
            loss.backward()
            self.optimizier.step()
            if i % 20 == 0:
                print("Epoch~{}->{}".format(i+1, loss.item()))
            self.loss_sequence.append(loss.item())
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


class DhpoTrainerDense():
    def __init__(self, dataset) -> None:
        self.data = Data()
        if dataset == IRIS:
            self.train_data, self.test_data = self.data.load_iris()
            ndim = 4
            nclass = 3
            self.model = DhpoDnn(ndim=ndim, nclass=nclass)
        if dataset == WINE:
            self.train_data, self.test_data = self.data.load_wine()
            ndim = 13
            nclass = 3
            self.model = DhpoDnn(ndim=ndim, nclass=nclass)
        if dataset == CAR:
            self.train_data, self.test_data = self.data.load_car()
            ndim = 21
            nclass = 4
            self.model = DhpoDnn(ndim=ndim, nclass=nclass)
        if dataset == AGARICUS:
            self.train_data, self.test_data = self.data.load_agaricus_lepiota()
            ndim = 112
            nclass = 2
            self.model = DhpoDnn(ndim=ndim, nclass=nclass)
        if dataset == DOTA2:
            self.train_data, self.test_data = self.data.load_dota2()
            ndim = 116
            nclass = 2
            self.model = DhpoDnn(ndim=ndim, nclass=nclass)

        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.reset_parameters()
        self.model.train()
        self.optimizier = torch.optim.Adam(
            self.model.parameters(), lr=0.001)
        self.loss_sequence = []
        for i in range(EPOCHSDENSE):
            x, y = self.train_data
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            self.optimizier.zero_grad()
            loss.backward()
            self.optimizier.step()
            if i % 20 == 0:
                print("Epoch~{}->{}".format(i+1, loss.item()))
            self.loss_sequence.append(loss.item())
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


if __name__ == "__main__":
    # trainer = Trainer(SVHN, TESTPARAM)
    # trainer = DhpoTrainer(MNIST)
    trainer = DhpoTrainerDense(WINE)
    trainer.train()
    accu = trainer.val()
    print(accu)
    # trainer = TrainerDense(IRIS, TESTPARAMDENSE)
    # trainer.train()
    # accu = trainer.val()
    # print(accu)

    pass

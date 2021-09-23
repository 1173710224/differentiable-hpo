from models import ConvMNIST, ConvSVHN, DhpoConvMNIST, DhpoConvSVHN, Dnn, DhpoDnn
from utils import Data, DataPrefetcher
import torch
from const import *
import torch.nn.functional as F
from torch.nn import Linear
import warnings
from random import randint
import ctypes
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, dataset, h_params) -> None:
        self.data = Data()
        self.h_params = h_params
        self.lr = h_params[LR]
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
        self.optimizier = torch.optim.SGD(
            self.model.parameters(), lr=h_params[LR])
        # self.optimizier = torch.optim.Adam(
        #     self.model.parameters(), lr=h_params[LR])
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        # self.optimizier = torch.optim.SGD(
        #     self.model.parameters(), lr=self.h_params[LR])
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
                # print(loss.item())
            avg_loss = loss_sum * 1.0/img_num
            print("Epoch~{}->{}".format(i+1, avg_loss))
            # break
        return

    def lr_train(self):
        self.model.train()
        self.last_param = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(param.size()))
        loss_sum = 0
        for i in range(EPOCHS):
            for img, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                print("current learning rate:{}".format(self.lr))
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.lr)
                preds = self.model(img)
                loss = F.cross_entropy(preds, label)
                optimizer.zero_grad()
                loss.backward()
                self.lr_autograd()
                self.lr_step()
                optimizer.step()
                print("loss:{}".format(loss.item()))
        return

    def lr_autograd(self):
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad)
        grad_sum = 0
        for i in range(len(self.last_param)):
            grad_sum += torch.sum(
                torch.mul(self.last_param[i], self.tmp_param[i]))
        self.lr_grad = -float(grad_sum)
        self.last_param = self.tmp_param
        print("learning rate's grad: {}".format(self.lr_grad))
        return

    def lr_step(self):
        self.lr -= METALR * self.lr_grad
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


class TrainerDense():
    def __init__(self, dataset, h_params) -> None:
        self.data = Data()
        self.lr = h_params[LR]
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

        # self.optimizier = torch.optim.Adam(
        #     self.model.parameters(), lr=h_params[LR])
        self.optimizier = torch.optim.SGD(
            self.model.parameters(), lr=h_params[LR])
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.loss_sequence = []
        self.model.train()
        for i in range(200):
            x, y = self.train_data
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            self.optimizier.zero_grad()
            loss.backward()
            self.optimizier.step()
            # for param in self.model.parameters():
            #     # print(param[0])
            #     print(param.grad)
            #     # break
            if i % 100 == 0:
                print("Epoch~{}->{}".format(i+1, loss.item()))
            self.loss_sequence.append(loss.item())
            # print()
        return

    def lr_train(self):
        self.model.train()
        self.last_param = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(
                param.size(), device=self.device))
        for i in range(200):
            x, y = self.train_data
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            loss.backward()
            self.lr_autograd()
            self.lr_step()
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print("loss:{}".format(loss.item()))
                print("current learning rate:{}, grad={}\n".format(
                    self.lr, self.lr_grad))
        return

    def lr_autograd(self):
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad)
        grad_sum = 0
        for i in range(len(self.last_param)):
            grad_sum += torch.sum(
                torch.mul(self.last_param[i], self.tmp_param[i]))
        self.lr_grad = -float(grad_sum)
        self.last_param = self.tmp_param
        return

    def lr_step(self):
        self.lr -= METALR * self.lr_grad
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        print(preds)
        print(y)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.model.reset_parameters()
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


class LrTrainer():
    def __init__(self, train_data, test_data, model, lr=0.5) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.train()
        # optimizer = torch.optim.Adagrad(
        #     self.model.parameters(), lr=0.01)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01)
        print("init loss:{}\n".format(F.mse_loss(self.model(self.x), self.y)))
        for i in range(EPOCHSDENSE):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            # for param in self.model.parameters():
            #     print("{},param_grad:{}".format(
            #         param, param.grad))
            print("Epoch~{}->loss:{}\n".format(i + 1, loss.item()))
            optimizer.step()
        return

    def lr_train(self):
        self.model.train()
        self.last_param = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(
                param.size(), device=self.device))
        for i in range(EPOCHSDENSE * 100):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            loss.backward()
            self.lr_autograd()
            self.lr_step()
            # for param in self.model.parameters():
            #     print("{},param_grad:{}".format(
            #         param, param.grad))
            print("Epoch~{}->lr:{},lr_grad:{},loss:{}\n".format(i + 1,
                                                                self.lr, self.lr_grad, loss.item()))
            # optimizer = torch.optim.Adam(
            #     self.model.parameters(), lr=self.lr)
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
            optimizer.step()
            optimizer.zero_grad()
            # if i % 100 == 0:
            #     print("loss:{}".format(loss.item()))
            #     print("current learning rate:{}, grad={}\n".format(
            #         self.lr, self.lr_grad))
        return

    def lr_autograd(self):
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad.clone())
        grad_sum = 0
        for i in range(len(self.last_param)):
            # print("last_grad:{},tmp_grad:{},hadamaji:{}".format(
            #     self.last_param[i], self.tmp_param[i], torch.mul(self.last_param[i], self.tmp_param[i])))
            grad_sum += torch.sum(
                torch.mul(self.last_param[i], self.tmp_param[i]))
        self.lr_grad = -float(grad_sum)
        # print("grad:{}\n".format(self.lr_grad))
        self.last_param = self.tmp_param
        return

    def lr_step(self):
        self.lr -= METALR * self.lr_grad
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu


class LrMatrixTrainer():
    def __init__(self, train_data, test_data, model, lr=0.5) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def lr_train(self):
        self.model.train()
        self.last_param = []
        self.lr_matrix = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(
                param.size(), device=self.device))
            self.lr_matrix.append(torch.ones(
                param.size(), device=self.device)/100)
        print("init loss:{}\n".format(F.mse_loss(self.model(self.x), self.y)))
        for i in range(EPOCHSDENSE):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            loss.backward()
            self.lr_autograd()
            self.lr_step()
            self.w_step()
            self.zero_grad()
            # print("Epoch~{}->\nlr_matrix:{},\nlr_grad:{},\nloss:{}\n".format(i + 1,
            #                                                                  self.lr_matrix, self.lr_grad, loss.item()))
            print("Epoch~{}->loss:{}\n".format(i + 1, loss.item()))
        return

    def lr_autograd(self):
        self.lr_grad = []
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad.clone())
        for i in range(len(self.last_param)):
            self.lr_grad.append(-torch.mul(
                self.last_param[i], self.tmp_param[i]))
        self.last_param = self.tmp_param
        return

    def lr_step(self):
        for i in range(len(self.lr_matrix)):
            self.lr_matrix[i] = self.lr_matrix[i] - METALR * self.lr_grad[i]
        return

    def w_step(self):
        # for param in self.model.parameters():
        #     print("param[0]:{}".format(param[0]))
        #     break
        i = 0
        for param in self.model.parameters():
            param.data -= torch.mul(param.grad, self.lr_matrix[i])
            i += 1
        # for param in self.model.parameters():
        #     print("param[0]:{}".format(param[0]))
        #     break
        return

    def zero_grad(self):
        self.model.zero_grad()
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu


if __name__ == "__main__":
    # trainer = TrainerDense(IRIS, TESTPARAMDENSE)
    # trainer.lr_train()
    # trainer.train()
    # accu = trainer.val()
    # print(accu)
    # model = Dnn(4, 3, TESTPARAMDENSE)

    # model = Linear(4, 1)
    # x = torch.rand((100, 4))
    # y = torch.sum(x, dim=1)
    # trainer = LrMatrixTrainer((x, y), (x, y), model)
    # trainer.lr_train()
    # model.reset_parameters()
    # trainer = LrTrainer((x, y), (x, y), model)
    # trainer.train()

    # data = Data()
    # train_data, test_data = data.load_iris()
    # ndim = 4
    # nclass = 3
    # model = Dnn(ndim=ndim, nclass=nclass, h_params=TESTPARAMDENSE)
    # trainer = LrMatrixTrainer(train_data, test_data, model)
    # trainer.lr_train()
    # trainer.train()
    if torch.cuda.is_available():
        print("1")
    else:
        print(0)
    pass

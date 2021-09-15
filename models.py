from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, PReLU
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
from utils import Data
import torch
import math


class ConvSVHN(nn.Module):
    def __init__(self, input_channel, ndim, nclass, h_params):
        super(ConvSVHN, self).__init__()

        self.ndim = ndim
        self.nclass = nclass
        self.h_params = h_params

        self.conv1_1 = Conv2d(in_channels=input_channel,
                              out_channels=h_params[CONV11CHANNEL], kernel_size=h_params[CONV11KERNEL], bias=False)
        self.conv1_2 = Conv2d(in_channels=h_params[CONV11CHANNEL],
                              out_channels=h_params[CONV12CHANNEL], kernel_size=h_params[CONV12KERNEL], bias=False)
        self.conv1_3 = Conv2d(in_channels=h_params[CONV12CHANNEL],
                              out_channels=h_params[CONV13CHANNEL], kernel_size=h_params[CONV13KERNEL], bias=False)
        if h_params[POOL1TYPE] == MAXPOOL:
            self.pool1 = MaxPool2d(kernel_size=h_params[POOL1KERNEL], stride=1)
        elif h_params[POOL1TYPE] == AVGPOOL:
            self.pool1 = AvgPool2d(kernel_size=h_params[POOL1KERNEL], stride=1)
        self.conv2_1 = Conv2d(in_channels=h_params[CONV13CHANNEL],
                              out_channels=h_params[CONV21CHANNEL], kernel_size=h_params[CONV21KERNEL], bias=False)
        self.conv2_2 = Conv2d(in_channels=h_params[CONV21CHANNEL],
                              out_channels=h_params[CONV22CHANNEL], kernel_size=h_params[CONV22KERNEL], bias=False)
        self.conv2_3 = Conv2d(in_channels=h_params[CONV22CHANNEL],
                              out_channels=h_params[CONV23CHANNEL], kernel_size=h_params[CONV23KERNEL], bias=False)
        if h_params[POOL2TYPE] == MAXPOOL:
            self.pool2 = MaxPool2d(kernel_size=h_params[POOL2KERNEL], stride=1)
        elif h_params[POOL2TYPE] == AVGPOOL:
            self.pool2 = AvgPool2d(kernel_size=h_params[POOL2KERNEL], stride=1)

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=self.get_flatten_dim(), out_features=1024)
        self.dense3 = Linear(in_features=1024, out_features=256)
        self.dense5 = Linear(in_features=256, out_features=nclass)
        return

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = F.sigmoid(self.dense1(x))
        x = F.sigmoid(self.dense3(x))
        res = self.dense5(x)
        return res

    def get_flatten_dim(self):
        ndim = self.ndim
        h_params = self.h_params
        decrease = sum(
            [
                h_params[CONV11KERNEL],
                h_params[CONV12KERNEL],
                h_params[CONV13KERNEL],
                h_params[POOL1KERNEL],
                h_params[CONV21KERNEL],
                h_params[CONV22KERNEL],
                h_params[CONV23KERNEL],
                h_params[POOL2KERNEL]
            ]) - 8
        return (ndim - decrease) ** 2 * h_params[CONV23CHANNEL]

    def get_pooling_dim(self):
        ndim = self.ndim
        h_params = self.h_params
        decrease = sum(
            [h_params[CONV11KERNEL], h_params[CONV12KERNEL],
             h_params[CONV13KERNEL], h_params[POOL1KERNEL],
             h_params[CONV21KERNEL], h_params[CONV22KERNEL],
             h_params[CONV23KERNEL], h_params[POOL2KERNEL]]) - 8
        return ndim - decrease

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()
        self.dense1.reset_parameters()
        self.dense3.reset_parameters()
        self.dense5.reset_parameters()
        return


class ConvMNIST(nn.Module):
    def __init__(self, input_channel, ndim, nclass, h_params):
        super(ConvMNIST, self).__init__()

        self.ndim = ndim
        self.nclass = nclass
        self.h_params = h_params

        self.conv1_1 = Conv2d(in_channels=input_channel,
                              out_channels=h_params[CONV11CHANNEL], kernel_size=h_params[CONV11KERNEL], bias=False)
        self.conv1_2 = Conv2d(in_channels=h_params[CONV11CHANNEL],
                              out_channels=h_params[CONV12CHANNEL], kernel_size=h_params[CONV12KERNEL], bias=False)
        self.conv1_3 = Conv2d(in_channels=h_params[CONV12CHANNEL],
                              out_channels=h_params[CONV13CHANNEL], kernel_size=h_params[CONV13KERNEL], bias=False)
        if h_params[POOL1TYPE] == MAXPOOL:
            self.pool1 = MaxPool2d(kernel_size=h_params[POOL1KERNEL], stride=1)
        elif h_params[POOL1TYPE] == AVGPOOL:
            self.pool1 = AvgPool2d(kernel_size=h_params[POOL1KERNEL], stride=1)
        self.conv2_1 = Conv2d(in_channels=h_params[CONV13CHANNEL],
                              out_channels=h_params[CONV21CHANNEL], kernel_size=h_params[CONV21KERNEL], bias=False)
        self.conv2_2 = Conv2d(in_channels=h_params[CONV21CHANNEL],
                              out_channels=h_params[CONV22CHANNEL], kernel_size=h_params[CONV22KERNEL], bias=False)
        self.conv2_3 = Conv2d(in_channels=h_params[CONV22CHANNEL],
                              out_channels=h_params[CONV23CHANNEL], kernel_size=h_params[CONV23KERNEL], bias=False)
        if h_params[POOL2TYPE] == MAXPOOL:
            self.pool2 = MaxPool2d(kernel_size=h_params[POOL2KERNEL], stride=1)
        elif h_params[POOL2TYPE] == AVGPOOL:
            self.pool2 = AvgPool2d(kernel_size=h_params[POOL2KERNEL], stride=1)

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=self.get_flatten_dim(), out_features=64)
        self.dense3 = Linear(in_features=64, out_features=32)
        self.dense5 = Linear(in_features=32, out_features=nclass)
        return

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.sigmoid(self.dense1(x))
        x = F.sigmoid(self.dense3(x))
        res = self.dense5(x)
        return res

    def get_flatten_dim(self):
        ndim = self.ndim
        h_params = self.h_params
        decrease = sum(
            [
                h_params[CONV11KERNEL],
                h_params[CONV12KERNEL],
                h_params[CONV13KERNEL],
                h_params[POOL1KERNEL],
                h_params[CONV21KERNEL],
                h_params[CONV22KERNEL],
                h_params[CONV23KERNEL],
                h_params[POOL2KERNEL]
            ]) - 8
        return (ndim - decrease) ** 2 * h_params[CONV23CHANNEL]

    def get_pooling_dim(self):
        ndim = self.ndim
        h_params = self.h_params
        decrease = sum(
            [h_params[CONV11KERNEL], h_params[CONV12KERNEL],
             h_params[CONV13KERNEL], h_params[POOL1KERNEL],
             h_params[CONV21KERNEL], h_params[CONV22KERNEL],
             h_params[CONV23KERNEL], h_params[POOL2KERNEL]]) - 8
        return ndim - decrease

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()
        self.dense1.reset_parameters()
        self.dense3.reset_parameters()
        self.dense5.reset_parameters()
        return


class DhpoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=CHANNELTOP, kernel_size=KERNELTOP):
        super(DhpoConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.opt_channel_size = None
        self.opt_kernel_size = None
        self.opt_channel_size_line = None
        self.opt_kernel_size_line = None
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.channel_base = Parameter(torch.Tensor(1, out_channels))
        self.channel_controller = Parameter(torch.Tensor(1))
        self.kernel_base = Parameter(
            torch.Tensor(1, kernel_size))
        self.kernel_controller = Parameter(torch.Tensor(1))

        self.up_traingle_for_channel = Parameter(torch.zeros(
            (self.out_channels, self.out_channels)), requires_grad=False)
        for i in range(self.out_channels):
            for j in range(self.out_channels):
                if i >= j:
                    self.up_traingle_for_channel[i, j] = 1

        self.up_traingle_for_kernel = Parameter(torch.zeros(
            (self.kernel_size, self.kernel_size)), requires_grad=False)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i >= j:
                    self.up_traingle_for_kernel[i, j] = 1

        self.reset_parameters()
        pass

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.channel_base, -1, 1)
        init.uniform_(self.kernel_base, -1, 1)
        init.uniform_(self.channel_controller, -1, 1)
        init.uniform_(self.kernel_controller, -1, 1)
        return

    def format_channel_size(self):
        EXPANDTIMES = 10
        channel_p = F.softmax(self.channel_base, dim=1)
        channel_score = torch.mm(channel_p, self.up_traingle_for_channel)
        channel_score_zeroed = channel_score - \
            torch.sigmoid(self.channel_controller)
        channel_score_expand = channel_score_zeroed * EXPANDTIMES
        channel_size = torch.sigmoid(channel_score_expand)
        self.opt_channel_size = round(float(torch.sum(channel_size)))
        self.opt_channel_size_line = channel_size
        return channel_size

    def format_kernel_size(self):
        EXPANDTIMES = 10
        kernel_p = F.softmax(self.kernel_base, dim=1)
        kernel_score = torch.mm(kernel_p, self.up_traingle_for_kernel)
        kernel_score_zeroed = kernel_score - \
            torch.sigmoid(self.kernel_controller)
        kernel_score_expand = kernel_score_zeroed * EXPANDTIMES
        kernel_size_line = torch.sigmoid(kernel_score_expand)
        kernel_size_matrix = torch.zeros((self.kernel_size, self.kernel_size))
        if torch.cuda.is_available():
            kernel_size_matrix = kernel_size_matrix.cuda()
        for i in range(self.kernel_size):
            kernel_size_matrix[:self.kernel_size-i, :self.kernel_size -
                               i] = kernel_size_line[0][self.kernel_size-i-1]
        self.opt_kernel_size = round(float(torch.sum(kernel_size_line)))
        self.opt_kernel_size_line = kernel_size_line
        return kernel_size_matrix

    def forward(self, x):
        kernel_size_matrix = self.format_kernel_size()
        x = F.conv2d(x, torch.mul(self.weight, kernel_size_matrix), padding=2)
        x = torch.mul(x, self.format_channel_size().unsqueeze(2).unsqueeze(3))
        return x

    def get_hparams(self):
        if self.opt_channel_size == None or self.opt_kernel_size == None:
            return None
        return self.opt_channel_size, self.opt_kernel_size

    def get_hparams_tensor(self):
        return self.opt_channel_size_line, self.opt_kernel_size_line


class DhpoPool2d(nn.Module):
    def __init__(self, kernel_size=KERNELTOP):
        super(DhpoPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.selecter_base = Parameter(torch.Tensor(8, 1, 1, 1, 1))
        self.reset_parameters()
        pass

    def reset_parameters(self):
        init.uniform_(self.selecter_base, -1, 1)
        return

    def forward(self, x):
        expanded_pool = torch.stack((
            F.max_pool2d(input=x, kernel_size=2, stride=1,
                         padding=1)[:, :, :-1, :-1],

            F.max_pool2d(input=x, kernel_size=3, stride=1,
                         padding=1),

            F.max_pool2d(input=x, kernel_size=4, stride=1,
                         padding=2)[:, :, :-1, :-1],

            F.max_pool2d(input=x, kernel_size=5, stride=1,
                         padding=2),
            F.avg_pool2d(input=x, kernel_size=2, stride=1,
                         padding=1)[:, :, :-1, :-1],

            F.avg_pool2d(input=x, kernel_size=3, stride=1,
                         padding=1),

            F.avg_pool2d(input=x, kernel_size=4, stride=1,
                         padding=2)[:, :, :-1, :-1],

            F.avg_pool2d(input=x, kernel_size=5, stride=1,
                         padding=2),
        ))
        selecter = F.softmax(self.selecter_base, dim=0)
        weighted_pool = torch.mul(expanded_pool, selecter)
        res = torch.sum(weighted_pool, dim=0)
        return res

    def get_hparams(self):
        index = int(self.selecter_base.argmax(dim=0))
        if index <= 3:
            return MAXPOOL, index + 2
        else:
            return AVGPOOL, index - 2

    def get_hparams_tensor(self):
        return F.softmax(self.selecter_base, dim=0)


class DhpoConvSVHN(nn.Module):
    def __init__(self, input_channel, ndim, nclass):
        super(DhpoConvSVHN, self).__init__()
        self.ndim = ndim
        self.nclass = nclass

        self.conv1_1 = DhpoConv2d(in_channels=input_channel)
        self.conv1_2 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv1_3 = DhpoConv2d(in_channels=CHANNELTOP)
        self.pool1 = DhpoPool2d()
        self.conv2_1 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv2_2 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv2_3 = DhpoConv2d(in_channels=CHANNELTOP)
        self.pool2 = DhpoPool2d()

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=CHANNELTOP * self.ndim * self.ndim, out_features=1024)
        self.dense3 = Linear(in_features=1024, out_features=256)
        self.dense5 = Linear(in_features=256, out_features=nclass)
        pass

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = F.sigmoid(self.dense1(x))
        x = F.sigmoid(self.dense3(x))
        res = self.dense5(x)
        return res

    def get_hparams(self):
        return {
            CONV11CHANNEL: self.conv1_1.get_hparams()[0],
            CONV11KERNEL: self.conv1_1.get_hparams()[1],
            CONV12CHANNEL: self.conv1_2.get_hparams()[0],
            CONV12KERNEL: self.conv1_2.get_hparams()[1],
            CONV13CHANNEL: self.conv1_3.get_hparams()[0],
            CONV13KERNEL: self.conv1_3.get_hparams()[1],
            POOL1TYPE: self.pool1.get_hparams()[0],
            POOL1KERNEL: self.pool1.get_hparams()[1],
            CONV21CHANNEL: self.conv2_1.get_hparams()[0],
            CONV21KERNEL: self.conv2_1.get_hparams()[1],
            CONV22CHANNEL: self.conv2_2.get_hparams()[0],
            CONV22KERNEL: self.conv2_2.get_hparams()[1],
            CONV23CHANNEL: self.conv2_3.get_hparams()[0],
            CONV23KERNEL: self.conv2_3.get_hparams()[1],
            POOL2TYPE: self.pool2.get_hparams()[0],
            POOL2KERNEL: self.pool2.get_hparams()[1],
            LR: 0.001,
        }

    def get_hparams_tensor(self):
        return {
            CONV11CHANNEL: self.conv1_1.get_hparams_tensor()[0],
            CONV11KERNEL: self.conv1_1.get_hparams_tensor()[1],
            CONV12CHANNEL: self.conv1_2.get_hparams_tensor()[0],
            CONV12KERNEL: self.conv1_2.get_hparams_tensor()[1],
            CONV13CHANNEL: self.conv1_3.get_hparams_tensor()[0],
            CONV13KERNEL: self.conv1_3.get_hparams_tensor()[1],
            "pool1": self.pool1.get_hparams_tensor().reshape((1, 8)),
            CONV21CHANNEL: self.conv2_1.get_hparams_tensor()[0],
            CONV21KERNEL: self.conv2_1.get_hparams_tensor()[1],
            CONV22CHANNEL: self.conv2_2.get_hparams_tensor()[0],
            CONV22KERNEL: self.conv2_2.get_hparams_tensor()[1],
            CONV23CHANNEL: self.conv2_3.get_hparams_tensor()[0],
            CONV23KERNEL: self.conv2_3.get_hparams_tensor()[1],
            "pool2": self.pool2.get_hparams_tensor().reshape((1, 8)),
            LR: 0.001,
        }

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()
        self.pool1.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()
        self.pool2.reset_parameters()
        self.dense1.reset_parameters()
        self.dense3.reset_parameters()
        self.dense5.reset_parameters()
        return


class DhpoConvMNIST(nn.Module):
    def __init__(self, input_channel, ndim, nclass):
        super(DhpoConvMNIST, self).__init__()
        self.ndim = ndim
        self.nclass = nclass

        self.conv1_1 = DhpoConv2d(in_channels=input_channel)
        self.conv1_2 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv1_3 = DhpoConv2d(in_channels=CHANNELTOP)
        self.pool1 = DhpoPool2d()
        self.conv2_1 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv2_2 = DhpoConv2d(in_channels=CHANNELTOP)
        self.conv2_3 = DhpoConv2d(in_channels=CHANNELTOP)
        self.pool2 = DhpoPool2d()

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=CHANNELTOP * self.ndim * self.ndim, out_features=64)
        self.dense3 = Linear(in_features=64, out_features=32)
        self.dense5 = Linear(in_features=32, out_features=nclass)
        pass

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.dense1(x))
        x = torch.sigmoid(self.dense3(x))
        res = self.dense5(x)
        return res

    def get_hparams(self):
        return {
            CONV11CHANNEL: self.conv1_1.get_hparams()[0],
            CONV11KERNEL: self.conv1_1.get_hparams()[1],
            CONV12CHANNEL: self.conv1_2.get_hparams()[0],
            CONV12KERNEL: self.conv1_2.get_hparams()[1],
            CONV13CHANNEL: self.conv1_3.get_hparams()[0],
            CONV13KERNEL: self.conv1_3.get_hparams()[1],
            POOL1TYPE: self.pool1.get_hparams()[0],
            POOL1KERNEL: self.pool1.get_hparams()[1],
            CONV21CHANNEL: self.conv2_1.get_hparams()[0],
            CONV21KERNEL: self.conv2_1.get_hparams()[1],
            CONV22CHANNEL: self.conv2_2.get_hparams()[0],
            CONV22KERNEL: self.conv2_2.get_hparams()[1],
            CONV23CHANNEL: self.conv2_3.get_hparams()[0],
            CONV23KERNEL: self.conv2_3.get_hparams()[1],
            POOL2TYPE: self.pool2.get_hparams()[0],
            POOL2KERNEL: self.pool2.get_hparams()[1],
            LR: 0.001,
        }

    def get_hparams_tensor(self):
        return {
            CONV11CHANNEL: self.conv1_1.get_hparams_tensor()[0],
            CONV11KERNEL: self.conv1_1.get_hparams_tensor()[1],
            CONV12CHANNEL: self.conv1_2.get_hparams_tensor()[0],
            CONV12KERNEL: self.conv1_2.get_hparams_tensor()[1],
            CONV13CHANNEL: self.conv1_3.get_hparams_tensor()[0],
            CONV13KERNEL: self.conv1_3.get_hparams_tensor()[1],
            "pool1": self.pool1.get_hparams_tensor().reshape((1, 8)),
            CONV21CHANNEL: self.conv2_1.get_hparams_tensor()[0],
            CONV21KERNEL: self.conv2_1.get_hparams_tensor()[1],
            CONV22CHANNEL: self.conv2_2.get_hparams_tensor()[0],
            CONV22KERNEL: self.conv2_2.get_hparams_tensor()[1],
            CONV23CHANNEL: self.conv2_3.get_hparams_tensor()[0],
            CONV23KERNEL: self.conv2_3.get_hparams_tensor()[1],
            "pool2": self.pool2.get_hparams_tensor().reshape((1, 8)),
            LR: 0.001,
        }

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()
        self.pool1.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()
        self.pool2.reset_parameters()
        self.dense1.reset_parameters()
        self.dense3.reset_parameters()
        self.dense5.reset_parameters()
        return


class Dnn(nn.Module):
    def __init__(self, ndim, nclass, h_params):
        super(Dnn, self).__init__()
        self.ndim = ndim
        self.nclass = nclass
        self.h_params = h_params

        self.dense1 = Linear(ndim, h_params[DENSE1SIZE])
        self.dense2 = Linear(h_params[DENSE1SIZE], h_params[DENSE2SIZE])
        self.dense3 = Linear(h_params[DENSE2SIZE], h_params[DENSE3SIZE])
        self.dense4 = Linear(h_params[DENSE3SIZE], nclass)
        self.prelu = PReLU()
    pass

    def forward(self, x):
        x = torch.sigmoid(self.dense1(x))
        x = self.prelu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        x = self.dense4(x)
        return x

    def reset_parameters(self):
        self.dense1.reset_parameters()
        self.dense2.reset_parameters()
        self.dense3.reset_parameters()
        self.dense4.reset_parameters()
        return


class DhpoLinear(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(DhpoLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dense = Linear(self.in_features, self.out_features)
        self.prelu = PReLU()
        self.selecter_base = Parameter(torch.Tensor(1, self.out_features))
        self.selecter_controller = Parameter(torch.Tensor(1))
        self.up_traingle = Parameter(torch.zeros(
            (self.out_features, self.out_features)), requires_grad=False)
        for i in range(self.out_features):
            for j in range(self.out_features):
                if i >= j:
                    self.up_traingle[i, j] = 1
        self.opt_selecter = None
        self.opt_selecter_tensor = None

        self.reset_parameters()
        pass

    def reset_parameters(self):
        self.dense.reset_parameters()
        init.uniform_(self.selecter_base, -1, 1)
        init.uniform_(self.selecter_controller, -1, 1)
        return

    def format_selecter(self):
        EXPANDTIMES = 10
        selecter_p = F.softmax(self.selecter_base, dim=1)
        selecter_score = torch.mm(selecter_p, self.up_traingle)
        selecter_score_zeroed = selecter_score - \
            torch.sigmoid(self.selecter_controller)
        selecter_score_expand = selecter_score_zeroed * EXPANDTIMES
        selecter = torch.sigmoid(selecter_score_expand)
        self.opt_selecter = round(float(torch.sum(selecter)))
        self.opt_selecter_tensor = selecter
        return selecter

    def forward(self, x):
        selecter = self.format_selecter()
        x = self.dense(x)
        if self.activation == SIDMOID:
            x = F.sigmoid(x)
        elif self.activation == PRELU:
            x = self.prelu(x)
        x = torch.mul(x, selecter)
        return x

    def get_hparams(self):
        return self.opt_selecter

    def get_hparams_tensor(self):
        return self.opt_selecter_tensor


class DhpoDnn(nn.Module):
    def __init__(self, ndim, nclass):
        super(DhpoDnn, self).__init__()
        self.ndim = ndim
        self.nclass = nclass

        self.dense1 = DhpoLinear(ndim, DENSETOP, SIDMOID)
        self.dense2 = DhpoLinear(DENSETOP, DENSETOP, PRELU)
        self.dense3 = DhpoLinear(DENSETOP, DENSETOP, SIDMOID)
        self.dense4 = Linear(DENSETOP, nclass)
        self.prelu = PReLU()
    pass

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def get_hparams(self):
        return {
            DENSE1SIZE: self.dense1.get_hparams(),
            DENSE2SIZE: self.dense2.get_hparams(),
            DENSE3SIZE: self.dense3.get_hparams(),
            LR: 0.001
        }

    def get_hparams_tensor(self):
        return {
            DENSE1SIZE: self.dense1.get_hparams_tensor(),
            DENSE2SIZE: self.dense2.get_hparams_tensor(),
            DENSE3SIZE: self.dense3.get_hparams_tensor(),
            LR: 0.001
        }

    def reset_parameters(self):
        self.dense1.reset_parameters()
        self.dense2.reset_parameters()
        self.dense3.reset_parameters()
        self.dense4.reset_parameters()
        return


if __name__ == "__main__":
    data = Data()
    # # exp1
    # train_loader, test_loader = data.load_mnist()
    # cnn = ConvMNIST(input_channel=1, ndim=28, nclass=10, h_params=TESTPARAM)
    # model = DhpoConvMNIST(1, 28, 10)
    # hconv = DhpoConv2d(1)
    # for img, label in train_loader:
    #     res = cnn(img)
    #     model(img)
    #     # res = hconv(img)
    #     # res[0][0][0][0].backward()
    #     # print(res)
    #     break
    # for p in hconv.parameters():
    #     print(p)

    # exp2
    train_data, test_data = data.load_iris()
    model = Dnn(4, 3, TESTPARAMDENSE)
    x, y = train_data
    preds = model(x)
    print(preds)
    pass

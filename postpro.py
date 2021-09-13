from const import *
import pickle
import numpy as np


class Parser():
    def __init__(self, dataset, model) -> None:
        self.dataset = dataset
        self.model = model
        path = "result/{}-{}".format(dataset, model)
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        pass

    def read(self, dataset, model):
        path = "result/{}-{}".format(dataset, model)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def get_accu_best(self):
        res = self.data[2]
        if self.model == RAND:
            res = self.data[1]
        if self.model == GENETICA or self.model == PARTICLESO:
            res = -res[0]
        res = round(float(res*100), 2)
        return res

    def get_accu_topk(self, k=3):
        data = []
        for item in self.data[3]:
            value = 0
            if self.model == BAYES:
                value = float(item['target'])
            elif self.model in [ZOOPT, RAND, GENETICA, PARTICLESO]:
                value = float(item[1])
            elif self.model == DHPO:
                value = float(item[1])
            data.append(value)
        data.sort(reverse=True)
        res = data[:k]
        return round(np.mean(res) * 100, 2), round(np.std(res) * 100, 2)

    def get_accu_all(self):
        data = []
        for item in self.data[3]:
            value = 0
            if self.model == BAYES:
                value = float(item['target'])
            elif self.model in [ZOOPT, RAND, GENETICA, PARTICLESO]:
                value = float(item[1])
            elif self.model == DHPO:
                value = float(item[1])
            data.append(value)
        data.sort(reverse=True)
        res = data
        return round(np.mean(res) * 100, 2), round(np.std(res) * 100, 2)

    def get_time(self):
        return self.data[0]


if __name__ == "__main__":
    for dataset in [MNIST, SVHN]:
        for model in MODELS:
            if model in [GENETICA, PARTICLESO]:
                continue
            data = Parser(dataset, model).get_time()
            # data = Parser(dataset, model).get_accu_best()
            # data = Parser(dataset, model).get_accu_topk(10)
            # data = Parser(dataset, model).get_accu_all()
            print("{}-{}:{}\n".format(dataset, model, data))
    for dataset in SMALLDATASETS:
        for model in MODELS:
            data = Parser(dataset, model).get_time()
            # data = Parser(dataset, model).get_accu_best()
            # data = Parser(dataset, model).get_accu_topk(10)
            # data = Parser(dataset, model).get_accu_all()
            print("{}-{}:{}\n".format(dataset, model, data))

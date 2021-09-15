from const import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from trainers import TrainerDense
from scipy.interpolate import make_interp_spline


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
        if self.model in [RAND, DEHBNAME, HYPERBAND]:
            res = self.data[1]
        if self.model == GENETICA or self.model == PARTICLESO:
            res = -res[0]
        res = round(float(res*100), 2)
        return res

    def get_param_best(self):
        res = self.data[1]
        if self.model in [RAND, DEHBNAME, HYPERBAND]:
            res = self.data[2]
        if isinstance(res, np.ndarray) or isinstance(res, list):
            res = {DENSE1SIZE: res[0],
                   DENSE2SIZE: res[1],
                   DENSE3SIZE: res[2],
                   LR: res[3]}
        if self.model in [BAYES]:
            res = {DENSE1SIZE: int(res[DENSE1SIZE]),
                   DENSE2SIZE: int(res[DENSE2SIZE]),
                   DENSE3SIZE: int(res[DENSE3SIZE]),
                   LR: 0}
        return res

    def get_accu_topk(self, k=3):
        data = []
        for item in self.data[3]:
            value = 0
            if self.model == BAYES:
                value = float(item['target'])
            elif self.model in [ZOOPT, RAND, GENETICA, PARTICLESO, DEHBNAME, HYPERBAND]:
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
            elif self.model in [ZOOPT, RAND, GENETICA, PARTICLESO, HYPERBAND, DEHBNAME]:
                value = float(item[1])
            elif self.model == DHPO:
                value = float(item[1])
            data.append(value)
        data.sort(reverse=True)
        res = data
        return round(np.mean(res) * 100, 2), round(np.std(res) * 100, 2)

    def get_time(self):
        return self.data[0]


def construct_table_data_with_dataset(dataset):
    if dataset in [MNIST, SVHN]:
        models = CONVMODELS
    else:
        models = MODELS
    for model in models:
        data = Parser(dataset, model)
        print(model)
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(int(data.get_time()),
                                                      round(
                                                          data.get_accu_best(), 2),
                                                      round(
                                                          data.get_accu_all()[0], 2),
                                                      round(
                                                          data.get_accu_all()[1], 2),
                                                      round(
                                                          data.get_accu_topk(5)[0], 2),
                                                      round(
                                                          data.get_accu_topk(5)[1], 2),
                                                      round(
                                                          data.get_accu_topk(10)[0], 2),
                                                      round(
                                                          data.get_accu_topk(10)[1], 2),
                                                      ))
    return


def contruct_table_data():
    for dataset in [MNIST, SVHN][:1]:
        for model in MODELS:
            if model in [GENETICA, PARTICLESO]:
                continue
            data = Parser(dataset, model).get_time()
            # data = Parser(dataset, model).get_accu_best()
            # data = Parser(dataset, model).get_accu_topk(5)
            # data = Parser(dataset, model).get_accu_topk(10)
            # data = Parser(dataset, model).get_accu_all()
            print("{}-{}:{}\n".format(dataset, model, data))
    for dataset in SMALLDATASETS:
        for model in MODELS:
            data = Parser(dataset, model).get_time()
            # data = Parser(dataset, model).get_accu_best()
            # data = Parser(dataset, model).get_accu_topk(5)
            # data = Parser(dataset, model).get_accu_topk(10)
            # data = Parser(dataset, model).get_accu_all()
            print("{}-{}:{}\n".format(dataset, model, data))
    return


def construct_loss_with_dataset_and_model(dataset, model):
    path = "result/{}-{}".format(dataset, model)
    with open(path, "rb") as f:
        data = pickle.load(f)
    res = data[-1]
    if model == BAYES:
        accu_seq = [item["target"] for item in data[3]]
    else:
        accu_seq = [item[1] for item in data[3]]

    if model in [DHPO]:
        accu_seq = [item.cpu().numpy() for item in accu_seq]
    index = np.argmax(accu_seq)
    if dataset == SVHN and model == DHPO_ONE_ROUND:
        return np.array(data[-1][index][0])
    if model in [DHPO, DEHBNAME]:
        res = np.array(data[-1][index])
    else:
        res = np.array(data[-1][index]).mean(axis=0)
        if model == DHPO_ONE_ROUND and dataset in SMALLDATASETS:
            res = np.array(data[-1][index])
    if model == DEHBNAME and dataset in SMALLDATASETS:
        res = res.mean(axis=0)
    if model in [HYPERBAND]:
        res = res[:30]
    # print(res)
    return res


def construc_loss_with_dataset(dataset):
    res = []
    if dataset in [MNIST, SVHN]:
        for model in CONVMODELS:
            res.append(construct_loss_with_dataset_and_model(dataset, model))
    else:
        for model in MODELS_with_out_hb:
            print(model)
            res.append(construct_loss_with_dataset_and_model(dataset, model))
    return [i + 1 for i in range(len(res[0]))], res


def plt_loss(dataset):
    x, data = construc_loss_with_dataset(dataset)
    if dataset in [MNIST, SVHN]:
        for i in range(len(data)):
            model = CONVMODELS[i]
            y = data[i]
            plt.plot(x, y, linewidth=1.5, label=model)
    else:
        for i in range(len(data)):
            model = MODELS_with_out_hb[i]
            print(model)
            y = data[i]
            plt.plot(x, y, linewidth=1.5, label=model)
    plt.legend()
    plt.savefig("loss_fig/{}.png".format(dataset))
    return


class ROCAUC():
    def __init__(self) -> None:
        self.dataset = AGARICUS
        self.models = MODELS
        pass

    def run(self):
        for model in self.models:
            parser = Parser(self.dataset, model)
            opt_param = parser.get_param_best()
            print(opt_param)
            trainer = TrainerDense(self.dataset, opt_param)
            trainer.train()
            preds = trainer.model(trainer.test_data[0])
            y = trainer.test_data[1]
            print(preds)
            fpr, tpr, threshold = roc_curve(
                y.cpu(), preds.cpu().detach().numpy())  # 计算真正率和假正率

            roc_auc = auc(fpr, tpr)  # 计算auc的值

            plt.figure()
            lw = 2
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('')
            plt.legend(loc="lower right")
            plt.show()
        return


class case_study():
    def __init__(self) -> None:
        pass

    def plt_loss(self):
        x = [i + 1 for i in range(30)]
        losses = np.array(loss_res).reshape((5, 3, 30)).mean(axis=1)
        for i in range(len(losses)):
            plt.plot(x, losses[i], linewidth=1.5,
                     label="{}th params".format(i + 1))
        plt.legend()
        plt.savefig("case_study_fig/cs_loss_global.png")
        plt.cla()

        # for i in range(len(losses)):
        #     plt.plot(x[5:15], losses[i][5:15], linewidth=1.5,
        #              label="{}th params".format(i + 1))
        # plt.legend()
        # plt.savefig("case_study_fig/cs_loss_local1.png")
        # plt.cla()

        # for i in range(len(losses)):
        #     plt.plot(x[20:30], losses[i][20:30], linewidth=1.5,
        #              label="{}th params".format(i + 1))
        # plt.legend()
        # plt.savefig("case_study_fig/cs_loss_local2.png")
        # plt.cla()

        for i in range(len(losses)):
            plt.plot(x[5:30], losses[i][5:30], linewidth=1.5,
                     label="{}th params".format(i + 1))
        plt.legend()
        plt.savefig("case_study_fig/cs_loss_local.png")
        plt.cla()

        return

    def plt_accu(self):
        x = np.array([i + 1 for i in range(5)])
        accus = np.array(accu_res)
        accu_l = accus[:, 0] - accus[:, 1]
        accu_u = accus[:, 0] + accus[:, 1]
        accu_mean = accus[:, 0]
        x_smooth = np.linspace(x.min(), x.max(), 300)
        accu_mean = make_interp_spline(x, accu_mean)(x_smooth)
        accu_l = make_interp_spline(x, accu_l)(x_smooth)
        accu_u = make_interp_spline(x, accu_u)(x_smooth)
        plt.plot(x_smooth, accu_mean, linewidth=1.5, c="r")
        plt.fill_between(x_smooth, accu_l, accu_u, color="blue", alpha=0.1)
        plt.xticks([1, 2, 3, 4, 5], ["1th", "2th", "3th", "4th", "5th"])
        plt.savefig("case_study_fig/cs_accu.png")
        return


if __name__ == "__main__":
    # dataset = SVHN
    # construct_loss_with_dataset_and_model(dataset, BAYES)
    # construct_loss_with_dataset_and_model(dataset, ZOOPT)
    # construct_loss_with_dataset_and_model(dataset, RAND)
    # construct_loss_with_dataset_and_model(dataset, DHPO)
    # construct_loss_with_dataset_and_model(dataset, HYPERBAND)
    # construct_loss_with_dataset_and_model(dataset, DEHBNAME)
    # construct_loss_with_dataset_and_model(dataset, DHPO_ONE_ROUND)
    # plt_loss(CAR)

    # construct_table_data_with_dataset(CAR)
    study = case_study()
    # study.plt_loss()
    study.plt_accu()
    pass

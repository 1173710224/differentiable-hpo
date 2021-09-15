import pickle
from const import *
from optimizers import Optimizer, OptimizerDense
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--dataset',
                    dest='dataset',
                    action='store',
                    default=MNIST,
                    help='specify a dataset')
parser.add_argument('--model',
                    dest='model',
                    action='store',
                    default="",
                    help='specify a dataset')
args = parser.parse_args()
if __name__ == "__main__":
    if args.model != "":
        dataset = args.dataset
        model = args.model
        if dataset in [MNIST, SVHN]:
            optimizer = Optimizer(dataset)
        else:
            optimizer = OptimizerDense(dataset)
        res = eval("optimizer.{}()".format(model))
        with open("result/{}-{}".format(dataset, model), "wb") as f:
            pickle.dump(res, f)
        print(res)
    elif args.dataset == MNIST or args.dataset == SVHN:
        dataset = args.dataset
        optimizer = Optimizer(dataset)
        res = optimizer.bayes()
        print(res)
        with open("result/{}-{}".format(dataset, BAYES), "wb") as f:
            pickle.dump(res, f)
        res = optimizer.zoopt()
        print(res)
        with open("result/{}-{}".format(dataset, ZOOPT), "wb") as f:
            pickle.dump(res, f)
        res = optimizer.rand()
        print(res)
        with open("result/{}-{}".format(dataset, RAND), "wb") as f:
            pickle.dump(res, f)
        res = optimizer.hyper_band()
        print(res)
        with open("result/{}-{}".format(dataset, HYPERBAND), "wb") as f:
            pickle.dump(res, f)
        res = optimizer.dhpo()
        print(res)
        with open("result/{}-{}".format(dataset, DHPO), "wb") as f:
            pickle.dump(res, f)
        res = optimizer.dhpo_oneround()
        print(res)
        with open("result/{}-{}".format(dataset, DHPO_ONE_ROUND), "wb") as f:
            pickle.dump(res, f)

    elif args.dataset == SMALLS:
        for dataset in SMALLDATASETS:
            optimizer = OptimizerDense(dataset)
            res = optimizer.bayes()
            print(res)
            with open("result/{}-{}".format(dataset, BAYES), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.zoopt()
            print(res)
            with open("result/{}-{}".format(dataset, ZOOPT), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.rand()
            print(res)
            with open("result/{}-{}".format(dataset, RAND), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.ga()
            print(res)
            with open("result/{}-{}".format(dataset, GENETICA), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.pso()
            print(res)
            with open("result/{}-{}".format(dataset, PARTICLESO), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.dhpo()
            print(res)
            with open("result/{}-{}".format(dataset, DHPO), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.dhpo_oneround()
            print(res)
            with open("result/{}-{}".format(dataset, DHPO_ONE_ROUND), "wb") as f:
                pickle.dump(res, f)
            res = optimizer.hyper_band()
            print(res)
            with open("result/{}-{}".format(dataset, HYPERBAND), "wb") as f:
                pickle.dump(res, f)
    pass

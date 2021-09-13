import pickle
from const import *


def read(dataset, model):
    path = "result/{}-{}".format(dataset, model)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    # for dataset in SMALLDATASETS:
    #     for model in MODELS:
    #         data = read(dataset, model)
    #         print("{}-{}\n{}\n".format(dataset, model, data[:3]))
    # for dataset in [MNIST, SVHN]:
    #     for model in MODELS:
    #         if model in [GENETICA, PARTICLESO]:
    #             continue
    #         data = read(dataset, model)
    #         print("{}-{}\n{}\n".format(dataset, model, data[:3]))
    # for dataset in SMALLDATASETS:
    #     for model in MODELS:
    #         data = read(dataset, model)
    #         print("{}-{}\n{}\n".format(dataset, model, data[:3]))
    # for dataset in [MNIST, SVHN]:
    #     for model in MODELS:
    #         if model in [GENETICA, PARTICLESO]:
    #             continue
    # data = read(IRIS, DEHBNAME)
    # print("{}-{}\n{}\n".format(IRIS, DEHBNAME, data[:3]))
    # data = read(MNIST, DEHBNAME)
    # print("{}-{}\n{}\n".format(MNIST, DEHBNAME, data[:4]))
    # for dataset in SMALLDATASETS:
        # data = read(dataset, DEHBNAME)
        # print("{}-{}\n{}\n".format(dataset, DEHBNAME, data[:4]))
    # data = read(SVHN, DEHBNAME)
    # print("{}-{}\n{}\n".format(SVHN, DEHBNAME, data))
    
    # print(read(SVHN, DHPO)[:3])
    # print(read(MNIST, ZOOPT)[:3])
    # data = read(SVHN, HYPERBAND)
    # print("{}-{}\n{}\n".format(SVHN, HYPERBAND, data[:3]))
    # data = read(MNIST, HYPERBAND)
    # print("{}-{}\n{}\n".format(MNIST, HYPERBAND, data[:3]))


    for dataset in SMALLDATASETS:
        data = read(dataset, DEHBNAME)
        print("{}-{}'s objective number:{}".format(dataset,DEHBNAME,len(data[4])))
    pass

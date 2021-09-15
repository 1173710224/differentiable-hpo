import numpy as np
from const import VALTIMES
from random import random
from math import log, ceil


class Hyperband:

    def __init__(self, get_params_function, try_params_function, data, it_n=27):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.data = data
        self.max_iter = it_n  	# maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = {}
        self.results["best_accu"] = 0
        self.results["best_hparams"] = 0
        self.results["respool"] = []
        self.results["loss_set"] = []
        self.best_accu = 0
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        # num = 0
        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            # n=30
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                val_losses = []
                print(len(T))
                for t in T:
                    self.counter += 1
                    if dry_run:
                        result = {'loss': random(),  'accu': random()}
                    else:
                        result = self.try_params(
                            n_iterations, t, self.data)		# <---
                    if result['accu'] == 0:
                        continue

                    acc = result["accu"]
                    loss = result['loss']
                    avg_loss = np.sum(loss)/VALTIMES
                    val_losses.append(avg_loss)
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        self.best_counter = self.counter
                    if acc > self.best_accu:
                        self.results["best_accu"] = acc
                        self.results["best_hparams"] = t
                    self.results["respool"].append((t, acc))
                    self.results["loss_set"].append(loss)
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices]
                T = T[0:int(n_configs / self.eta)]
        return self.results

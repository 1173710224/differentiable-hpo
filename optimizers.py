from torch.optim import optimizer
from const import *
from zoopt import Dimension, Objective, Parameter, Opt
from trainers import DhpoTrainerDense, Trainer, TrainerDense, DhpoTrainer
from bayes_opt import BayesianOptimization
from random import randint, random
from sko.GA import GA
from sko.PSO import PSO
import time
import pickle
import numpy as np
from DEHB.dehb import DEHB

def transform_space(param_space, configuration):
    assert len(configuration) == len(param_space)
    config_dict = dict()
    for i, (k, v) in enumerate(param_space.items()):
        value = configuration[i]
        lower, upper = v[0], v[1]
        is_log = v[3]
        if is_log:
            # performs linear scaling in the log-space
            log_range = np.log(upper) - np.log(lower)
            value = np.exp(np.log(lower) + log_range * value)
        else:
            # linear scaling within the range of the parameter
            value = lower + (upper - lower) * value
        if v[2] == int:
            value = np.round(value).astype(int)
        if k == "":
            config_dict[k] = POOLINDEX2TYPE[value]
        else:
            config_dict[k] = value
    return config_dict

class Optimizer():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        pass

    def bayes(self):
        self.loss_set = []
        optimizer = BayesianOptimization(self.max_obj, {
            CONV11CHANNEL: (1, CHANNELTOP+0.99),
            CONV11KERNEL: (2, KERNELTOP+0.99),
            CONV12CHANNEL: (1, CHANNELTOP+0.99),
            CONV12KERNEL: (2, KERNELTOP+0.99),
            CONV13CHANNEL: (1, CHANNELTOP+0.99),
            CONV13KERNEL: (2, KERNELTOP+0.99),
            POOL1TYPE: (0, 1+0.99),
            POOL1KERNEL: (2, KERNELTOP+0.99),
            CONV21CHANNEL: (1, CHANNELTOP+0.99),
            CONV21KERNEL: (2, KERNELTOP+0.99),
            CONV22CHANNEL: (1, CHANNELTOP+0.99),
            CONV22KERNEL: (2, KERNELTOP+0.99),
            CONV23CHANNEL: (1, CHANNELTOP+0.99),
            CONV23KERNEL: (2, KERNELTOP+0.99),
            POOL2TYPE: (0, 1+0.99),
            POOL2KERNEL: (2, KERNELTOP+0.99),
            LR: (0.0001, 0.01),
        })
        st = time.perf_counter()
        optimizer.maximize(init_points=BAYESINIT, n_iter=BAYESITER)
        return time.perf_counter() - st, optimizer.max["params"], optimizer.max["target"], optimizer.res, self.loss_set

    def max_obj(self, conv11_channel_size, conv11_kernel_size, conv12_channel_size, conv12_kernel_size, conv13_channel_size, conv13_kernel_size, pooling1_type, pooling1_kernel_size, conv21_channel_size, conv21_kernel_size, conv22_channel_size, conv22_kernel_size, conv23_channel_size, conv23_kernel_size, pooling2_type, pooling2_kernel_size, learning_rate):
        h_params = {
            CONV11CHANNEL: int(conv11_channel_size),
            CONV11KERNEL: int(conv11_kernel_size),
            CONV12CHANNEL: int(conv12_channel_size),
            CONV12KERNEL: int(conv12_kernel_size),
            CONV13CHANNEL: int(conv13_channel_size),
            CONV13KERNEL: int(conv13_kernel_size),
            POOL1TYPE: POOLINDEX2TYPE[int(pooling1_type)],
            POOL1KERNEL: int(pooling1_kernel_size),
            CONV21CHANNEL: int(conv21_channel_size),
            CONV21KERNEL: int(conv21_kernel_size),
            CONV22CHANNEL: int(conv22_channel_size),
            CONV22KERNEL: int(conv22_kernel_size),
            CONV23CHANNEL: int(conv23_channel_size),
            CONV23KERNEL: int(conv23_kernel_size),
            POOL2TYPE: POOLINDEX2TYPE[int(pooling2_type)],
            POOL2KERNEL: int(pooling2_kernel_size),
            LR: learning_rate,
        }
        trainer = Trainer(self.dataset, h_params)
        if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
            print("bad hparams")
            return 0
        else:
            accu = float(trainer.objective())
            self.loss_set.append(trainer.multi_loss_seq)
            print("accu:{}".format(accu))
            return accu

    def zoopt(self):
        self.loss_set = []
        self.zoopt_respool = []
        channel_top = CHANNELTOP
        kernel_top = KERNELTOP
        dim = Dimension(
            17,
            [[1, channel_top],
             [2, kernel_top],
             [1, channel_top],
             [2, kernel_top],
             [1, channel_top],
             [2, kernel_top],
             [0, 1],
             [2, kernel_top],

             [1, channel_top],
             [2, kernel_top],
             [1, channel_top],
             [2, kernel_top],
             [1, channel_top],
             [2, kernel_top],
             [0, 1],
             [2, kernel_top],

             [0.0001, 0.01]],
            [False, False, False,
             False, False, False,
             False, False, False,
             False, False, False,
             False, False, False,
             False, True]
        )
        obj = Objective(self.min_obj, dim)
        st = time.perf_counter()
        solution = Opt.min(obj, Parameter(budget=ZOOPTBUDGET))
        return time.perf_counter() - st, solution.get_x(), -solution.get_value(), self.zoopt_respool, self.loss_set

    def min_obj(self, solution):
        h_params = self.format_param(solution)
        trainer = Trainer(self.dataset, h_params)
        if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
            print("bad hparams")
            return 0
        else:
            accu = trainer.objective()
            self.loss_set.append(trainer.multi_loss_seq)
            print("accu:{}".format(accu))
            self.zoopt_respool.append((h_params, accu))
            return -accu

    def format_param(self, solution):
        ans = {}
        param_values = solution.get_x()
        for i in range(len(PARAMNAMES)):
            name = PARAMNAMES[i]
            ans[name] = param_values[i]
            if name == POOL1TYPE or name == POOL2TYPE:
                ans[name] = POOLINDEX2TYPE[param_values[i]]
        return ans

    def rand(self):
        self.loss_set = []
        self.rand_respool = []
        num = 0
        best_hparams = None
        best_accu = 0
        st = time.perf_counter()
        while True:
            h_params = {
                CONV11CHANNEL: randint(1, CHANNELTOP),
                CONV11KERNEL: randint(2, KERNELTOP),
                CONV12CHANNEL: randint(1, CHANNELTOP),
                CONV12KERNEL: randint(2, KERNELTOP),
                CONV13CHANNEL: randint(1, CHANNELTOP),
                CONV13KERNEL: randint(2, KERNELTOP),
                POOL1TYPE: POOLINDEX2TYPE[randint(0, 1)],
                POOL1KERNEL: randint(2, KERNELTOP),
                CONV21CHANNEL: randint(1, CHANNELTOP),
                CONV21KERNEL: randint(2, KERNELTOP),
                CONV22CHANNEL: randint(1, CHANNELTOP),
                CONV22KERNEL: randint(2, KERNELTOP),
                CONV23CHANNEL: randint(1, CHANNELTOP),
                CONV23KERNEL: randint(2, KERNELTOP),
                POOL2TYPE: POOLINDEX2TYPE[randint(0, 1)],
                POOL2KERNEL: randint(2, KERNELTOP),
                LR: random() * (0.01 - 0.0001) + 0.0001,
            }
            trainer = Trainer(self.dataset, h_params)
            if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
                print("bad hparams")
                continue
            else:
                num += 1
                accu = trainer.objective()
                self.loss_set.append(trainer.multi_loss_seq)
                self.rand_respool.append((h_params, accu))
                if accu > best_accu:
                    best_accu = accu
                    best_hparams = h_params
                if num == RANDTIMES:
                    break
                print("accu:{}".format(accu))
        return time.perf_counter() - st, best_accu, best_hparams, self.rand_respool, self.loss_set

    def ga(self):
        self.loss_set = []
        self.ea_respool = []
        ga = GA(func=self.schaffer, n_dim=17, size_pop=EAPOP, max_iter=EAITER, prob_mut=0.001,
                lb=[1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0.0001],
                ub=[CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, 1+0.99, KERNELTOP+0.99,
                    CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, 1+0.99, KERNELTOP+0.99, 0.01], precision=1e-7)
        st = time.perf_counter()
        best_params, best_accu = ga.run()
        return time.perf_counter() - st, best_params, best_accu, self.ea_respool, self.loss_set

    def pso(self):
        self.loss_set = []
        self.ea_respool = []
        pso = PSO(func=self.schaffer, n_dim=17, pop=EAPOP, max_iter=EAITER,
                  lb=[1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0.0001],
                  ub=[CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, 1+0.99, KERNELTOP+0.99,
                      CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, CHANNELTOP+0.99, KERNELTOP+0.99, 1+0.99, KERNELTOP+0.99, 0.01],
                  w=0.8, c1=0.5, c2=0.5)
        st = time.perf_counter()
        best_params, best_accu = pso.run()
        return time.perf_counter() - st, best_params, best_accu, self.ea_respool, self.loss_set

    def schaffer(self, param_values):
        h_params = self.ea_format_param(param_values)
        trainer = Trainer(self.dataset, h_params)
        if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
            return 0
        else:
            accu = trainer.objective()
            self.loss_set.append(trainer.multi_loss_seq)
            self.ea_respool.append((h_params, accu))
            print("accu:{}".format(accu))
            return -accu

    def ea_format_param(self, param_values):
        ans = {}
        print(param_values)
        for i in range(len(PARAMNAMES)):
            name = PARAMNAMES[i]
            ans[name] = int(param_values[i])
            if name == POOL1TYPE or name == POOL2TYPE:
                ans[name] = POOLINDEX2TYPE[int(param_values[i])]
        return ans
    
    def ea_format_param_dehp(self, param_values):
        ans = {}
        print(param_values)
        for name in PARAMNAMES:
            ans[name] = int(param_values[name])
            if name == POOL1TYPE or name == POOL2TYPE:
                ans[name] = POOLINDEX2TYPE[int(param_values[name])]
        return ans

    def dhpo(self):
        self.loss_set = []
        self.dhpo_respool = []
        best_accu = 0
        best_hparams = None
        trainer = DhpoTrainer(self.dataset)
        st = time.perf_counter()
        for _ in range(DHPOTIMES):
            trainer.model.reset_parameters()
            trainer.train()
            self.loss_set.append(trainer.loss_sequence)
            accu = trainer.val()
            self.dhpo_respool.append((trainer.model.get_hparams(), accu))
            if accu > best_accu:
                best_accu = accu
                best_hparams = trainer.model.get_hparams()
            print("accu:{}".format(accu))
        return time.perf_counter() - st, best_hparams, best_accu, self.dhpo_respool, self.loss_set

    def dhpo_oneround(self):
        self.loss_set = []
        self.dhpo_respool = []
        best_accu = 0
        best_hparams = None
        trainer = DhpoTrainer(self.dataset)
        st = time.perf_counter()
        for _ in range(1):
            trainer.model.reset_parameters()
            trainer.train()
            self.loss_set.append(trainer.loss_sequence)
            accu = trainer.objective()
            self.dhpo_respool.append((trainer.model.get_hparams(), accu))
            if accu > best_accu:
                best_accu = accu
                best_hparams = trainer.model.get_hparams()
            print("accu:{}".format(accu))
        return time.perf_counter() - st, best_hparams, best_accu, self.dhpo_respool, self.loss_set

    def hyper_band(self):
        from hyperband import Hyperband
        from get_try_params import get_params_conv, try_params_conv
        hb = Hyperband(get_params_conv, try_params_conv, self.dataset)
        st = time.perf_counter()
        results = hb.run()
        best_accu = results["best_accu"]
        best_hparams = results["best_hparams"]
        respool = results["respool"]
        loss_set = results["loss_set"]
        return time.perf_counter() - st, best_accu, best_hparams, respool, loss_set

    def dehb(self):
        self.loss_set = []
        self.ea_respool = []
        param_space = {
            CONV11CHANNEL: [1,CHANNELTOP,int,False],
            CONV11KERNEL: [2,KERNELTOP,int,False],
            CONV12CHANNEL: [1,CHANNELTOP,int,False],
            CONV12KERNEL: [2,KERNELTOP,int,False],
            CONV13CHANNEL: [1,CHANNELTOP,int,False],
            CONV13KERNEL: [2,KERNELTOP,int,False],
            POOL1TYPE: [0,1,int,False],
            POOL1KERNEL: [2,KERNELTOP,int,False],
            CONV21CHANNEL: [1,CHANNELTOP,int,False],
            CONV21KERNEL: [2,KERNELTOP,int,False],
            CONV22CHANNEL: [1,CHANNELTOP,int,False],
            CONV22KERNEL: [2,KERNELTOP,int,False],
            CONV23CHANNEL: [1,CHANNELTOP,int,False],
            CONV23KERNEL: [2,KERNELTOP,int,False],
            POOL2TYPE: [0,1,int,False],
            POOL2KERNEL: [2,KERNELTOP,int,False],
            LR: [0.0001, 0.01, float, False],
        }
        dimensions = len(param_space)
        # Declaring the fidelity range
        min_budget, max_budget = 2, 50
        dehb = DEHB(
            f=self.target_function, 
            dimensions=dimensions, 
            min_budget=min_budget, 
            max_budget=max_budget,
            n_workers=1,
            output_path = "./dehp_out"
        )
        trajectory, runtime, history = dehb.run(
            fevals=DEHBALLCOST, 
            verbose=False,
            save_intermediate=False,
            max_budget=dehb.max_budget,
            param_space=param_space
        )
        return runtime, -dehb.inc_score, transform_space(param_space, dehb.inc_config), self.ea_respool, self.loss_set

    def target_function(self, config, budget, **kwargs):
        max_budget = kwargs["max_budget"]
        
        # Mapping [0, 1]-vector to Sklearn parameters
        param_space = kwargs["param_space"]
        config = transform_space(param_space, config)
        
        if budget is None:
            budget = max_budget
        config = self.ea_format_param_dehp(config)
        trainer = Trainer(self.dataset, config)
        st = time.perf_counter()
        if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
            return {
                "fitness": -1,  # DE/DEHB minimizes
                "cost": -1,
                "info": {
                    "test_score": -1,
                    "budget": -1
                }
            }
        accu = trainer.objective()
        self.loss_set.append(trainer.loss_sequence)
        self.ea_respool.append((config, accu))
        print("accu:{}".format(accu))
        cost = time.perf_counter() - st
        result = {
            "fitness": -accu,  # DE/DEHB minimizes
            "cost": cost,
            "info": {
                "test_score": accu,
                "budget": budget
            }
        }
        return result

class OptimizerDense():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        pass

    def bayes(self):
        self.loss_set = []
        optimizer = BayesianOptimization(self.max_obj, {
            DENSE1SIZE: (1, DENSETOP + 0.99),
            DENSE2SIZE: (1, DENSETOP + 0.99),
            DENSE3SIZE: (1, DENSETOP + 0.99),
            LR: (0.0001, 0.01),
        })
        st = time.perf_counter()
        optimizer.maximize(init_points=BAYESINIT, n_iter=BAYESITER)
        return time.perf_counter() - st, optimizer.max["params"], optimizer.max["target"], optimizer.res, self.loss_set

    def max_obj(self, dense1_size, dense2_size, dense3_size, learning_rate):
        h_params = {
            DENSE1SIZE: int(dense1_size),
            DENSE2SIZE: int(dense2_size),
            DENSE3SIZE: int(dense3_size),
            LR: learning_rate
        }
        trainer = TrainerDense(self.dataset, h_params)
        accu = float(trainer.objective())
        self.loss_set.append(trainer.multi_loss_seq)
        print("accu:{}".format(accu))
        return accu

    def zoopt(self):
        self.loss_set = []
        self.zoopt_respool = []
        dim = Dimension(
            4,
            [[1, DENSETOP],
             [2, DENSETOP],
             [1, DENSETOP],
             [0.0001, 0.01]],
            [False, False, False, True]
        )
        st = time.perf_counter()
        obj = Objective(self.min_obj, dim)
        solution = Opt.min(obj, Parameter(budget=ZOOPTBUDGET))
        return time.perf_counter() - st, solution.get_x(), -solution.get_value(), self.zoopt_respool, self.loss_set

    def min_obj(self, solution):
        h_params = self.format_param(solution)
        trainer = TrainerDense(self.dataset, h_params)
        accu = trainer.objective()
        self.loss_set.append(trainer.multi_loss_seq)
        print("accu:{}".format(accu))
        self.zoopt_respool.append((h_params, accu))
        return -accu

    def format_param(self, solution):
        ans = {}
        param_values = solution.get_x()
        for i in range(len(PARAMNAMESDENSE)):
            name = PARAMNAMESDENSE[i]
            ans[name] = param_values[i]
        return ans

    def rand(self):
        self.loss_set = []
        self.rand_respool = []
        num = 0
        best_hparams = None
        best_accu = 0
        st = time.perf_counter()
        while True:
            h_params = {
                DENSE1SIZE: randint(1, DENSETOP),
                DENSE2SIZE: randint(1, DENSETOP),
                DENSE3SIZE: randint(1, DENSETOP),
                LR: random() * (0.01 - 0.0001) + 0.0001,
            }
            trainer = TrainerDense(self.dataset, h_params)
            num += 1
            accu = trainer.objective()
            self.loss_set.append(trainer.multi_loss_seq)
            self.rand_respool.append((h_params, accu))
            if accu > best_accu:
                best_accu = accu
                best_hparams = h_params
            if num == RANDTIMES:
                break
            print("accu:{}".format(accu))
        return time.perf_counter() - st, best_accu, best_hparams, self.rand_respool, self.loss_set

    def ga(self):
        self.loss_set = []
        self.ea_respool = []
        ga = GA(func=self.schaffer, n_dim=4, size_pop=EAPOP, max_iter=EAITER, prob_mut=0.001,
                lb=[1, 1, 1, 0.0001],
                ub=[DENSETOP+0.99, DENSETOP+0.99, DENSETOP+0.99, 0.01], precision=1e-7)
        st = time.perf_counter()
        best_params, best_accu = ga.run()
        return time.perf_counter() - st, best_params, best_accu, self.ea_respool, self.loss_set

    def pso(self):
        self.loss_set = []
        self.ea_respool = []
        pso = PSO(func=self.schaffer, n_dim=4, pop=EAPOP, max_iter=EAITER,
                  lb=[1, 1, 1, 0.0001],
                  ub=[DENSETOP+0.99, DENSETOP+0.99, DENSETOP+0.99, 0.01],
                  w=0.8, c1=0.5, c2=0.5)
        st = time.perf_counter()
        best_params, best_accu = pso.run()
        return time.perf_counter() - st, best_params, best_accu, self.ea_respool, self.loss_set

    def schaffer(self, param_values):
        h_params = self.ea_format_param(param_values)
        trainer = TrainerDense(self.dataset, h_params)
        accu = trainer.objective()
        self.loss_set.append(trainer.multi_loss_seq)
        self.ea_respool.append((h_params, accu))
        print("accu:{}".format(accu))
        return -accu

    def ea_format_param(self, param_values):
        ans = {}
        for i in range(len(PARAMNAMESDENSE)):
            name = PARAMNAMESDENSE[i]
            ans[name] = int(param_values[i])
        return ans

    def ea_format_param_dehp(self, param_values):
        ans = {}
        print(param_values)
        for name in PARAMNAMESDENSE:
            ans[name] = int(param_values[name])
        return ans

    def dhpo(self):
        self.loss_set = []
        self.dhpo_respool = []
        best_accu = 0
        best_hparams = None
        trainer = DhpoTrainerDense(self.dataset)
        st = time.perf_counter()
        for _ in range(DHPOTIMES):
            trainer.model.reset_parameters()
            trainer.train()
            self.loss_set.append(trainer.loss_sequence)
            accu = trainer.val()
            self.dhpo_respool.append((trainer.model.get_hparams(), accu))
            if accu > best_accu:
                best_accu = accu
                best_hparams = trainer.model.get_hparams()
            print("accu:{}".format(accu))
        return time.perf_counter() - st, best_hparams, best_accu, self.dhpo_respool, self.loss_set

    def dhpo_oneround(self):
        self.loss_set = []
        self.dhpo_respool = []
        best_accu = 0
        best_hparams = None
        trainer = DhpoTrainerDense(self.dataset)
        st = time.perf_counter()
        for _ in range(1):
            trainer.model.reset_parameters()
            trainer.train()
            self.loss_set.append(trainer.loss_sequence)
            accu = trainer.val()
            self.dhpo_respool.append((trainer.model.get_hparams(), accu))
            if accu > best_accu:
                best_accu = accu
                best_hparams = trainer.model.get_hparams()
            print("accu:{}".format(accu))
        return time.perf_counter() - st, best_hparams, best_accu, self.dhpo_respool, self.loss_set

    def hyper_band(self):
        from hyperband import Hyperband
        from get_try_params import get_params_dense, try_params_dense
        hb = Hyperband(get_params_dense, try_params_dense, self.dataset)
        st = time.perf_counter()
        results = hb.run()
        best_accu = results["best_accu"]
        best_hparams = results["best_hparams"]
        respool = results["respool"]
        loss_set = results["loss_set"]
        return time.perf_counter() - st, best_accu, best_hparams, respool, loss_set

    def dehb(self):
        self.loss_set = []
        self.ea_respool = []
        param_space = {
            DENSE1SIZE: [1,DENSETOP,int,False],
            DENSE2SIZE: [1,DENSETOP,int,False],
            DENSE3SIZE: [1,DENSETOP,int,False],
            LR: [0.0001,0.01,float,False],
        }
        dimensions = len(param_space)
        # Declaring the fidelity range
        min_budget, max_budget = 2, 50
        st = time.perf_counter()
        dehb = DEHB(
            f=self.target_function, 
            dimensions=dimensions, 
            min_budget=min_budget, 
            max_budget=max_budget,
            n_workers=1,
            output_path = "./dehp_out"
        )
        trajectory, runtime, history = dehb.run(
            fevals=DEHBALLCOST, 
            verbose=False,
            save_intermediate=False,
            max_budget=dehb.max_budget,
            param_space=param_space,
        )
        return time.perf_counter()-st, -dehb.inc_score, transform_space(param_space, dehb.inc_config), self.ea_respool, self.loss_set

    def target_function(self, config, budget, **kwargs):
        max_budget = kwargs["max_budget"]
        
        # Mapping [0, 1]-vector to Sklearn parameters
        param_space = kwargs["param_space"]
        config = transform_space(param_space, config)
        
        if budget is None:
            budget = max_budget
        config = self.ea_format_param_dehp(config)
        trainer = TrainerDense(self.dataset, config)
        st = time.perf_counter()       
        accu = trainer.objective()
        self.loss_set.append(trainer.multi_loss_seq)
        self.ea_respool.append((config, accu))
        print("accu:{}".format(accu))
        cost = time.perf_counter() - st
        result = {
            "fitness": -accu,  # DE/DEHB minimizes
            "cost": cost,
            "info": {
                "test_score": accu,
                "budget": budget
            }
        }
        return result


if __name__ == "__main__":
    for dataset in [MNIST, SVHN]:
        optimizer = Optimizer(dataset)
        res = optimizer.dehb()
        print(res)
        with open("result/{}-{}".format(dataset, DEHBNAME), "wb") as f:
            pickle.dump(res, f)
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

        
    for dataset in SMALLDATASETS:
        optimizer = OptimizerDense(dataset)
        res = optimizer.dehb()
        print(res)
        with open("result/{}-{}".format(dataset, DEHBNAME), "wb") as f:
            pickle.dump(res, f)
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
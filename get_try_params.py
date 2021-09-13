# meta classifier
from trainers import Trainer, TrainerDense
from const import *

try:
	from hyperopt import hp
	from hyperopt.pyll.stochastic import sample
except ImportError:
	print ("In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else.")
	
#	

# handle floats which should be integers
# works with flat params
def handle_integers( params ):
	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v
	
	return new_params

space_conv = {
                CONV11CHANNEL: hp.choice("c11",[i for i in range(1, CHANNELTOP)]),
                CONV11KERNEL: hp.choice("k11",[i for i in range(2, KERNELTOP)]),
                CONV12CHANNEL: hp.choice("c12",[i for i in range(1, CHANNELTOP)]),
                CONV12KERNEL: hp.choice("k12",[i for i in range(2, KERNELTOP)]),
                CONV13CHANNEL: hp.choice("c13",[i for i in range(1, CHANNELTOP)]),
                CONV13KERNEL: hp.choice("k13",[i for i in range(2, KERNELTOP)]),
                POOL1TYPE: hp.choice("pt1",[MAXPOOL, AVGPOOL]),
                POOL1KERNEL: hp.choice("pk1",[i for i in range(2, KERNELTOP)]),
                CONV21CHANNEL: hp.choice("c21",[i for i in range(1, CHANNELTOP)]),
                CONV21KERNEL: hp.choice("k21",[i for i in range(2, KERNELTOP)]),
                CONV22CHANNEL: hp.choice("c22",[i for i in range(1, CHANNELTOP)]),
                CONV22KERNEL: hp.choice("k22",[i for i in range(2, KERNELTOP)]),
                CONV23CHANNEL: hp.choice("c23",[i for i in range(1, CHANNELTOP)]),
                CONV23KERNEL: hp.choice("k23",[i for i in range(2, KERNELTOP)]),
                POOL2TYPE: hp.choice("pt2",[MAXPOOL, AVGPOOL]),
                POOL2KERNEL: hp.choice("pk2",[i for i in range(2, KERNELTOP)]),
                LR: hp.uniform('lr', 0, 1)*(0.01 - 0.0001) + 0.0001,
            }

def get_params_conv():
	params = sample(space_conv)
	return handle_integers(params)

def try_params_conv(iteration, params, data):
    trainer = Trainer(data, params, int(round(iteration)))
    if trainer.model.get_flatten_dim() < 16 or trainer.model.get_pooling_dim() < 0:
        print("bad hparams")
        return {'loss':0, 'accu':0}
    else:
        accu = trainer.objective()
        result = {}
        result['loss'] = trainer.multi_loss_seq
        result['accu'] = accu
    return result

space_dense = {
                DENSE1SIZE: hp.choice("d1",[i for i in range(1, DENSETOP)]),
                DENSE2SIZE: hp.choice("d2",[i for i in range(1, DENSETOP)]),
                DENSE3SIZE: hp.choice("d3",[i for i in range(1, DENSETOP)]),
                LR: hp.uniform('lr', 0, 1)*(0.01 - 0.0001) + 0.0001,}

def get_params_dense():
	params = sample(space_dense)
	return handle_integers(params)

def try_params_dense(iteration, params, data):
    trainer = TrainerDense(data, params, int(round(iteration)))

    accu = trainer.objective()
    result = {}
    result['loss'] = trainer.multi_loss_seq
    result['accu'] = accu

    return result
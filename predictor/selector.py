import os
import random
import time
from importlib import reload

import mxnet as mx

import prepare
import util
import memonger


ins_num = 5
ins_prices = [0.085, 0.17, 0.34, 0.68, 1.53]
ins_mems = [4, 8, 16, 32, 72]
ins_cores = [2, 4, 8, 16, 36]

C = 1.4 # TODO: test the accuracy of C
layers = [3, 24, 36, 3]
batch_size = 256
dshape = (batch_size, 3, 64, 64)

DEBUG = True

def predictNetMem(mod):
    act_mem = memonger.get_cost(mod.symbol, data=dshape)
    return act_mem * 1.0 / 1024 / 1024 / 1024 + C

def predictNetTime(mod, dshape, num_core):
    # allocate memory given the input data and label shapes
    train_data = prepare.get_train_iter(dshape)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))

    if not DEBUG:
        num_trails = 1000
        conv_model = prepare.get_trained_model(num_trails, "conv2d", "conv2d_feature")
        pool_model = prepare.get_trained_model(num_trails, "pooling", "pooling_feature")
        fc_model = prepare.get_trained_model(num_trails, "fc", "fc_feature")
        bn_model = prepare.get_trained_model(num_trails, "bn", "bn_feature")
        operator_models = {
            "convolution": conv_model,
            "pooling": pool_model,
            "fullyconnected": fc_model,
            "batchnorm": bn_model,
        }

        predict_train_time = prepare.predict_network(mod, operator_models, dshape, num_core)
    else: # Perfect prediction
        os.environ['OMP_NUM_THREADS'] = str(num_core)
        # Reimport to make environmental variable work
        reload(mx)


        data_iter = mx.io.NDArrayIter(data=mx.random.uniform(-1, 1, dshape), label=mx.random.uniform(-1, 1, (dshape[0])), batch_size=dshape[0])
        for batch in data_iter:
            start = time.time()
            mod.forward(batch)
            mod.backward()
            mx.nd.waitall()
            end = time.time()
            predict_train_time = end - start

    return predict_train_time

def getAllCkptMods():
    thresholds = range(200, 1000, 500)
    models = []
    for threshold in thresholds:
        models.append(util.get_model(dshape, layers=layers, checkpoint=threshold))
    return models


def getInsMinCost(ins_price, ins_mem, ins_core, T):
    # cost = training time * price of instance
    # subject to
    # Training time < T
    # peak memory usage < memory of an instance

    min_cost = 10000000
    mods = getAllCkptMods()
    min_ckpt_mod = None
    # enumerate different ckpt policies
    for cur_ckp_mod in mods:
        min_ckpt_mod = None
        mem = predictNetMem(cur_ckp_mod)
        if mem > ins_mem:
            # use too much memory
            print("Use too much memory")
            continue
        t = predictNetTime(cur_ckp_mod, dshape, ins_core)
        if t > T:
            # does not match time constraint
            print("Take too long to train, predict train time %f, train time limit %f" % (t, T))
            continue
        cur_cost = t * ins_price
        min_cost = min_cost if min_cost < cur_cost else cur_cost
        min_ckpt_mod = min_ckpt_mod if min_cost < cur_cost else cur_ckp_mod

    if min_ckpt_mod is None:
        print("[Error] Cannot find instance meet the training time constraint")
    return min_cost, min_ckpt_mod


def getInstance(T):
    min_cost = 0
    min_ins_idx = -1
    for i in range(ins_num):
        ins_cost, _ = getInsMinCost(ins_prices[i], ins_mems[i], ins_cores[i], T)
        if ins_cost < min_cost:
            min_cost = ins_cost
            min_ins_idx = i
    return min_ins_idx, min_cost


if __name__ == "__main__":
    print("----------------")
    # our method
    min_idx, min_cost = getInstance(100)

    # random selection
    rand_idx = 0

    # max selection
    max_idx = ins_num - 1

    # just fit selection
    no_ckpt_mod = util.get_model(dshape, layers, checkpoint=0)
    mod_mem_no_ckpt = predictNetMem(no_ckpt_mod)
    fit_idx = [i for i, x in enumerate(ins_mems) if x > mod_mem_no_ckpt][0]

    print("Auto Select %d %f, Just fix Select %d" % (min_idx, min_cost, fit_idx))






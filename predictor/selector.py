import os
import gc
import psutil
import random
import time
from importlib import reload

import mxnet as mx

import prepare
import util
import memonger


ins_num = 5
ins_prices = [0.085, 0.17, 0.34, 0.68, 1.53]
#ins_mems = [4, 8, 16, 32, 72]
#ins_cores = [2, 4, 8, 16, 36]
ins_prices =[0.34]
ins_num = 1
ins_mems = [8]
ins_cores = [4]

C = 1.4 # TODO: test the accuracy of C

layers = [3, 24, 36, 3]
batch_size = 2048
dshape = (batch_size, 3, 64, 64)

DEBUG = True
NUM_ITER = 10000

def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    # print('memory GB:', memoryUse)
    return memoryUse

def predictNetMem(mod):
    act_mem = memonger.get_cost(mod.symbol, data=dshape)
    return act_mem * 1.0 / 1024 + C

def predictNetTime(mod, dshape, num_core):
    # allocate memory given the input data and label shapes
    train_data = util.get_train_iter(dshape)
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
        mx.cpu().empty_cache()


        data_iter = mx.io.NDArrayIter(data=mx.random.uniform(-1, 1, dshape), label=mx.random.uniform(-1, 1, (dshape[0])), batch_size=dshape[0])
        for batch in data_iter:
            start = time.time()
            mod.forward(batch, is_train=True)
            mx.nd.waitall()
            mod.backward()
            mx.nd.waitall()
            mem = cpuStats()
            end = time.time()
            predict_train_time = end - start
        del mod
        del data_iter
        gc.collect()

    return predict_train_time, mem

def getAllCkptMods():
    thresholds = range(700,2000, 500)
    models = []
    for threshold in thresholds:
        mod = util.get_model(dshape, layers=layers, checkpoint=threshold)
        models.append(mod)
#        print(mod.symbol.debug_str())
    return models


def getInsMinCost(ins_price, ins_mem, ins_core, T):
    # cost = training time * price of instance
    # subject to
    # Training time < T
    # peak memory usage < memory of an instance

    min_cost = 10000000
    mods = getAllCkptMods()
    min_ckpt_mod = None
    min_t = 0
    min_mem = 0
    # enumerate different ckpt policies
    for cur_ckp_mod in mods:
        min_ckpt_mod = None
        mem = predictNetMem(cur_ckp_mod)
        print("Predict Memory:", mem)
        t, mem = predictNetTime(cur_ckp_mod, dshape, ins_core)
        #print("Actual Memory:", mem)

        if mem > ins_mem:
            # use too much memory
            print("Use too much memory")
            continue

        if t > T:
            # does not match time constraint
            print("Take too long to train, predict train time %f, train time limit %f" % (t, T))
            continue

        cur_cost = t / 3600.0 * ins_price * NUM_ITER

        print("Cur cost", t, ins_price, cur_cost, min_cost)
        min_ckpt_mod = min_ckpt_mod if min_cost < cur_cost else cur_ckp_mod
        min_t = min_t if min_cost < cur_cost else t
        min_mem = min_mem if min_cost < cur_cost else mem
        min_cost = min_cost if min_cost < cur_cost else cur_cost

    if min_ckpt_mod is None:
        print("[Error] Cannot find instance meet the training time constraint")
    return min_cost, min_ckpt_mod, min_t, min_mem


def getInstance(T):
    min_cost = 100000
    min_ins_idx = -1
    min_t = -1
    min_mem = -1
    for i in range(ins_num):
        ins_cost, _, ins_t, ins_mem = getInsMinCost(ins_prices[i], ins_mems[i], ins_cores[i], T)
        if ins_cost < min_cost:
            min_cost = ins_cost
            min_ins_idx = i
            min_t = ins_t
            min_mem = ins_mem

    return min_ins_idx, min_cost, min_t, min_mem


if __name__ == "__main__":
    print("----------------")
    # our method
    min_idx, min_cost, min_t, min_mem = getInstance(100)

    # random selection
    rand_idx = 0

    # max selection
    max_idx = ins_num - 1

    # just fit selection
    no_ckpt_mod = util.get_model(dshape, layers, checkpoint=0)
    mod_mem_no_ckpt = predictNetMem(no_ckpt_mod)
    print("Predict No Checkpoint Mem", mod_mem_no_ckpt)
    fit_inst = [i for i, x in enumerate(ins_mems) if x > mod_mem_no_ckpt]
    fit_idx = -1 if len(fit_inst) == 0 else fit_inst[0]
    mod_t_no_ckpt, mem = predictNetTime(no_ckpt_mod, dshape, ins_cores[fit_idx])
    print("Actual No Checkpoint Mem", mem)

    print("Auto Select %d %f %f %f,\nJust fix Select %d %f %f %f" % (min_idx, min_cost, min_t, min_mem, fit_idx, ins_prices[fit_idx] * mod_t_no_ckpt / 3600.0 * NUM_ITER , mod_t_no_ckpt, ins_mems[fit_idx]))






import time
import sys

import mxnet as mx
from mxnet import profiler

sys.path.append('../')
import util


profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')


if __name__ == "__main__":
    dshape = (1280, 3, 64, 64)
    repeat_times = 10

    mod = util.getResNet50Model()
    train_data = util.get_train_iter(dshape)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))


    i = 0
    for batch in train_data:
        if i == 1:
            profiler.set_state('run')
        mod.forward(batch, is_train=True)
        mx.nd.waitall()
        mod.backward()
        i += 1
        if i == repeat_times:
            break

    mx.nd.waitall()
    profiler.set_state('stop')
    #profiler.dump()
    print(profiler.dumps())


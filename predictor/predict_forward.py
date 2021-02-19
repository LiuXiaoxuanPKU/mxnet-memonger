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
    dshape = (128, 3, 64, 64)
    repeat_times = 10

    model_name = "res152"
    mod = util.get_model(dshape,0,model_name)
    train_data = util.get_train_iter(dshape)
    if model_name == "vgg":
        label_name = "prob_label"
    else:
        label_name = "softmax_label"
    print(train_data.provide_label)
    label = mx.io.DataDesc(name=label_name,
            shape = train_data.provide_label[0].shape,
            dtype = train_data.provide_label[0].dtype,
            layout = train_data.provide_label[0].layout)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=[label])
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))


    i = 0
    for batch in train_data:
        if i == 1:
            profiler.set_state('run')
        mod.forward(batch, is_train=True)
        mod.backward()
        i += 1
        if i == repeat_times:
            break

    mx.nd.waitall()
    profiler.set_state('stop')
    profiler.dump()
    print(profiler.dumps())


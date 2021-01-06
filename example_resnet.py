import sys
import os
import math
import mxnet as mx

import time
import psutil
import gc

from mxnet import profiler
from util import *

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')

def cpuStats():
    # print(sys.version)
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    # print('memory GB:', memoryUse)
    return memoryUse


jitter_param = 0.4
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]

def get_train_iter(dshape):
    return mx.io.ImageRecordIter(
        path_imgrec='./tiny-imagenet_train.rec',
        path_imgidx='./tiny-imagenet_train.idx',
        preprocess_threads=4,
        shuffle=True,
        batch_size=dshape[0],

        data_shape=(dshape[1], dshape[2], dshape[3]),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )



if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    threshold = int(sys.argv[2])
    num_threads = str(sys.argv[3])

    layers = [3, 24, 36, 3]
    dshape = (batch_size, 3, 38, 38)
    mod = get_model(dshape, layers=layers, checkpoint=threshold)

    # allocate memory given the input data and label shapes
    train_data = get_train_iter(dshape)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    # use accuracy as the metric
    metric = mx.metric.create('acc')

    arg_params, aux_params = mod.get_params()
    param_size = 0
    for key in arg_params.keys():
        param_size += arg_params[key].size * 4
    for key in aux_params.keys():
       # print(key, aux_params[key])
        param_size += aux_params[key].size * 4
    print("Parameter size", param_size / 1024 / 1024, " MB")

    repeat_times = 10
    profiler.set_state('run')
    # profiler.pause()
    # train 5 epochs, i.e. going over the data iter one pass
    start = time.time()
    for epoch in range(5):
        train_data.reset()
        metric.reset()
        for i, batch in enumerate(train_data):
            if i == 1:
                profiler.resume()
            mod.forward(batch, is_train=True)       # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()                          # compute gradients
            mod.update()                            # update parameters
            if i == repeat_times: # benchmark 100 iterations
                break

        # print('Epoch %d, Training %s' % (epoch, metric.get()))
        mx.nd.waitall()
        profiler.set_state('stop')
        profiler.dump()
        end = time.time()
        time_per_img = (end - start) * 1.0 / batch_size / repeat_times
        print("batch\tthreshold\tthread number\ttime per image\tmemory (GB)")
        print("%d\t%d\t%s\t%s\t%f" %(batch_size, threshold, os.environ["MXNET_CPU_WORKER_NTHREADS"], time_per_img,  cpuStats()))
import sys
import os
import math
import mxnet as mx
import memonger

import time
import psutil
import gc

import numpy as np
from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet import gluon, nd
from gluoncv.utils import makedirs, TrainingHistory
from mxnet import autograd as ag
from mxnet import profiler, monitor

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


def ConvModule(sym, num_filter, kernel, pad=(0, 0), stride=(1, 1), fix_gamma=True):
    conv = mx.sym.Convolution(data=sym, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=fix_gamma)
    act = mx.sym.LeakyReLU(data=bn, act_type="leaky") # same memory to our act, less than CuDNN one
    return act

def ResModule(sym, base_filter, stage, layer, fix_gamma=True):
    num_f = base_filter * int(math.pow(2, stage))
    s = 1
    if stage != 0 and layer == 0:
        s = 2
    conv1 = ConvModule(sym, num_f, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv2 = ConvModule(conv1, num_f, kernel=(3, 3), pad=(1, 1), stride=(s, s))
    conv3 = ConvModule(conv2, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(1, 1))

    if layer == 0:
        sym = ConvModule(sym, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(s, s))

    sum_sym = sym + conv3
    # Annotate the critical points that can be saved as inter-stage parameter
    sym._set_attr(mirror_stage='True')
    return sum_sym

def get_symbol(layers):
    """Get a 4-stage residual net, with configurations specified as layers.

    Parameters
    ----------
    layers : list of stage configuratrion
    """
    assert(len(layers) == 4)
    base_filter = 64
    data = mx.sym.Variable(name='data')
    conv1 = ConvModule(data, base_filter, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    mp1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2))
    sym = mp1
    for j in range(len(layers)):
        for i in range(layers[j]):
            sym = ResModule(sym, base_filter, j, i)

    # for i in range(15):
    #     sym = ConvModule(sym, base_filter, kernel=(3, 3), pad=(2, 2), stride=(1, 1))
    #
    avg = mx.symbol.Pooling(data=sym, kernel=(2, 2), stride=(1, 1), name="global_pool", pool_type='avg')
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=200, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return net


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

def get_model(dshape, layers, checkpoint=0):
    net = get_symbol(layers)
    old_cost = memonger.get_cost(net, data=dshape)
    print('Old feature map cost=%d MB' % old_cost)
    if checkpoint > 0:
      #  net = memonger.search_plan(net, data=dshape)
        plan_info = {}
        net = memonger.make_mirror_plan(net, checkpoint, plan_info, data=dshape)
        print(plan_info)
        new_cost = memonger.get_cost(net, data=dshape)
        print('New feature map cost=%d MB' % new_cost)
        exit()
    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    return mod



if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    threshold = int(sys.argv[2])
    num_threads = str(sys.argv[3])

    layers = [3, 24, 36, 3]
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
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
        exit()

#
#
# num_cpus = 4
# ctx = [mx.cpu(i) for i in range(num_cpus)]
#
# lr_decay = 0.1
# # Epochs where learning rate decays
# lr_decay_epoch = [30, 60, 90, np.inf]
#
# # Nesterov accelerated gradient descent
# optimizer = 'nag'
# # Set parameters
# optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}
#
# # # Define our trainer for net
# # trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
# #
# # loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
#
# epochs = 120
# lr_decay_count = 0
# log_interval = 50
#
# acc_top1 = mx.metric.Accuracy()
# acc_top5 = mx.metric.TopKAccuracy(5)
# train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
#                                  'validation-top1-err', 'validation-top5-err'])
#
#
#
# def test(ctx, val_data):
#     acc_top1_val = mx.metric.Accuracy()
#     acc_top5_val = mx.metric.TopKAccuracy(5)
#     for i, batch in enumerate(val_data):
#         data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
#         label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
#         outputs = [net(X) for X in data]
#         acc_top1_val.update(label, outputs)
#         acc_top5_val.update(label, outputs)
#
#     _, top1 = acc_top1_val.get()
#     _, top5 = acc_top5_val.get()
#     return (1 - top1, 1 - top5)
#
# for epoch in range(epochs):
#     tic = time.time()
#     btic = time.time()
#     acc_top1.reset()
#     acc_top5.reset()
#
#     # if lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
#     #     trainer.set_learning_rate(trainer.learning_rate*lr_decay)
#     #     lr_decay_count += 1
#
#     for i, batch in enumerate(train_data):
#         print(type(batch))
#         data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
#         print(type(data))
#         label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
#         # with ag.record():
#         #     outputs = [net(X) for X in data]
#         #     loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
#         # ag.backward(loss)
#         # trainer.step(batch_size)
#         # acc_top1.update(label, outputs)
#         # acc_top5.update(label, outputs)
#         # if log_interval and not (i + 1) % log_interval:
#         #     _, top1 = acc_top1.get()
#         #     _, top5 = acc_top5.get()
#         #     err_top1, err_top5 = (1-top1, 1-top5)
#         #     print('Epoch[%d] Batch [%d]     Speed: %f samples/sec   top1-err=%f     top5-err=%f'%(
#         #               epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1, err_top5))
#         #     btic = time.time()
#
#     # _, top1 = acc_top1.get()
#     # _, top5 = acc_top5.get()
#     # err_top1, err_top5 = (1-top1, 1-top5)
#     #
#     # err_top1_val, err_top5_val = test(ctx, val_data)
#     # train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])
#     #
#     # print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epoch, err_top1, err_top5))
#     # print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
#     # print('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))
#
# # train_history.plot(['training-top1-err', 'validation-top1-err'])


# output_total_size = 0
# weight_total_size = 0
# gradient_total_size = 0
# def cal_output(x):
#     print("I am in output-----")
#     global output_total_size
#     output_total_size += x.size * 4
#     return mx.nd.array([x.size * 4])
#
# def cal_weight(x):
#     global weight_total_size
#     weight_total_size += x.size * 4
#     return mx.nd.array([x.size * 4])
#
# def cal_gradient(x):
#     global gradient_total_size
#     gradient_total_size += x.size * 4
#     return mx.nd.array([x.size * 4])


#output_mon = monitor.Monitor(1, stat_func=cal_output, pattern='.*output.*', sort=True)
#weight_mon = monitor.Monitor(1, stat_func=cal_weight, pattern='.*', sort=True)
#gradient_mon = monitor.Monitor(1, stat_func=cal_gradient, pattern='.*backward.*', sort=True)

#mod.install_monitor(output_mon)
#mod.install_monitor(weight_mon)
#mod.install_monitor(gradient_mon)

# val_data = mx.io.ImageRecordIter(
#     path_imgrec         = './tiny-imagenet_test.rec',
#     path_imgidx         = './tiny-imagenet_test.idx',
#     preprocess_threads  = 32,
#     shuffle             = False,
#     batch_size          = 256,
#
#     resize              = 256,
#     data_shape          = (3, 64, 64),
#     mean_r              = mean_rgb[0],
#     mean_g              = mean_rgb[1],
#     mean_b              = mean_rgb[2],
#     std_r               = std_rgb[0],
#     std_g               = std_rgb[1],
#     std_b               = std_rgb[2],
# )
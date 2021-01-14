import random
import time
import numpy as np
from numpy.polynomial.polynomial import polyfit

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os
import sys
sys.path.append('../')
from util import *

batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
heights = [16, 32, 64, 128, 256]
widths = [16, 32, 64, 128, 256]
kernels = [1, 3, 5, 7, 9, 11, 13, 15, 17]
strides = [1, 2, 3, 5, 7]
pads = [1, 2, 3]
num_filters = [12, 24, 36, 48, 96, 128, 256]
layouts = [None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC']
pooling_types = ['max', 'avg']
hidden_max = 500
hidden_min = 100

batches = [1]
heights = [3]
widths = [3]
kernels = [1]
strides = [1]
pads = [0]
num_filters = [1]
DEBUG = False

repeat_times = 100
if mx.context.num_gpus():
    device = mx.gpu()
else:
    device = mx.cpu()
metric = mx.metric.Loss()

random.seed(3)

def SGD(key, weight, grad, grad_norm, lr=0.001):
    # key is key for weight, we can customize update rule
    # weight is weight array
    # grad is grad array
    # lr is learning rate
    # grad_norm is scalar to norm gradient, usually it is batch_size
    norm = 1.0 / grad_norm
    # here we can bias' learning rate 2 times larger than weight
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
    else:
        pass

def get_executor(operator, dshape):
    label = mx.sym.Variable('label')
    loss = mx.symbol.MakeLoss(mx.sym.square(label - operator))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(operator), loss])
    executor = pred_loss.simple_bind(ctx=device, data=dshape, grad_req='write')
    return executor

def generate_params():
    all_params = {}
    channel = 3
    while True:
        batch_size = batches[random.randint(0, len(batches) - 1)]
        height = heights[random.randint(0, len(heights) - 1)]
        width = widths[random.randint(0, len(widths) - 1)]
        if batch_size * height * width < (2 << 20):
            break
    kernel_value = min(height, width)
    while kernel_value >= min(height, width):
        kernel_value = kernels[random.randint(0, len(kernels) - 1)]
    kernel = (kernel_value, kernel_value)
    stride_value = strides[random.randint(0, len(strides) - 1)]
    stride = (stride_value, stride_value)
    pad_value = pads[random.randint(0, len(pads) - 1)]
    pad = (pad_value, pad_value)
    num_filter = num_filters[random.randint(0, len(num_filters) - 1)]
    layout = layouts[random.randint(0, len(layouts) - 1)]

    pooling_type = pooling_types[random.randint(0, len(pooling_types) - 1)]
    hidden_num = random.randint(hidden_min, hidden_max)

    all_params['channel'] = channel
    all_params['batch_size'] = batch_size
    all_params['height'] = height
    all_params['width'] = width
    all_params['kernel_value'] = kernel_value
    all_params['stride_value'] = stride_value
    all_params['pad_value'] = pad_value
    all_params['num_filter'] = num_filter
    all_params['pooling_type'] = pooling_type
    all_params['hidden_num'] = hidden_num
    return all_params


def generate_training_data(num_trails, file_name, name="conv2d"):
    for i in range(num_trails):
        params = {}
        data = mx.symbol.stop_gradient(mx.symbol.Variable('data'))

        all_params = generate_params()
        params['batch_size'] = all_params['batch_size']
        params['height'] = all_params['height']
        params['width'] = all_params['width']
        dshape = (all_params['batch_size'],
                  all_params['channel'],
                  all_params['height'],
                  all_params['width'])

        if name == "conv2d":
        # 1. generate the operator
            # conv2d
            print(params)
            params['kernel_value'] = all_params['kernel_value']
            params['stride_value'] = all_params['stride_value']
            params['pad_value'] = all_params['pad_value']
            params['num_filter'] = all_params['num_filter']
            operator = mx.sym.Convolution(data=data,
                                      kernel=(all_params['kernel_value'], all_params['kernel_value']),
                                      stride=(all_params['stride_value'], all_params['stride_value']),
                                      pad=(all_params['pad_value'], all_params['pad_value']),
                                      num_filter=all_params['num_filter'],
                                      name=name)

        elif name == "bn":
            operator = mx.symbol.BatchNorm(data, name='bn')

        elif name == "pooling":
            params['kernel_value'] = all_params['kernel_value']
            params['stride_value'] = all_params['stride_value']
#            params['pooling_type'] = all_params['pooling_type']
            # pooling
            operator = mx.symbol.Pooling(data=data,
                                         kernel=(all_params['kernel_value'], all_params['kernel_value']),
                                         stride=(all_params['stride_value'], all_params['stride_value']),
                                         pool_type=all_params['pooling_type'])

        elif name == "relu":
            raise NotImplemented
        elif name == "fc":
            params['hidden_num'] = all_params['hidden_num']
            operator = mx.symbol.FullyConnected(data=data,
                                                num_hidden=all_params['hidden_num'],
                                                name='fc')

        # Get executor
        executor = get_executor(operator, dshape)
        args = executor.arg_dict
        print(args.keys())

        # 1. Generate inputs, outputs
        inputs = []
        labels = []
        if name + '_weight' in args.keys():
            args[name + '_weight'][:] = mx.random.uniform(-1, 1, args[name + '_weight'].shape)
        if name + '_bias' in args.keys():
            args[name + '_bias'][:] = mx.random.uniform(-1, 1, args[name + '_bias'].shape)
        if name + '_gamma' in args.keys():
            args[name + '_gamma'][:] = mx.random.uniform(-1, 1, args[name + '_gamma'].shape)
        if name + '_beta' in args.keys():
            args[name + '_beta'][:] = mx.random.uniform(-1, 1, args[name + '_beta'].shape)

        for _ in range(repeat_times):
            data = mx.nd.random.uniform(0, 1,
                                        shape=dshape,
                                        ctx=device, dtype='float32')
            inputs.append(data)
            args["data"][:] = data
            executor.forward(is_train=False)
            labels.append(executor.outputs[0].copy())


        if name + '_weight' in args.keys():
            args[name + '_weight'][:] = mx.random.uniform(-1, 1, args[name + '_weight'].shape)
        if name + '_bias' in args.keys():
            args[name + '_bias'][:] = mx.random.uniform(-1, 1, args[name + '_bias'].shape)
        if name + '_gamma' in args.keys():
            args[name + '_gamma'][:] = mx.random.uniform(-1, 1, args[name + '_gamma'].shape)
        if name + '_beta' in args.keys():
            args[name + '_beta'][:] = mx.random.uniform(-1, 1, args[name + '_beta'].shape)

        # 2. run the operator multiple times, get the training time
        start = time.time()
        keys = operator.list_arguments()
        for i in range(repeat_times):
            out = executor.forward(is_train=True, data=inputs[i], label=labels[i])
            if DEBUG:
               # print(out[0])
                loss = (out[0] - labels[i]).asnumpy()
                print("Loss", i, loss.sum())

            executor.backward()
            for key in keys:
                SGD(key, executor.arg_dict[key], executor.grad_dict[key], grad_norm=all_params['batch_size'])

        mx.nd.waitall()
        end = time.time()
#        print("Exec time: %f" % (end - start))

        # 3. record to file, param1, param2, param3..., training time
        with open(file_name, "a") as f:
            for key in params.keys():
                f.write(key + ",")
            f.write("train_time\n")
            for key in params.keys():
                f.write(str(params[key]) + ",")
            f.write(str(end - start) + "\n")


def extract_features(file_name):
    x = []
    y = []
    feature_names = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                feature_names = line.split(',')
            else:
                x.append([float(x) for x in line.split(',')[:-1]])
                y.append([float(line.split(',')[-1])])

    return feature_names, x, y 

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    print(np.array(y_test).shape)
    return X_train, y_train, X_test, y_test


def train(x, y, name="poly"):
    model = None
    print(x[0])
    print(y[0])
    if name == "svm":
        model = svm.SVR()
    elif name == 'linear':
        model = LinearRegression()
    elif name == 'poly':
        model = Pipeline([
                ("poly", PolynomialFeatures(degree=4)),
                ("std_scaler", StandardScaler()),
                ("lin_reg", LinearRegression())
                ])
    elif name == "fc":
        pass
    model.fit(x, y)
    print("Train accuracy :", model.score(x, y))
    return model


def test(x, y, model):
    acc = model.score(x, y)
    print("Test accuracy :", acc)

def get_trained_model(num_trails, opt, filename):
    print("Fit model for operator " + opt + "-" * 20)
    if not os.path.exists(filename):
        generate_training_data(num_trails, filename, opt)
    feature_names, x, y = extract_features(filename)
    train_x, train_y, test_x, test_y = split(x, y)
    model = train(train_x, train_y)
    test(test_x, test_y, model)
    return model, feature_names

def predict_network(mod, models):
    sym = mod.symbol
    params = mod.get_params()
    outputs = mod.get_outputs()
    for param in params:
        for key in param.keys():
            print (key, param[key].shape)
        print("-" * 20)

    for output in outputs:
        print(type(output))

    # total_time = 0
    # for internal in sym.get_internals():
    #     if internal.name.startswith("batchnorm") and len(internal.name.split("_")) == 1:
    #         print("In Batchnorm: ", internal.infer_shape())
    #         exit()
    #     elif internal.name.startswith("convolution") and len(internal.name.split("_")) == 1:
    #         print("In Convolution: ", internal)
    #     elif internal.name.startswith("leakyrelu") and len(internal.name.split("_")) == 1:
    #         print("In LeakyRelu: ", internal)


if __name__ == "__main__":
    layers = [3, 24, 36, 3]
    batch_size = 128
    dshape = (batch_size, 3, 38, 38)
    mod = get_model(dshape, layers=layers, checkpoint=0)

    # allocate memory given the input data and label shapes
    train_data = get_train_iter(dshape)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))

    predict_network(mod, {})

    num_trails = 100

    # model_conv, conv_fea = get_trained_model(num_trails, "conv2d", "conv2d_feature")
    # model_pooling, pool_fea = get_trained_model(num_trails, "pooling", "pooling_feature")
    # model_fc, fc_fea = get_trained_model(num_trails, "fc", "fc_feature")
    # model_bn, bn_fea = get_trained_model(num_trails, "bn", "bn_feature")
    # print(bn_fea)

    # module_trails = 1
    # predict_time = []
    # actual_time = []
    # for i in range(module_trails):
    #     data = mx.symbol.stop_gradient(mx.symbol.Variable('data'))
    #     all_params = generate_params()
    #     conv = ConvModule(data,
    #                       all_params['num_filter'],
    #                       kernel=(all_params['kernel_value'], all_params['kernel_value']),
    #                       pad=(all_params['pad_value'], all_params['pad_value']),
    #                       stride=(all_params['stride_value'], all_params['stride_value']), fix_gamma=True)
    #     executor = get_executor(conv, dshape=(all_params['batch_size'],
    #                                           all_params['channel'],
    #                                           all_params['height'],
    #                                           all_params['width']))
    #
    #     start = time.time()
    #     for i in range(repeat_times):
    #         executor.forward()
    #         executor.backward()
    #     mx.nd.waitall()
    #     end = time.time()
    #     print("Actual run time:", end - start)
    #     actual_time.append(end - start)
    #
    #     if 'train_time\n' in conv_fea:
    #         conv_fea.remove('train_time\n')
    #     if 'train_time\n' in bn_fea:
    #         bn_fea.remove('train_time\n')
    #
    #     pred_conv = model_conv.predict(pd.DataFrame(all_params, index=[0])[conv_fea].to_numpy())
    #     pred_bn = model_bn.predict(pd.DataFrame(all_params, index=[0])[bn_fea].to_numpy())
    #     print("Predict run time", pred_conv + pred_bn)
    #     predict_time.append(pred_conv + pred_bn)
    #
    # with open("predict_result.txt", "w") as f:
    #     for i, t in enumerate(predict_time):
    #         f.write("%f,%f\n" % (predict_time[i], actual_time[i]))
    #
    # plt.scatter(predict_time, actual_time)
    # m, b = polyfit(predict_time, actual_time, 1)
    # plt.plot(predict_time, m * predict_time + b)
    # score = r2_score(predict_time, actual_time)
    # print("R^2 for prediction vs actual run time", score)
    # plt.savefig("result")
    #
    #
    #









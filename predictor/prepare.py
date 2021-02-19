import os
import random
import sys
import time

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

sys.path.append('/Users/xiaoxuanliu/Documents/UCB/research/mxnet-memonger')
from util import *
from ast import literal_eval as make_tuple

# batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# heights = [16, 32, 64, 128]
# widths = [16, 32, 64, 128]
# kernels = [1, 3, 5, 7, 9, 11, 13, 15, 17]
# strides = [1, 2, 3, 5, 7]
# pads = [1, 2, 3]
# num_filters = [12, 24, 36, 48, 96, 128]
layouts = [None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC']
pool_types = ['max', 'avg']
hidden_max = 500
hidden_min = 100
cpu_num = 8
DEBUG = False
SPLIT = "order"

batches = [1]
heights = [3]
widths = [3]
kernels = [1]
strides = [1]
pads = [0]
num_filters = [1]


repeat_times = 100
if mx.context.num_gpus():
    device = mx.gpu()
else:
    device = mx.cpu()
metric = mx.metric.Loss()

random.seed(42)


C_conv = 1
C_pool = 2
C_bn = 260

def cost_mode(read_num, write_num, flops):
    return (max(read_num, flops) + write_num)

def cost_model_predict(name, data):
    if name == "conv":
        output_w = int((data['W'] + 2 * data['pad'][0] - data['kernel'][0] + 1) / data['stride'][0])
        output_h = int((data['H'] + 2 * data['pad'][1] - data['kernel'][1] + 1) / data['stride'][1])
        write_num = output_w * output_h * data['N'] * data['OC']
        read_num = write_num * data['kernel'][0] * data['kernel'][1] * data['IC']
        flops = data['H'] * data['W'] * data['kernel'][0] * data['kernel'][1] * data['IC'] * data['OC'] * data['N'] * 2
        forward_t = C_conv * cost_mode(read_num, write_num, flops)
        backward_t = forward_t * 2
        return forward_t, backward_t
    elif name == "pool":
        read_num = 0
        write_num = 0
        flops = 0
        return 0, 0
    elif name == "batchnorm":
        write_num = data['N'] * data['C'] * data['H'] * data['W']
        read_num = data['N'] * data['C'] * data['H'] * data['W'] * 2
        flops = data['N'] * data['C'] * data['H'] * data['W'] * 2
        forward_t = C_bn * cost_mode(read_num, write_num, flops)
        backward_t = forward_t
        return forward_t, backward_t
    elif name == "relu":
        return 0, 0
    else:
        print ("Unsupport operator type %s, return 0" % name)
        return 0, 0


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

    pool_type = pool_types[random.randint(0, len(pool_types) - 1)]
    hidden_num = random.randint(hidden_min, hidden_max)

    all_params['channel'] = channel
    all_params['batch_size'] = batch_size
    all_params['height'] = height
    all_params['width'] = width
    all_params['kernel_value'] = kernel_value
    all_params['stride_value'] = stride_value
    all_params['pad_value'] = pad_value
    all_params['num_filter'] = num_filter
    all_params['pool_type'] = pool_type
    all_params['hidden_num'] = hidden_num

    # number of threads to run the program
    all_params['cpu_num'] = str(random.randint(1, cpu_num))
    return all_params


def get_train_time(category, executor, operator, inputs, labels, all_params):
    keys = operator.list_arguments()
    if category == "forward+backward":
        start = time.time()
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

    elif category == "forward":
        start = time.time()
        for i in range(repeat_times):
            out = executor.forward(is_train=True, data=inputs[i], label=labels[i])
        mx.nd.waitall()
        end = time.time()

    elif category == "backward":
        start = time.time()
        for i in range(repeat_times):
            executor.backward()
            for key in keys:
                SGD(key, executor.arg_dict[key], executor.grad_dict[key], grad_norm=all_params['batch_size'])
        mx.nd.waitall()
        end = time.time()

    print("Exec time: %f" % (end - start))
    return end - start


def generate_training_data(num_trails, file_name, category, name, cpu_num):
    print(name)
    for i in range(num_trails):
        params = {}
        data = mx.symbol.stop_gradient(mx.symbol.Variable('data'))

        all_params = generate_params()
        params['batch_size'] = all_params['batch_size']
        params['height'] = all_params['height'] if name != "fc" else all_params['batch_size']
        params['width'] = all_params['width'] if name != "fc" else all_params['height'] * all_params['width']
        params['cpu_num'] = cpu_num

        dshape = (all_params['batch_size'],
                  all_params['channel'],
                  all_params['height'],
                  all_params['width'])
        dshape = dshape if name != "fc" else (dshape[0], dshape[2] * dshape[3])

        if name == "conv":
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

        elif name == "pool":
            params['kernel_value'] = all_params['kernel_value']
            params['stride_value'] = all_params['stride_value']
            operator = mx.symbol.Pooling(data=data,
                                         kernel=(all_params['kernel_value'], all_params['kernel_value']),
                                         stride=(all_params['stride_value'], all_params['stride_value']),
                                         pool_type=all_params['pool_type'])

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
        train_time = get_train_time(category, executor, operator, inputs, labels, all_params)

        # 3. record to file, param1, param2, param3..., training time
        with open(file_name, "a") as f:
            for key in params.keys():
                f.write(key + ",")
            f.write("train_time\n")
            for key in params.keys():
                f.write(str(params[key]) + ",")
            f.write(str(train_time) + "\n")


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

def random_split(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
    return train_x, train_y, test_x, test_y

def order_split(x, y, feature_names):
    test_per = 0.33
    data = np.array([l + y[i] for i,l in enumerate(x)])
    print(data.shape, feature_names)
    df = pd.DataFrame(data, columns=feature_names)
    sort_feature = feature_names[1]
    print("Sort by feature values ", sort_feature )
    df = df.sort_values(by=sort_feature)
    train_data = df.to_numpy()[:int(df.shape[0]*(1-test_per))]
    test_data = df.to_numpy()[int(df.shape[0]*(1-test_per)):]
    print(test_data[-1])
    train_x = train_data[:, :-1]
    print (train_x[:5])
    train_y = train_data[:, -1]
    print (train_y[:5])
    test_x = test_data[:, :-1]
    print (test_x[:5])
    test_y = test_data[:, -1]
    print (test_y[:5])
    return train_x, train_y, test_x, test_y

def train_test(filename):
    feature_names, x, y = extract_features(filename)
    if SPLIT == "random":
        train_x, train_y, test_x, test_y = random_split(x, y)
    elif SPLIT == "order":
        train_x, train_y, test_x, test_y = order_split(x, y, feature_names)
    else:
        print("[Error] Unknown split type ", SPLIT)
        exit()
    model = train(train_x, train_y)
    test(test_x, test_y, model)
    # TODO: feature_names
    return model

def get_trained_model(num_trails, opt, filename, cpu_num):
    print("Fit model for operator " + opt + "-" * 20)
    forward_filename = filename + "_forward"
    backward_filename = filename + "_backward"
    print(forward_filename)
    if not os.path.exists(forward_filename):
        generate_training_data(num_trails, forward_filename, "forward", opt, cpu_num)

    if not os.path.exists(backward_filename):
        generate_training_data(num_trails, backward_filename, "backward", opt, cpu_num)

    # forward
    forward_model = train_test(forward_filename)
    backward_model = train_test(backward_filename)

    # backward
    return {"forward" : forward_model, "backward" : backward_model}


def copy_symbol(name, attrs):
    op = None
    data = mx.symbol.Variable('data')
    print(name)
    if name.find("conv") != -1:
        op = mx.sym.Convolution(data=data,
                                kernel=make_tuple(attrs['kernel']),
                                stride=make_tuple(attrs['stride']),
                                pad=make_tuple(attrs['pad']),
                                num_filter=make_tuple(attrs['num_filter']))
    elif name.find("leaky") != -1:
        op = mx.sym.LeakyReLU(data=data, act_type="leaky")  # same memory to our act, less than CuDNN one
    elif name.find("batchnorm") != -1 or name.find("bn") != -1:
        print("In batchnorm-------")
        print(attrs)
        op = mx.sym.BatchNorm(data=data, fix_gamma=True)
    elif name.find("pool") != -1:
        op = mx.sym.Pooling(data=data, pool_type=attrs['pool_type'],
                            kernel=make_tuple(attrs['kernel']),
                            stride=make_tuple(attrs['stride']))
    elif name.find("fc") != -1:
        op = mx.symbol.FullyConnected(data=data, num_hidden=int(attrs['num_hidden']))
    elif name.find("relu") != -1:
        op = mx.symbol.relu(data=data)
    elif name.find("flatten") != -1:
        op = mx.symbol.Flatten(data=data, name='flatten')
    return op


def generate_x(mod, dshape, num_core):
    sym = mod.symbol
    attrs = sym.list_attr()
    print(sym.name)
    name = sym.name
    data = {}

    if name.find("dense") != -1:
        batch_size = 1
        height = dshape[0]
        width = dshape[1]
    else:
        batch_size = dshape[0]
        height = dshape[2]
        width = dshape[3]

    cpu_num = num_core
    if name.find("conv") != -1:
        # batch_size,height,width,kernel_value,stride_value,pad_value,num_filter,cpu_num
        kernel = make_tuple(attrs['kernel'])
        stride = make_tuple(attrs['stride'])
        pad = make_tuple(attrs['pad'])
        num_filter = make_tuple(attrs['num_filter'])
        # cpu_num = int(os.environ['OMP_NUM_THREADS'])
        data['N'] = batch_size
        data['IC'] = dshape[1]
        data['H'] = height
        data['W'] = width
        data['OC'] = num_filter
        data['stride'] = stride
        data['pad'] = pad
        data['kernel'] = kernel
        name = "conv"
        return [[batch_size, height, width, cpu_num, kernel[0], stride[0], pad[0], num_filter]], data, name

    elif name.find("batchnorm") != -1 or name.find("bn") != -1:
        # batch_size,height,width
        name = "batchnorm"
        data['N'] = batch_size
        data['C'] = dshape[1]
        data['H'] = height
        data['W'] = width
        return [[batch_size, height, width, cpu_num]], data, name

    elif name.find("pool") != -1:
        # batch_size,height,width,kernel_value,stride_value
        kernel_value = make_tuple(attrs['kernel'])[0]
        stride_value = make_tuple(attrs['stride'])[0]
        name = "pool"
        return [[batch_size, height, width, cpu_num, kernel_value, stride_value]], data, name
    elif name.find("dense") != -1:
        # batch_size,height,width,hidden_num
        hidden_num = int(attrs['num_hidden'])
        name = "dense"
        return [[batch_size, height, width, cpu_num, hidden_num]], data, name
    else:
        print("[Error] Unsupport type ", name)
        return None, None, name

def get_opt_name(layer_name):
    opts = ['conv', 'pool', 'batchnorm', 'dense', 'bn']
    for opt in opts:
        if layer_name.find(opt) != -1:
            return opt
    return None

def predict_network(mod, models, org_dshape, num_core):
    sym = mod.symbol
    total_train_time = 0
    total_conv_time = 0
    total_bn_time = 0

    dshape = org_dshape
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])

    all_layers = sym.get_internals()
    for layer_name in all_layers.list_outputs():
        if layer_name.endswith("output"):
            # print(layer_name)

            print(all_layers[layer_name].list_attr())
            operator_sym = copy_symbol(layer_name, all_layers[layer_name].list_attr())

            if operator_sym is None:
                operator_sym = all_layers[layer_name]
                dshape = org_dshape

            # print(all_layers[layer_name].debug_str())

            # Execute to get shape information
            operator_mod = mx.mod.Module(symbol=operator_sym, label_names=None, context=device)
            operator_mod.bind(for_training=False, data_shapes=[('data', dshape)])
            if layer_name.find("softmax") == -1:
                operator_mod.init_params()
                operator_mod.forward(Batch([mx.random.uniform(-1, 1, dshape)]))

                opt_name = get_opt_name(layer_name)
                opt_t = 0
                # plus layer is not counted
                if opt_name:
                    x, data, name = generate_x(operator_mod, dshape, num_core)
                    forward_t, backward_t = cost_model_predict(name, data)
                    if name == "batchnorm":
                        total_bn_time += forward_t
                    if name == "conv":
                        total_conv_time += forward_t

                    # x = generate_x(operator_mod, dshape, num_core)
                    # print(layer_name, x)
                    # forward_t = models[opt_name]["forward"].predict(x)
                    # backward_t = models[opt_name]["backward"].predict(x)

                    print(opt_name, ", forward ", forward_t, ", backward ", backward_t)
                    opt_t = forward_t + 2 * backward_t if operator_mod.symbol.attr("mirror_stage") else forward_t + backward_t

                dshape = operator_mod.get_outputs()[0].shape
                total_train_time += opt_t


    print("Predict forward conv ", total_conv_time)
    print("Predict forward bn ", total_bn_time)

    return total_train_time


if __name__ == "__main__":
    batch_size = 256
    dshape = (batch_size, 3, 224, 224)

    num_trails = 1000
    cpu_num = int(sys.argv[1])

    # conv_filename = "conv"
    # generate_training_data(num_trails, conv_filename + "_forward", "forward", "conv", cpu_num)
    # generate_training_data(num_trails, conv_filename + "_backward", "backward", "conv", cpu_num)
    #
    # pooling_filename = "pooling"
    # generate_training_data(num_trails, pooling_filename + "_forward", "forward", "pooling", cpu_num)
    # generate_training_data(num_trails, pooling_filename + "_backward", "backward", "pooling", cpu_num)
    #
    # bn_filename = "bn"
    # generate_training_data(num_trails, bn_filename + "_forward", "forward", "bn", cpu_num)
    # generate_training_data(num_trails, bn_filename + "_backward", "backward", "bn", cpu_num)
    #
    # fc_filename = "fc"
    # generate_training_data(num_trails, fc_filename + "_forward", "forward", "fc", cpu_num)
    # generate_training_data(num_trails, fc_filename + "_backward", "backward", "fc", cpu_num)
    #

    # dir = "./"
    # pool_model = get_trained_model(num_trails, "pooling", dir + "pooling", cpu_num)
    # exit()
    # conv_model= get_trained_model(num_trails,  "conv", dir + "conv", cpu_num)
    # fc_model = get_trained_model(num_trails, "fc", dir + "fc", cpu_num)
    # bn_model = get_trained_model(num_trails, "bn", dir + "bn", cpu_num)
    # operator_models = {
    #     "conv" : conv_model,
    #     "pooling" : pool_model,
    #     "dense" : fc_model,
    #     "batchnorm" : bn_model,
    # }
    #
    # print(conv_model['forward'].predict([[256, 64, 64, 8, 5, 1, 1, 256]]))
    # print('---------------------------')
    # exit()

    mod = get_model(dshape, 0, "res50")

    # allocate memory given the input data and label shapes
    train_data = get_train_iter(dshape)
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))

    # predict_train_time = predict_network(mod, operator_models, dshape, 8)
    predict_train_time = predict_network(mod, None, dshape, 8)
    print("Predict train time", predict_train_time)

    # start = time.time()
    # for epoch in range(5):
    #     print("Eopch---", epoch)
    #     train_data.reset()
    #     metric.reset()
    #     for i, batch in enumerate(train_data):
    #         print("Batch---", i)
    #         mod.forward(batch, is_train=True)  # compute predictions
    #         mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
    #         mod.backward()  # compute gradients
    #         mod.update()  # update parameters
    #         mx.nd.waitall()
    #         exit()
    #         if i == repeat_times:  # benchmark 100 iterations
    #             break
    #
    #     # print('Epoch %d, Training %s' % (epoch, metric.get()))
    #     mx.nd.waitall()
    #     end = time.time()
    #     time_per_img = (end - start)
    #     print("Actual train time", time_per_img)


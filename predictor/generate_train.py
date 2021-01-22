import time
import sys

import mxnet as mx
from ast import literal_eval as make_tuple

sys.path.append('../')
from util import *

DEVICE = mx.cpu()

def copy_symbol(name, attrs):
    op = None
    data = mx.symbol.Variable('data')
    names = name.split("_")
    if names[0].startswith("conv"):
        name = names[0]
    elif len(names) == 2 or names[1].startswith("pool") or names[1].startswith("dense")\
            or names[1].startswith("conv") or  names[1].startswith("batchnorm"):
        name = names[1]
    else:
        name = names[2]
    if name.startswith("conv"):
        op = mx.sym.Convolution(data=data,
                                kernel=make_tuple(attrs['kernel']),
                                stride=make_tuple(attrs['stride']),
                                pad=make_tuple(attrs['pad']),
                                num_filter=make_tuple(attrs['num_filter']))
    elif name.startswith("leaky"):
        op = mx.sym.LeakyReLU(data=data, act_type="leaky")  # same memory to our act, less than CuDNN one
    elif name.startswith("batchnorm"):
        op = mx.sym.BatchNorm(data=data, fix_gamma=True)
    elif name.startswith("pool"):
        op = mx.sym.Pooling(data=data, pool_type=attrs['pool_type'],
                            kernel=make_tuple(attrs['kernel']),
                            stride=make_tuple(attrs['stride']))
    elif name.startswith("dense"):
        op = mx.symbol.FullyConnected(data=data, num_hidden=int(attrs['num_hidden']), name="dense")
    elif name.startswith("relu"):
        op = mx.symbol.relu(data=data)
    elif name.startswith("flatten"):
        op = mx.symbol.Flatten(data=data, name='flatten')
    return op


def get_opt_name(layer_name):
    opts = ['conv', 'pool', 'batchnorm', 'dense']
    print(layer_name)
    names = layer_name.split('_')
    if len(names) < 2:
        return None
    for opt in opts:
        if names[2].startswith(opt) or names[1].startswith(opt):
            return opt
    return None

def generate_x(mod, dshape, num_core):
    sym = mod.symbol
    attrs = sym.list_attr()
    name = sym.name

    if name.startswith("dense"):
        batch_size = 1
        height = dshape[0]
        width = dshape[1]
        channel = 1
    else:
        batch_size = dshape[0]
        channel = dshape[1]
        height = dshape[2]
        width = dshape[3]

    cpu_num = num_core
    if name.startswith("conv"):
        # batch_size,channel,height,width,cpu_num, kernel_value,stride_value,pad_value,num_filter,cpu_num
        kernel_value = make_tuple(attrs['kernel'])[0]
        stride_value = make_tuple(attrs['stride'])[0]
        pad_value = make_tuple(attrs['pad'])[0]
        num_filter = make_tuple(attrs['num_filter'])
        # cpu_num = int(os.environ['OMP_NUM_THREADS'])
        return [[batch_size, channel, height, width, cpu_num, kernel_value, stride_value, pad_value, num_filter]]
    elif name.startswith("batchnorm"):
        # batch_size,channel,height,width,cpu_num
        return [[batch_size, channel, height, width, cpu_num]]
    elif name.startswith("pooling"):
        # batch_size,channel,height,width,cpu_num,kernel_value,stride_value
        kernel_value = make_tuple(attrs['kernel'])[0]
        stride_value = make_tuple(attrs['stride'])[0]
        return [[batch_size, channel, height, width, cpu_num, kernel_value, stride_value]]
    elif name.startswith("dense"):
        # batch_size,channel,height,width,cpu_num, hidden_num
        hidden_num = int(attrs['num_hidden'])
        return [[batch_size, channel, height, width, cpu_num, hidden_num]]


def generate_mod_x(mod, org_dshape, num_core):
    sym = mod.symbol
    xs = {}
    dshape = org_dshape
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])

    print(type(mod._aux_params))
    all_layers = sym.get_internals()
    for layer_name in all_layers.list_outputs():
        if layer_name.endswith("output"):
            operator_sym = copy_symbol(layer_name, all_layers[layer_name].list_attr())

            if operator_sym is None:
                operator_sym = all_layers[layer_name]
                dshape = org_dshape

            # Execute to get shape information
            operator_mod = mx.mod.Module(symbol=operator_sym, label_names=None, context=DEVICE)
            print(layer_name, dshape, operator_sym.list_attr())
            operator_mod.bind(for_training=False, data_shapes=[('data', dshape)])
            if layer_name != "softmax_output":
                operator_mod.init_params()
                operator_mod.forward(Batch([mx.random.uniform(-1, 1, dshape)]))
                opt_name = get_opt_name(layer_name)
                # plus layer is not counted
                if opt_name:
                    x = generate_x(operator_mod, dshape, num_core)
                    if not opt_name in xs.keys():
                        xs[opt_name] = [x]
                    else:
                        xs[opt_name].append(x)
                dshape = operator_mod.get_outputs()[0].shape

    return xs

def build_sym(x, opt_name):
    data = mx.symbol.stop_gradient(mx.symbol.Variable('data'))
    batch_size = x[0]
    channel = x[1]
    heigth = x[2]
    width = x[3]

    if opt_name == "conv":
        # [[batch_size, channel, height, width, cpu_num, kernel_value, stride_value, pad_value, num_filter]]
        operator = mx.sym.Convolution(data=data,
                                      kernel=(x[5], x[5]),
                                      stride=(x[6], x[6]),
                                      pad=(x[7], x[7]),
                                      num_filter=x[8],
                                      name=opt_name)
    elif opt_name == "batchnorm":
        # batch_size,channel,height,width,cpu_num
        operator = mx.symbol.BatchNorm(data, name=opt_name)
    elif opt_name == "pool":
        # batch_size,channel,height,width,cpu_num, kernel_value,stride_value
        operator = mx.symbol.Pooling(data=data,
                                     kernel=(x[5], x[5]),
                                     stride=(x[6], x[6]),
                                     pool_type='avg')
    elif opt_name == "dense":
        # batch_size,channel,height,width,cpu_num, hidden_num
        operator = mx.symbol.FullyConnected(data=data,
                                            num_hidden=x[5],
                                            name=opt_name)
    else:
        print("[Error] Unknown operator name " + opt_name)
        exit()
    return operator, (batch_size, channel, heigth, width)

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

def get_train_time(category, executor, operator, dshape):
    repeat_times = 10
    inputs = []
    labels = []
    keys = operator.list_arguments()

    for _ in range(repeat_times):
        data = mx.nd.random.uniform(0, 1,
                                    shape=dshape,
                                    ctx=DEVICE, dtype='float32')
        inputs.append(data)
        executor.forward(is_train=False)
        labels.append(executor.outputs[0].copy())

    if category == "forward":
        start = time.time()
        for i in range(repeat_times):
            _ = executor.forward(is_train=True, data=inputs[i], label=labels[i])
        mx.nd.waitall()
        end = time.time()

    elif category == "backward":
        start = time.time()
        for i in range(repeat_times):
            executor.backward()
            for key in keys:
                SGD(key, executor.arg_dict[key], executor.grad_dict[key], grad_norm=shape[0])
        mx.nd.waitall()
        end = time.time()

    exec_time = (1.0 * (end - start) / repeat_times)
    print("Exec time: %f" % exec_time)
    return exec_time

def build_lookup_table(xs):
    for opt_name in xs.keys():
        if opt_name == "conv":
            continue
        data = xs[opt_name]
        for x in data:
            x = x[0]
            sym, dshape = build_sym(x, opt_name)
            label = mx.sym.Variable('label')
            loss = mx.symbol.MakeLoss(mx.sym.square(label - sym))
            pred_loss = mx.sym.Group([mx.sym.BlockGrad(sym), loss])
            executor = pred_loss.simple_bind(ctx=DEVICE, data=dshape, grad_req='write')
            file_prefix = "table_"
            with open(file_prefix + opt_name + "_forward", "a") as f:
                forward_t = get_train_time("forward", executor, sym, dshape)
                for feature in x:
                    f.write(str(feature)+",")
                f.write(str(forward_t)+"\n")

            with open(file_prefix + opt_name + "_backward", "a") as f:
                backward_t = get_train_time("backward", executor, sym, dshape)
                for feature in x:
                    f.write(str(feature)+",")
                f.write(str(backward_t)+"\n")


def generate_train(mod, shape, num_core):
    xs = generate_mod_x(mod, shape, num_core)
    print(xs)
    build_lookup_table(xs)


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    num_core = int(sys.argv[2])

    mod = getResNet50Model()
    shape = (batch_size, 3, 64, 64)
    generate_train(mod, shape, num_core)
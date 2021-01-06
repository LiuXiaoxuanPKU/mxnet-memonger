import random
import time

import mxnet as mx
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
heights = [16, 32, 64, 128, 256]
widths = [16, 32, 64, 128, 256]
kernels = [1, 3, 5, 7, 9, 11, 13, 15, 17]
strides = [1, 2, 3, 5, 7]
pads = [1, 2, 3]
num_filters = [12, 24, 36, 48, 96, 128, 256]
layouts = [None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC']

# batches = [1]
# heights = [3]
# widths = [3]
# kernels = [1]
# strides = [1]
# pads = [0]
# num_filters = [1]
DEBUG = False

repeat_times = 100
if mx.context.num_gpus():
    device = mx.gpu()
else:
    device = mx.cpu()
metric = mx.metric.Loss()

random.seed(0)

def SGD(key, weight, grad, grad_norm, lr=0.00001):
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

def generate_training_data(num_trails, file_name, name="conv2d"):
    for i in range(num_trails):
        params = {}
        if name == "conv2d":
        # 1. generate the operator
            # conv2d
            channel = 1
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

            params['batch_size'] = batch_size
            params['height'] = height
            params['width'] = width
            params['kernel_value'] = kernel_value
            params['stride_value'] = stride_value
            params['pad_value'] = pad_value
            params['num_filter'] = num_filter
            print(params)

            data = mx.symbol.stop_gradient(mx.symbol.Variable('data'))
            conv = mx.sym.Convolution(data=data,
                                      kernel=kernel,
                                      stride=stride,
                                      pad=pad,
                                      num_filter=num_filter,
                                      name=name)
            label = mx.sym.Variable('label')
            loss = mx.symbol.MakeLoss(mx.sym.square(label - conv))
            pred_loss = mx.sym.Group([mx.sym.BlockGrad(conv), loss])

            executor = pred_loss.simple_bind(ctx=device, data=(batch_size, channel, height, width), grad_req='write')
            # grads = executor.grad_dict
            # aux_states = executor.aux_dict
            # outputs = dict(zip(conv.list_outputs(), executor.outputs))
            #
            # # print("args %s" % args.keys())
            # # print("grads %s" % grads.keys())
            # # print("aux_states %s" % aux_states.keys())
            # # print("outputs %s" % outputs.keys())
            # # print("-" * 20)


        elif type == "pooling":
            pass
            # pooling
        elif type == "relu":
            pass
            # relu
        # fully connected


        dshape = (batch_size, channel, height, width)
        args = executor.arg_dict
        operator = conv
        loss = pred_loss

        # 1. Generate inputs, outputs
        inputs = []
        labels = []
        args[name + '_weight'][:] = mx.random.uniform(-1, 1, args[name + '_weight'].shape)
        args[name + '_bias'][:] = mx.random.uniform(-1, 1, args[name + '_bias'].shape)
        for _ in range(repeat_times):
            data = mx.nd.random.uniform(0, 1,
                                        shape=dshape,
                                        ctx=device, dtype='float32')
            inputs.append(data)
            args["data"][:] = data
            executor.forward(is_train=False)
            labels.append(executor.outputs[0])

        args[name + '_weight'][:] = mx.random.uniform(-1, 1, args[name + '_weight'].shape)
        args[name + '_bias'][:] = mx.random.uniform(-1, 1, args[name + '_bias'].shape)

        # 2. run the operator multiple times, get the training time
        start = time.time()
        keys = operator.list_arguments()
        ex = loss.simple_bind(ctx=device, data=(batch_size, channel, height, width))
        for i in range(repeat_times):
            ex.forward(is_train=True, data=inputs[i], label=labels[i])
            ex.backward()
            for key in keys:
                SGD(key, ex.arg_dict[key], ex.grad_dict[key], grad_norm=batch_size)

            if DEBUG:
                loss = (ex.outputs[0] - labels[i]).asnumpy()
                print("Loss", i, loss.sum())

        mx.nd.waitall()
        end = time.time()
        print("Exec time: %f" % (end - start))

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
    return X_train, y_train, X_test, y_test


def train(x, y, name="svm"):
    model = None
    if name == "svm":
        model = svm.SVC(kernel='linear')
        model.fit(x, y)
    print("Train accuracy :", r2_score(y, model.predict(x)))
    return model


def test(x, y, model):
    predict_y = model.predict(x)
    acc = r2_score(y, predict_y)
    print("Test accuracy :", acc)

if __name__ == "__main__":
    # layers = [3, 24, 36, 3]
    # batch_size = 128
    # dshape = (batch_size, 3, 38, 38)
    # mod = get_model(dshape, layers=layers, checkpoint=0)
    file_name = "conv2d_feature"
    num_trails = 100
    #generate_training_data(num_trails, file_name)
    feature_names, x, y = extract_features(file_name)
    train_x, train_y, test_x, test_y = split(x, y)
    model = train(train_x, train_y)
    # test(test_x, test_y, model)




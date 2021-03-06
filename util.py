import math

import mxnet as mx
import memonger


def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    params = block.collect_params()
    arg_params = {}
    aux_params = {}
    for k, v in params.items():
        if v._stype == 'default':
            data = v.data()
        else:
            raise NotImplemented("stype {} is not yet supported for parameters in block2symbol.")
        arg_params[k] = data
        aux_params[k] = data
    return sym, arg_params, aux_params

def getResNet50Model():
    net = mx.gluon.model_zoo.vision.resnet50_v1(pretrained=True)
    sym, arg_params, aux_params = block2symbol(net)
    # Need name = softmax so that label_names can handle softmax_label
    mx_sym = mx.sym.SoftmaxOutput(data=sym, name='softmax')
    model = mx.mod.Module(symbol=mx_sym, context=mx.cpu(),
                          label_names=['softmax_label'])

    return model

def getAlexNet():
    net = mx.gluon.model_zoo.vision.alexnet(pretrained=True)
    sym, arg_params, aux_params = block2symbol(net)
    # Need name = softmax so that label_names can handle softmax_label
    mx_sym = mx.sym.SoftmaxOutput(data=sym, name='softmax')
    model = mx.mod.Module(symbol=mx_sym, context=mx.cpu(),
                          label_names=['softmax_label'])
    return model

def getVGG16():
    net = mx.gluon.model_zoo.vision.vgg16(pretrained=True)
    sym, arg_params, aux_params = block2symbol(net)
    # Need name = softmax so that label_names can handle softmax_label
    mx_sym = mx.sym.SoftmaxOutput(data=sym, name='softmax')
    model = mx.mod.Module(symbol=mx_sym, context=mx.cpu(),
                          label_names=['softmax_label'])
    return model

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

    avg = mx.symbol.Pooling(data=sym, kernel=(2, 2), stride=(1, 1), name="global_pool", pool_type='avg')
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=200, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return net


def get_model(dshape, checkpoint, name):
    if name == "res50":
        net = getResNet50Model()
        net = net.symbol
    elif name == "vgg":
        net, arg_params, aux_params = mx.model.load_checkpoint("vgg16", 0)
    elif name == "res152":
        net, arg_params, aux_params = mx.model.load_checkpoint("resnet-152", 0)
    else:
        print("Unsupport network type ", name)
        raise NotImplementedError

    old_cost = memonger.get_cost(net, data=dshape)
    print('Old feature map cost=%d MB' % old_cost)
    if checkpoint > 0:
      #  net = memonger.search_plan(net, data=dshape)
        plan_info = {}
        net = memonger.make_mirror_plan(net, checkpoint, plan_info, data=dshape)
        print(plan_info)
        new_cost = memonger.get_cost(net, data=dshape)
        print('New feature map cost=%d MB' % new_cost)
    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    return mod

def get_train_iter(dshape):
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    return mx.io.ImageRecordIter(
        # path_imgrec='/home/ec2-user/mxnet-memonger/tiny-imagenet_train.rec',
        # path_imgidx='/home/ec2-user/mxnet-memonger/tiny-imagenet_train.idx',
        path_imgrec='/Users/xiaoxuanliu/Documents/UCB/research/mxnet-memonger/tiny-imagenet_train.rec',
        path_imgidx='/Users/xiaoxuanliu/Documents/UCB/research//mxnet-memonger/tiny-imagenet_train.idx',
        preprocess_threads=36,
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
    get_model((128,3,64,64), 0, "res152")

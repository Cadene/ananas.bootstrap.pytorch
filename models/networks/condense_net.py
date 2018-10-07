from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class LearnedGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, 
                 condense_factor=None, dropout_rate=0.):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :] = self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])
        
    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### flatten
    x = x.view(batchsize, -1, height, width)
    return x


class CondensingLinear(nn.Module):
    def __init__(self, model, drop_rate=0.5):
        super(CondensingLinear, self).__init__()
        self.in_features = int(model.in_features*drop_rate)
        self.out_features = model.out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))
        _, index = model.weight.data.abs().sum(0).sort()
        index = index[model.in_features-self.in_features:]
        self.linear.bias.data = model.bias.data.clone()
        for i in range(self.in_features):
            self.index[i] = index[i]
            self.linear.weight.data[:, i] = model.weight.data[:, index[i]]

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondensingConv(nn.Module):
    def __init__(self, model):
        super(CondensingConv, self).__init__()
        self.in_channels = model.conv.in_channels \
                         * model.groups // model.condense_factor
        self.out_channels = model.conv.out_channels
        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=model.conv.kernel_size,
                              padding=model.conv.padding,
                              groups=self.groups,
                              bias=False,
                              stride=model.conv.stride)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        index = 0
        mask = model._mask.mean(-1).mean(-1)
        for i in range(self.groups):
            for j in range(model.conv.in_channels):
                if index < (self.in_channels // self.groups) * (i + 1) \
                         and mask[i, j] == 1:
                    for k in range(self.out_channels // self.groups):
                        idx_i = int(k + i * (self.out_channels // self.groups))
                        idx_j = index % (self.in_channels // self.groups)
                        self.conv.weight.data[idx_i, idx_j, :, :] = \
                            model.conv.weight.data[int(i + k * self.groups), j, :, :]
                        self.norm.weight.data[index] = model.norm.weight.data[j]
                        self.norm.bias.data[index] = model.norm.bias.data[j]
                        self.norm.running_mean[index] = model.norm.running_mean[j]
                        self.norm.running_var[index] = model.norm.running_var[j]
                    self.index[index] = j
                    index += 1

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseLinear(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.5):
        super(CondenseLinear, self).__init__()
        self.in_features = int(in_features*drop_rate)
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer('index', torch.LongTensor(self.in_features))

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondenseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1):
        super(CondenseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=self.groups,
                              bias=False)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


__all__ = ['CondenseNet']


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=args.condense_factor,
                                       dropout_rate=args.dropout_rate)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

def condense_net(nin=3):
    
    class DummyClass: pass
    args = DummyClass()
    #args.stages = [2,4,5,6]
    args.stages = [14,14,14]
    args.growth = [8,16,32] # CIFAR10
    #args.growth = [8,16,32,64,128] # 10258M parans
    #args.growth = [12,20,40,80,160] # 16490M params
    #args.growth = [16,24,48,92,172] # 22070M params
    #args.growth = [20,32,52,100,188] # 28684M params
    #args.growth = [24,40,56,108,196] # 35521 params
    args.bottleneck = 4
    args.condense_factor = 4
    args.dropout_rate = 0
    args.group_1x1 = 4
    args.group_3x3 = 4
    args.nin = nin
    args.output_channels = None
    args.data = 'cifar10'
    args.num_classes = 10
    
    model = CondenseNet(args)
    return model


if __name__ == '__main__':
    net = condense_net(nin=3)
    print(net)

    net.cuda()
    #print(net.extra_loss())
    # from network_stats import measure_model
    # nops, nparams, dt = measure_model(net, 4, 416, 640)
    # print("%s: ops: %.2fM, params: %.2fM fwd dt: %.2fs" % ('CondenseNet', nops/1e6, nparams/1e6, dt))

    x = torch.randn(1,3,32,32).cuda()
    out =net(x)
    # net.cuda()
    net(x, progress=0)
    net(x, progress=0.2)
    net(x, progress=0.4)
    net(x, progress=0.6)
    net(x, progress=0.8)
    net(x, progress=1)
    net(x, progress=1.2)

    """
    progress:0.2 delta:4 count:0 |W|:512 |W=0|:128
    progress:0.2 delta:6 count:0 |W|:768 |W=0|:192
    progress:0.2 delta:8 count:0 |W|:2048 |W=0|:512
    progress:0.2 delta:12 count:0 |W|:3072 |W=0|:768
    progress:0.2 delta:16 count:0 |W|:4096 |W=0|:1024
    progress:0.2 delta:20 count:0 |W|:5120 |W=0|:1280
    progress:0.2 delta:24 count:0 |W|:12288 |W=0|:3072
    progress:0.2 delta:32 count:0 |W|:16384 |W=0|:4096
    progress:0.2 delta:40 count:0 |W|:20480 |W=0|:5120
    progress:0.2 delta:48 count:0 |W|:24576 |W=0|:6144
    progress:0.2 delta:56 count:0 |W|:28672 |W=0|:7168
    progress:0.2 delta:64 count:0 |W|:65536 |W=0|:16384
    progress:0.2 delta:80 count:0 |W|:81920 |W=0|:20480
    progress:0.2 delta:96 count:0 |W|:98304 |W=0|:24576
    progress:0.2 delta:112 count:0 |W|:114688 |W=0|:28672
    progress:0.2 delta:128 count:0 |W|:131072 |W=0|:32768
    progress:0.2 delta:144 count:0 |W|:147456 |W=0|:36864
    progress:0.2 delta:160 count:0 |W|:327680 |W=0|:81920
    progress:0.2 delta:192 count:0 |W|:393216 |W=0|:98304
    progress:0.2 delta:224 count:0 |W|:458752 |W=0|:114688
    progress:0.2 delta:256 count:0 |W|:524288 |W=0|:131072
    progress:0.4 delta:4 count:4 |W|:512 |W=0|:256
    progress:0.4 delta:6 count:6 |W|:768 |W=0|:384
    progress:0.4 delta:8 count:8 |W|:2048 |W=0|:1024
    progress:0.4 delta:12 count:12 |W|:3072 |W=0|:1536
    progress:0.4 delta:16 count:16 |W|:4096 |W=0|:2048
    progress:0.4 delta:20 count:20 |W|:5120 |W=0|:2560
    progress:0.4 delta:24 count:24 |W|:12288 |W=0|:6144
    progress:0.4 delta:32 count:32 |W|:16384 |W=0|:8192
    progress:0.4 delta:40 count:40 |W|:20480 |W=0|:10240
    progress:0.4 delta:48 count:48 |W|:24576 |W=0|:12288
    progress:0.4 delta:56 count:56 |W|:28672 |W=0|:14336
    progress:0.4 delta:64 count:64 |W|:65536 |W=0|:32768
    progress:0.4 delta:80 count:80 |W|:81920 |W=0|:40960
    progress:0.4 delta:96 count:96 |W|:98304 |W=0|:49152
    progress:0.4 delta:112 count:112 |W|:114688 |W=0|:57344
    progress:0.4 delta:128 count:128 |W|:131072 |W=0|:65536
    progress:0.4 delta:144 count:144 |W|:147456 |W=0|:73728
    progress:0.4 delta:160 count:160 |W|:327680 |W=0|:163840
    progress:0.4 delta:192 count:192 |W|:393216 |W=0|:196608
    progress:0.4 delta:224 count:224 |W|:458752 |W=0|:229376
    progress:0.4 delta:256 count:256 |W|:524288 |W=0|:262144
    progress:0.6 delta:4 count:8 |W|:512 |W=0|:384
    progress:0.6 delta:6 count:12 |W|:768 |W=0|:576
    progress:0.6 delta:8 count:16 |W|:2048 |W=0|:1536
    progress:0.6 delta:12 count:24 |W|:3072 |W=0|:2304
    progress:0.6 delta:16 count:32 |W|:4096 |W=0|:3072
    progress:0.6 delta:20 count:40 |W|:5120 |W=0|:3840
    progress:0.6 delta:24 count:48 |W|:12288 |W=0|:9216
    progress:0.6 delta:32 count:64 |W|:16384 |W=0|:12288
    progress:0.6 delta:40 count:80 |W|:20480 |W=0|:15360
    progress:0.6 delta:48 count:96 |W|:24576 |W=0|:18432
    progress:0.6 delta:56 count:112 |W|:28672 |W=0|:21504
    progress:0.6 delta:64 count:128 |W|:65536 |W=0|:49152
    progress:0.6 delta:80 count:160 |W|:81920 |W=0|:61440
    progress:0.6 delta:96 count:192 |W|:98304 |W=0|:73728
    progress:0.6 delta:112 count:224 |W|:114688 |W=0|:86016
    progress:0.6 delta:128 count:256 |W|:131072 |W=0|:98304
    progress:0.6 delta:144 count:288 |W|:147456 |W=0|:110592
    progress:0.6 delta:160 count:320 |W|:327680 |W=0|:245760
    progress:0.6 delta:192 count:384 |W|:393216 |W=0|:294912
    progress:0.6 delta:224 count:448 |W|:458752 |W=0|:344064
    progress:0.6 delta:256 count:512 |W|:524288 |W=0|:393216
    """


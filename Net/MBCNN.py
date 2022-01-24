import x2paddle
from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from Net.MBCNN_class import *
import paddle.nn.functional as F
from Util.util_collections  import *

class My_MBCNN(nn.Layer):

    def __init__(self, nFilters, multi=True):
        super().__init__()
        self.__name__ = "My_MBCNN"
        self.imagesize = 256
        self.sigmoid = nn.Sigmoid()

        self.Depth2space1 = nn.PixelShuffle(2)
        self.conv_func1 = conv_relu1(12, nFilters * 2, 3, padding=1)
        self.pre_block1 = pre_block((1, 2, 3, 2, 1))
        self.conv_func2 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block2 = pre_block((1, 2, 3, 2, 1))


        self.conv_func3 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block3 = pre_block((1, 2, 2, 2, 1))

        self.conv_func4 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block4 = pre_block((1, 2, 2, 2, 1))

        self.global_block1 = global_block(self.imagesize // 8)
        self.pos_block1 = pos_block((1, 2, 2, 2, 1))
        self.conv1 = conv1(128, 12, 3)

        self.conv_func5 = conv_relu1(131, nFilters * 2, 1, padding=0)
        self.global_block2 = global_block(self.imagesize // 4)
        self.pre_block5 = pre_block((1, 2, 3, 2, 1))
        self.global_block3 = global_block(self.imagesize // 4)
        self.pos_block2 = pos_block((1, 2, 3, 2, 1))
        self.conv2 = conv1(128, 12, 3)

        self.conv_func6 = conv_relu1(131, nFilters * 2, 1, padding=0)
        self.global_block4 = global_block(self.imagesize // 2)
        self.pre_block6 = pre_block((1, 2, 3, 2, 1))
        self.global_block5 = global_block(self.imagesize // 2)
        self.pos_block3 = pos_block((1, 2, 3, 2, 1))
        self.conv3 = conv1(128, 12, 3)

        self.conv_func7 = conv_relu1(131, nFilters * 2, 1, padding=0)
        self.global_block6 = global_block(self.imagesize // 2)
        self.pre_block7 = pre_block((1, 2, 3, 2, 1))
        self.global_block7 = global_block(self.imagesize // 2)
        self.pos_block4 = pos_block((1, 2, 3, 2, 1))
        self.conv4 = conv1(128, 12, 3)

    def forward(self, x):
        output_list = []
        shape = list(x.shape)
        batch, channel, height, width = shape

        _x = pixel_unshuffle(x)

        t1 = self.conv_func1(_x)
        t1 = self.pre_block1(t1)
        t2 = F.pad(t1, (1, 1, 1, 1))

        t2 = self.conv_func2(t2)
        t2 = self.pre_block2(t2)
        t3 = F.pad(t2, (1, 1, 1, 1))

        t3 = self.conv_func3(t3)
        t3 = self.pre_block3(t3)
        t4 = F.pad(t3, (1, 1, 1, 1))

        t4 = self.conv_func4(t4)
        t4 = self.pre_block4(t4)
        t4 = self.global_block1(t4)
        t4 = self.pos_block1(t4)
        t4_out = self.conv1(t4)

        t4_out = self.Depth2space1(t4_out)


        output_list.append(t4_out)
        _t3 = torch2paddle.concat([t3, t4_out], dim=-3)
        _t3 = self.conv_func5(_t3)
        _t3 = self.global_block2(_t3)
        _t3 = self.pre_block5(_t3)
        _t3 = self.global_block3(_t3)
        _t3 = self.pos_block2(_t3)
        t3_out = self.conv2(_t3)

        t3_out = self.Depth2space1(t3_out)

        output_list.append(t3_out)
        _t2 = torch2paddle.concat([t3_out, t2], dim=-3)
        _t2 = self.conv_func6(_t2)
        _t2 = self.global_block4(_t2)
        _t2 = self.pre_block6(_t2)
        _t2 = self.global_block5(_t2)
        _t2 = self.pos_block3(_t2)
        t2_out = self.conv3(_t2)

        t2_out = self.Depth2space1(t2_out)
        output_list.append(t2_out)
        _t1 = torch2paddle.concat([t1, t2_out], dim=-3)
        _t1 = self.conv_func7(_t1)
        _t1 = self.global_block6(_t1)
        _t1 = self.pre_block7(_t1)
        _t1 = self.global_block7(_t1)
        _t1 = self.pos_block4(_t1)
        _t1 = self.conv4(_t1)
        y = self.Depth2space1(_t1)
        y = self.sigmoid(y)
        output_list.append(y)
        return output_list


class MoireCNN(nn.Layer):

    def conv_block(self, channels):
        convs = [nn.Conv2D(channels, channels, 3, 1, 1), x2paddle.
            torch2paddle.ReLU(True)] * 5
        return nn.Sequential(*convs)

    def up_conv_block(self, *channels):
        layer_nums = len(channels) - 1
        up_convs = []
        for num in range(layer_nums):
            up_convs += [torch2paddle.Conv2DTranspose(channels[num],
                channels[num + 1], 4, 2, 1), x2paddle.torch2paddle.ReLU(True)]
        up_convs += [nn.Conv2D(32, 3, 3, 1, 1)]
        return nn.Sequential(*up_convs)

    def __init__(self):
        super().__init__()
        self.s11 = nn.Sequential(nn.Conv2D(3, 32, 3, 1, 1), x2paddle.
            torch2paddle.ReLU(True), nn.Conv2D(32, 32, 3, 1, 1))
        self.s12 = self.up_conv_block()
        self.s13 = self.conv_block(32)
        self.s21 = nn.Sequential(nn.Conv2D(32, 32, 3, 2, 1), x2paddle.
            torch2paddle.ReLU(True), nn.Conv2D(32, 64, 3, 1, 1))
        self.s22 = self.up_conv_block(64, 32)
        self.s23 = self.conv_block(64)
        init_conv = [nn.Conv2D(64, 64, 3, 2, 1), x2paddle.torch2paddle.ReLU
            (True), nn.Conv2D(64, 64, 3, 1, 1)]

        self.s31 = nn.Sequential(*init_conv)
        self.s32 = self.up_conv_block(64, 64, 32)
        self.s33 = self.conv_block(64)

        self.s41 = nn.Sequential(*init_conv)
        self.s42 = self.up_conv_block(64, 64, 32, 32)
        self.s43 = self.conv_block(64)

        self.s51 = nn.Sequential(*init_conv)
        self.s52 = self.up_conv_block(64, 64, 32, 32, 32)
        self.s53 = self.conv_block(64)

    def forward(self, x):
        x1 = self.s11(x)
        x2 = self.s21(x1)
        x3 = self.s31(x2)
        x4 = self.s41(x3)
        x5 = self.s51(x4)

        x1 = self.s12(self.s13(x1))
        x2 = self.s22(self.s23(x2))
        x3 = self.s32(self.s33(x3))
        x4 = self.s42(self.s43(x4))
        x5 = self.s52(self.s53(x5))
        x = x1 + x2 + x3 + x4 + x5
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.fill_(0)

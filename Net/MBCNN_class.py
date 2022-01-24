from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import numpy as np
from math import cos
from math import pi
from math import sqrt
import paddle.nn.functional as F

class ScaleLayer2(nn.Layer):

    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = paddle.create_parameter(
                            shape=paddle.to_tensor(paddle.ones((64, 1, 1, 1))).shape, 
                            dtype=str(paddle.to_tensor(paddle.ones((64, 1, 1,1))).numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(paddle.ones((64, 1, 1, 1)))))
        self.it_weights.stop_gradient = False

    def forward(self, inputs):
        y = inputs * self.it_weights
        return self.ReLU(y)

    def compute_output_shape(self, input_shape):
        return input_shape


class Kernel(nn.Layer):

    def __init__(self):
        super().__init__()
        conv_shape = (64, 64, 1, 1)
        kernel = paddle.zeros(conv_shape).cuda()
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        self.kernel = paddle.to_tensor(kernel) ######pytoch is torch.autograd.Variable(kernel)

    def forward(self):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape


class adaptive_implicit_trans(nn.Layer):

    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = ScaleLayer2()
        self.kernel = Kernel()

    def forward(self, inputs):
        self.kernel1 = self.it_weights(self.kernel())
        y = F.conv2d(inputs,self.kernel1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaleLayer(nn.Layer):

    def __init__(self, s):
        super().__init__()
        self.kernel = paddle.create_parameter(shape=paddle.to_tensor(s).
            shape, dtype=str(paddle.to_tensor(s).numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(paddle.
            to_tensor(s)))
        self.kernel.stop_gradient = False

    def forward(self, input):
        y = input * self.kernel
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class conv_relu1(nn.Layer):

    def __init__(self, channel, filters, kernel, padding=1, stride=1):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
            padding=self.padding, stride=self.stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y


class conv_relu_in_block(nn.Layer):

    def __init__(self, channel, filters, kernel, padding=1, stride=1,
        dilation=1):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.relu = nn.ReLU()
        if kernel == 1:
            self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
                padding=0)
        elif self.stride == 2:
            self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
                padding=1, stride=self.stride)
        elif self.dilation == 1:
            self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
                padding=1, dilation=self.dilation)
        elif self.dilation == 2:
            padding = self.dilation
            self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
                padding=padding, dilation=self.dilation)
        elif self.dilation == 3:
            padding = self.dilation
            self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
                padding=padding, dilation=self.dilation)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y


class conv1(nn.Layer):

    def __init__(self, channel, filters, kernel, padding=1, strides=1):
        super().__init__()
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.padding = padding
        self.conv = nn.Conv2D(self.channel, self.filters, self.kernel,
            padding=self.padding, stride=strides)

    def forward(self, x):
        y = self.conv(x)
        return y


class pre_block(nn.Layer):

    def __init__(self, dilation):
        super().__init__()
        self.nFilters = 64
        self.dilation = dilation
        self.conv_relu1 = conv_relu_in_block(128, self.nFilters, 3, padding
            =1, dilation=self.dilation[0])
        self.conv_relu2 = conv_relu_in_block(192, self.nFilters, 3, padding
            =1, dilation=self.dilation[1])
        self.conv_relu3 = conv_relu_in_block(256, self.nFilters, 3, padding
            =1, dilation=self.dilation[2])
        self.conv_relu4 = conv_relu_in_block(320, self.nFilters, 3, padding
            =1, dilation=self.dilation[3])
        self.conv_relu5 = conv_relu_in_block(384, self.nFilters, 3, padding
            =1, dilation=self.dilation[4])
        self.conv1 = conv1(448, self.nFilters, 3, padding=1)
        self.conv2 = conv1(64, self.nFilters * 2, 1, padding=0)
        self.relu = nn.ReLU()
        self.adaptive_implicit_trans1 = adaptive_implicit_trans()
        self.ScaleLayer1 = ScaleLayer(0.1)

    def forward(self, x):
        t = x
        _t = self.conv_relu1(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_relu2(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_relu3(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_relu4(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_relu5(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        t = self.conv1(t)
        t = self.adaptive_implicit_trans1(t)
        t = self.conv2(t)
        t = self.ScaleLayer1(t)
        t = paddle.add(x, t)
        return t


class global_block(nn.Layer):

    def __init__(self, x):
        super().__init__()
        self.size = (x + 2) // 2
        self.avgkernel_size = x
        self.nFilters = 64
        self.conv_func1 = conv_relu_in_block(128, self.nFilters * 4, 3,
            stride=2)
        self.GlobalAveragePooling2D = nn.AdaptiveAvgPool2D(1)
        self.dense1 = nn.Linear(256, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.conv_func2 = conv_relu_in_block(128, self.nFilters * 4, 1,
            padding=0)
        self.conv_func3 = conv_relu_in_block(256, self.nFilters * 2, 1,
            padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        t = F.pad(x, (1, 1, 1, 1))
        t = self.conv_func1(t)
        t = self.GlobalAveragePooling2D(t)
        t = t.squeeze(2)
        t = t.squeeze(2)
        t = self.dense1(t)
        t = self.relu(t)
        t = self.dense2(t)
        t = self.relu(t)
        t = self.dense3(t)
        _t = self.conv_func2(x)
        t = t.unsqueeze(2)
        t = t.unsqueeze(2)
        _t = paddle.multiply(_t, t)
        _t = self.conv_func3(_t)
        return _t


class pos_block(nn.Layer):

    def __init__(self, dilation):
        super().__init__()
        self.nFilters = 64
        self.dilation = dilation
        self.conv_func1 = conv_relu_in_block(128, self.nFilters, 3,
            dilation=self.dilation[0])
        self.conv_func2 = conv_relu_in_block(192, self.nFilters, 3,
            dilation=self.dilation[1])
        self.conv_func3 = conv_relu_in_block(256, self.nFilters, 3,
            dilation=self.dilation[2])
        self.conv_func4 = conv_relu_in_block(320, self.nFilters, 3,
            dilation=self.dilation[3])
        self.conv_func5 = conv_relu_in_block(384, self.nFilters, 3,
            dilation=self.dilation[4])
        self.conv_func_last = conv_relu_in_block(448, self.nFilters * 2, 1,
            padding=0)

    def forward(self, x):
        t = x
        _t = self.conv_func1(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_func2(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_func3(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_func4(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        _t = self.conv_func5(t)
        t = torch2paddle.concat([_t, t], dim=-3)
        t = self.conv_func_last(t)
        return t

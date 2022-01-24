from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from x2paddle import torch2paddle
import logging
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image
from paddle.vision import transforms
import matplotlib.pyplot as plt
import paddle.vision
import paddle

pow_func = []


class L2_LOSS(nn.Layer):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L2_LOSS, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = paddle.add(X, -Y)
        square = paddle.square(diff)
        loss = torch2paddle.sum(square) / X.size(0)
        return loss


class L1_LOSS(nn.Layer):

    def __init__(self):
        super(L1_LOSS, self).__init__()
        self.eps = 1e-06

    def forward(self, Ximage, Ytarget):
        diff = paddle.add(Ximage, -Ytarget)
        abs = paddle.abs(diff)
        loss = torch2paddle.sum(abs) / Ximage.size(0)
        return loss


class L1_Advanced_Sobel_Loss(nn.Layer):

    def __init__(self, device=paddle.set_device('gpu')):
        super().__init__()
        self.device = device
        self.conv_op_x = nn.Conv2D(3, 1, 3, bias_attr=False)
        self.conv_op_y = nn.Conv2D(3, 1, 3, bias_attr=False)
        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0, -1, -2]], [[
            2, 1, 0], [1, 0, -1], [0, -1, -2]], [[2, 1, 0], [1, 0, -1], [0,
            -1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], [[
            0, 1, 2], [-1, 0, 1], [-2, -1, 0]], [[0, 1, 2], [-1, 0, 1], [-2,
            -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        self.conv_op_x.weight.set_value(paddle.to_tensor(sobel_kernel_x))
        self.conv_op_y.weight.set_value(paddle.to_tensor(sobel_kernel_y))
        self.conv_op_x.weight.stop_gradient = True
        self.conv_op_y.weight.stop_gradient = True

    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = paddle.abs(edge_Y_xoutputs) + paddle.abs(
            edge_Y_youtputs)
        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = paddle.abs(edge_Y_x) + paddle.abs(edge_Y_y)
        diff = paddle.add(edge_Youtputs, -edge_Y)
        error = paddle.abs(diff)
        loss = torch2paddle.sum(error) / outputs.size(0)
        return loss


class L1_Sobel_Loss(nn.Layer):

    def __init__(self, device=paddle.set_device('gpu')):
        super(L1_Sobel_Loss, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2D(3, 3, 3, bias_attr=False)
        self.conv_op_y = nn.Conv2D(3, 3, 3, bias_attr=False)
        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[
            1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 0, -1], [2, 0, -2], [1,
            0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[
            1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[1, 2, 1], [0, 0, 0], [-1,
            -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        self.conv_op_x.weight.set_value(paddle.to_tensor(sobel_kernel_x))
        self.conv_op_y.weight.set_value(paddle.to_tensor(sobel_kernel_y))
        self.conv_op_x.weight.stop_gradient = True
        self.conv_op_y.weight.stop_gradient = True

    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = paddle.abs(edge_Y_xoutputs) + paddle.abs(
            edge_Y_youtputs)
        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = paddle.abs(edge_Y_x) + paddle.abs(edge_Y_y)
        diff = paddle.add(edge_Youtputs, -edge_Y)
        error = paddle.abs(diff)
        loss = torch2paddle.sum(error)
        return loss


class edge_making(nn.Layer):

    def __init__(self):
        super(edge_making, self).__init__()
        self.conv_op_x, self.conv_op_y = self.make_sobel_layer()

    def forward(self, output):
        output = output * 2 - 1
        edge_X_x = self.conv_op_x(output)
        edge_X_y = self.conv_op_y(output)
        edge_X = paddle.abs(edge_X_x) + paddle.abs(edge_X_y)
        return edge_X

    def make_sobel_layer(self):
        conv_op_x = nn.Conv2D(3, 3, 3, bias_attr=False)
        conv_op_y = nn.Conv2D(3, 3, 3, bias_attr=False)
        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[
            1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 0, -1], [2, 0, -2], [1,
            0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[
            1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[1, 2, 1], [0, 0, 0], [-1,
            -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        conv_op_x.weight.data = paddle.to_tensor(sobel_kernel_x)
        conv_op_y.weight.data = paddle.to_tensor(sobel_kernel_y)
        conv_op_x.weight.requires_grad = False
        conv_op_y.weight.requires_grad = False
        return conv_op_x, conv_op_y


class Sobelloss_L1(nn.Layer):
    """edge_loss"""

    def __init__(self):
        super().__init__()
        self.eps = 1e-06

    def forward(self, image, target, cuda=True):
        x_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        x_filter3 = np.zeros((3, 3, 3))
        y_filter3 = np.zeros((3, 3, 3))
        x_filter3[:, :, 0] = x_filter
        x_filter3[:, :, 1] = x_filter
        x_filter3[:, :, 2] = x_filter
        y_filter3[:, :, 0] = y_filter
        y_filter3[:, :, 1] = y_filter
        y_filter3[:, :, 2] = y_filter
        convx = nn.Conv2D(3, 3, kernel_size=3, stride=1, padding=1,
            bias_attr=False)
        convy = nn.Conv2D(3, 3, kernel_size=3, stride=1, padding=1,
            bias_attr=False)
        weights_x = paddle.to_tensor(x_filter3).float().unsqueeze(0).unsqueeze(
            0)
        weights_y = paddle.to_tensor(y_filter3).float().unsqueeze(0).unsqueeze(
            0)
        if cuda:
            weights_x = weights_x.cuda()
            weights_y = weights_y.cuda()
        convx.weight = paddle.create_parameter(shape=weights_x.shape, dtype
            =str(weights_x.numpy().dtype), default_initializer=paddle.nn.
            initializer.Assign(weights_x))
        convx.weight.stop_gradient = False
        convy.weight = paddle.create_parameter(shape=weights_y.shape, dtype
            =str(weights_y.numpy().dtype), default_initializer=paddle.nn.
            initializer.Assign(weights_y))
        convy.weight.stop_gradient = False
        convx.weight.requires_grad = False
        convy.weight.requires_grad = False
        g1_x = convx(image)
        g2_x = convx(target)
        g1_y = convy(image)
        g2_y = convy(target)
        g_1 = paddle.sqrt(paddle.pow(g1_x, 2) + paddle.pow(g1_y, 2))
        g_2 = paddle.sqrt(paddle.pow(g2_x, 2) + paddle.pow(g2_y, 2))
        loss = paddle.sqrt((g_1-g_2).pow(2))
        loss = torch2paddle.sum(loss) / image.size(0)
        return loss


class L1_Charbonnier_loss(nn.Layer):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = paddle.add(X, -Y)
        error = paddle.sqrt(diff * diff + self.eps)
        loss = torch2paddle.sum(error) / X.size(0)
        return loss


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch2paddle.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

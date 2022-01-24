from x2paddle import torch2paddle
import argparse
import os
import random
import sys
import PIL
import numpy as np
import paddle
from paddle import nn
import paddle.optimizer as optim
from x2paddle.torch2paddle import DataLoader
from tqdm import tqdm
from paddle.vision import transforms
import matplotlib
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im
from Util.util_collections import save_single_image
from Util.util_collections import PSNR
from Util.util_collections import Time2Str
from dataset.dataset import *
import dataset.Meter as meter
from skimage.metrics import peak_signal_noise_ratio
import paddle.vision
import cv2


# load pretrained model
def load_pretrained_model_v2(model, pretrained_model):
    if pretrained_model is not None:
        # print('Loading pretrained model from {}'.format(pretrained_model))
        print('loading from pretrained model state_dict')

        # if os.path.exists(pretrained_model):
        if pretrained_model is not None:
            # para_state_dict = paddle.load(pretrained_model)
            para_state_dict = pretrained_model

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif para_state_dict[k].shape == [] and model_state_dict[k].shape==[1]:
                    model_state_dict[k] = para_state_dict[k].unsqueeze(0)
                    num_params_loaded += 1
                elif list(para_state_dict[k].shape) != list(
                          model_state_dict[k].shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                    print(para_state_dict[k])
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        print(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def test(args, model):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = paddle.set_device('gpu') if paddle.is_compiled_with_cuda(
        ) else paddle.set_device('cpu')
    args.save_prefix = args.save_prefix + Time2Str() + '_Test_AIM_psnr'
    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)
    print('paddle devices = ', args.device)
    print('save_path = ', args.save_prefix)
    Moiredata_test = My_Moire_dataset_test_mode_1(args.testmode_path, crop=False)

    def yolo_dataset_collate(batch):
        moire = [item[0] for item in batch]
        label = [item[1] for item in batch]
        return moire, label
    test_dataloader = DataLoader(Moiredata_test, batch_size=1, shuffle=False, \
                                    num_workers=args.num_worker, drop_last=False, \
                                    collate_fn=yolo_dataset_collate)
    # model = paddle.DataParallel(model)
    # model = model.to(args.device)
    if args.Test_pretrained_path:
        file_ext = args.Test_pretrained_path.split('.')[-1]
        if file_ext == 'pdiparams':
            # load method 1: 加载pytorch上得到的模型
            # 加载模型的statedict
            checkpoint = paddle.load(args.Train_pretrained_path)
            # pytorch中statedict().keys()比paddle中多了定语"module.""
            # 因此删掉key的前7个元素，重新构造dict，并进行加载
            new_checkpoint_param = {}
            for k in checkpoint.keys():
                list_k = list(k)
                del list_k[:7]
                new_checkpoint_param[''.join(list_k)] = checkpoint[k]
            load_pretrained_model_v2(model, new_checkpoint_param)
        elif file_ext == 'tar':
            # 这个分支暂不支持
            checkpoint = paddle.load(args.Test_pretrained_path)
            model.load_state_dict(checkpoint['model'])
    model.eval()
    image_train_path_demoire = '{0}/{1}'.format(args.save_prefix,'TEST_Demoirefolder')
    if not os.path.exists(image_train_path_demoire):
        os.makedirs(image_train_path_demoire)
    for ii, (val_moires, labels) in tqdm(enumerate(test_dataloader)):
        with paddle.no_grad():
            output1 = []
            output2 = []
            for i in range(len(val_moires)):
                val_moires[i] = val_moires[i].to(args.device)
                output = test_model(model, val_moires[i])
                output_ = my_test_model(model, val_moires[i])
                output1.append(output)
                output2.append(output_)
        val_moires = tensor2im(val_moires)
        bs = len(val_moires)
        for jj in range(bs):
            output, output_, moire, label = output1[jj], output2[jj
                ], val_moires[jj], labels[jj]
            output_0 = tensor2im(output)
            output_ = tensor2im(output_)
            if peak_signal_noise_ratio(output_0, output_) < 18:
                with paddle.no_grad():
                    output = test_model(model, output)
            output_0 = tensor2im(output)
            img_path = os.path.join(image_train_path_demoire, label) + '.jpg'
            save_single_image(output_0, img_path)

# 经过Resize再生成
def my_test_model(model, val_moires):
    shape = val_moires.shape
    # val_moires = val_moires.view(1, shape[0], shape[1], shape[2])
    img_size = 1024
    Resize = transforms.Resize((img_size, img_size))
    moire = Resize(val_moires)
    clear = model(moire.unsqueeze(0))
    clear_img = clear[3].view(shape[0], img_size, img_size)
    Resize = transforms.Compose([transforms.Resize((shape[1], shape[2]))])
    img = Resize(clear_img)
    return img


def test_model_alpha(model, val_moires):
    scale = 256
    shape = list(val_moires.shape)
    val_moires = val_moires.view(shape[0], shape[1], shape[2], shape[3])
    image = paddle.zeros([shape[1], shape[2], shape[3]]).requires_grad_(False)
    x = shape[-2]
    x_i = x // scale
    y = shape[-1]
    y_i = y // scale
    imgs = []
    for i in range(x_i):
        img = []
        for j in range(y_i):
            img.append(val_moires[:, :, i * scale:(i + 1) * scale, j * scale:(j + 1) * scale])
        img = torch2paddle.concat(img, dim=0)
        imgs.append(img)
    if len(imgs) > 1:
        imgs = torch2paddle.concat(imgs, dim=0)
    else:
        imgs = imgs[0]
    output4, output3, output2, output1 = model(imgs)
    for i in range(x_i):
        for j in range(y_i):
            image[:, i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = output1[i * y_i + j]
    for i in range(x_i):
        output4, output3, output2, output1 = model(val_moires[:, :, i * scale:(i + 1) * scale, -scale:])
        image[:, i * scale:(i + 1) * scale, -scale:] = output1[0]
    for j in range(y_i):
        output4, output3, output2, output1 = model(val_moires[:, :, -scale:, j * scale:(j + 1) * scale])
        image[:, -scale:, j * scale:(j + 1) * scale] = output1[0]
    output4, output3, output2, output1 = model(val_moires[:, :, -scale:, -scale:])
    image[:, -scale:, -scale:] = output1[0]
    return image

# 切成patch再生成
def test_model(model, val_moires):
    scale = 256
    shape = list(val_moires.shape)
    val_moires = val_moires.view(1, shape[0], shape[1], shape[2])
    image = paddle.zeros(shape).requires_grad_(False)
    x = shape[-2]
    x_i = x // scale
    y = shape[-1]
    y_i = y // scale
    x_r = x_i - 1
    y_r = y_i - 1
    alpha_x, alpha_moires_x = get_alpha_moire(model, val_moires, x_r, y_i, scale, 1)
    alpha_y, alpha_moires_y = get_alpha_moire(model, val_moires, x_i, y_r, scale, 0)
    alpha_y_x, alpha_moires_y_x = get_alpha_moire(model, val_moires[:, :, :, scale // 2:], \
                                                x_r, y_r, scale, 1)
    alpha_moires_y[:, scale // 2:x_r * scale + scale // 2, :] = \
                        alpha_moires_y[:, scale // 2:x_r * scale + scale // 2, :] * (1 - alpha_y_x) + \
                        alpha_moires_y_x * alpha_y_x
    imgs = []
    for i in range(x_i):
        img = []
        for j in range(y_i):
            img.append(val_moires[:, :, i * scale:(i + 1) * scale, j *
                scale:(j + 1) * scale])
        img = torch2paddle.concat(img, dim=0)
        imgs.append(img)
    imgs = torch2paddle.concat(imgs, dim=0)
    output4, output3, output2, output1 = model(imgs)
    for i in range(x_i):
        for j in range(y_i):
            image[:, i * scale:(i + 1) * scale, j * scale:(j + 1) * scale
                ] = output1[i * y_i + j]
    output4, output3, output2, output1 = model(val_moires[:, :, -scale:, -
        scale:])
    image[:, -scale:, -scale:] = output1[0]
    image[:, scale // 2:x_r * scale + scale // 2, :y_i * scale] = image[:, 
        scale // 2:x_r * scale + scale // 2, :y_i * scale] * (1 - alpha_x
        ) + alpha_moires_x * alpha_x
    image[:, :x_i * scale, scale // 2:y_r * scale + scale // 2] = image[:,
        :x_i * scale, scale // 2:y_r * scale + scale // 2] * (1 - alpha_y
        ) + alpha_moires_y * alpha_y
    for i in range(x_i):
        output4, output3, output2, output1 = model(val_moires[:, :, i *
            scale:(i + 1) * scale, -scale:])
        image[:, i * scale:(i + 1) * scale, -scale:] = output1[0]
    alpha_x_x, alpha_moires_x_x = get_alpha_moire(model, val_moires[:, :, :
        x_i * scale, -scale:], x_r, 1, scale, 1)
    alpha_x_y, alpha_moires_x_y = get_alpha_moire(model, val_moires[:, :, :
        x_i * scale, -(scale * 2):], x_i, 1, scale, 0)
    alpha_x_y_x, alpha_moires_x_y_x = get_alpha_moire(model, val_moires[:,
        :, :x_i * scale, -(scale * 3 // 2):], x_r, 1, scale, 1)
    alpha_moires_x_y[:, scale // 2:x_r * scale + scale // 2, :
        ] = alpha_moires_x_y[:, scale // 2:x_r * scale + scale // 2, :] * (
        1 - alpha_x_y_x) + alpha_moires_x_y_x * alpha_x_y_x
    image[:, scale // 2:scale // 2 + x_r * scale, -scale:] = image[:, scale //
        2:scale // 2 + x_r * scale, -scale:] * (1 - alpha_x_x
        ) + alpha_moires_x_x * alpha_x_x
    image[:, :x_i * scale, -(scale * 3 // 2):-(scale // 2)] = image[:, :x_i *
        scale, -(scale * 3 // 2):-(scale // 2)] * (1 - alpha_x_y
        ) + alpha_moires_x_y * alpha_x_y
    for j in range(y_i):
        output4, output3, output2, output1 = model(val_moires[:, :, -scale:,
            j * scale:(j + 1) * scale])
        image[:, -scale:, j * scale:(j + 1) * scale] = output1[0]
    alpha_y_y, alpha_moires_y_y = get_alpha_moire(model, val_moires[:, :, -
        scale:, :y_i * scale], 1, y_r, scale, 0)
    alpha_y_x, alpha_moires_y_x = get_alpha_moire(model, val_moires[:, :, -
        (scale * 2):, :y_i * scale], 1, y_i, scale, 1)
    alpha_y_x_y, alpha_moires_y_x_y = get_alpha_moire(model, val_moires[:,
        :, -(scale * 3 // 2):, :y_i * scale], 1, y_r, scale, 0)
    alpha_moires_y_x[:, :, scale // 2:scale // 2 + y_r * scale
        ] = alpha_moires_y_x[:, :, scale // 2:scale // 2 + y_r * scale] * (
        1 - alpha_y_x_y) + alpha_moires_y_x_y * alpha_y_x_y
    image[:, -scale:, scale // 2:scale // 2 + y_r * scale] = image[:, -
        scale:, scale // 2:scale // 2 + y_r * scale] * (1 - alpha_y_y
        ) + alpha_moires_y_y * alpha_y_y
    image[:, -(scale * 3 // 2):-(scale // 2), :y_i * scale] = image[:, -(
        scale * 3 // 2):-(scale // 2), :y_i * scale] * (1 - alpha_y_x
        ) + alpha_moires_y_x * alpha_y_x
    return image


def get_alpha_moire(model, moire, x, y, scale, flags):
    alpha = paddle.zeros([x * scale, y * scale]).requires_grad_(False)
    a = get_alpha(scale, flags)
    for i in range(x):
        for j in range(y):
            alpha[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = a
    alpha = alpha.view(1, alpha.shape[0], alpha.shape[1]).repeat(3, 1, 1)
    if flags == 1:
        alpha_moire = moire[:, :, scale // 2:scale // 2 + x * scale, :y * scale
            ]
    if flags == 0:
        alpha_moire = moire[:, :, :x * scale, scale // 2:scale // 2 + y * scale
            ]
    alpha_moire = test_model_alpha(model, alpha_moire)
    return alpha, alpha_moire


def get_alpha(scale, x):
    alpha = paddle.zeros([scale, scale]).requires_grad_(False)
    y = get_weight(scale // 2, 0.05)
    for i in range(scale // 2):
        if x == 0:
            alpha[:, i] = y[i] * paddle.ones([scale]).requires_grad_(False)
            alpha[:, scale - i - 1] = y[i] * paddle.ones([scale]).requires_grad_(False)
        if x == 1:
            alpha[i, :] = y[i] * paddle.ones([scale]).requires_grad_(False)
            alpha[scale - i - 1, :] = y[i] * paddle.ones([scale]).requires_grad_(False)
    return alpha


def get_weight(d, k):
    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y

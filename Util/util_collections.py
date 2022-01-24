from x2paddle import torch2paddle
import os
import numpy as np
import paddle
import paddle.nn as nn
import colour
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from paddle import to_tensor
import math
import time


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150, 200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def Time2Str():
    sec = time.time()
    tm = time.localtime(sec)
    time_str = '21' + '{:02d}'.format(tm.tm_mon) \
               + '{:02d}'.format(tm.tm_mday) \
               + '_' + '{:02d}'.format(tm.tm_hour + 9) \
               + ':{:02d}'.format(tm.tm_min)
    return time_str


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150, 200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def tensor2im(input_image, imtype=np.uint8):
    # print('line 43 in util_collections', type(input_image))
    if isinstance(input_image, paddle.Tensor):
        image_tensor = input_image.detach()
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.swapaxes(image_numpy, -3, -2)
    image_numpy = np.swapaxes(image_numpy, -2, -1)
    return image_numpy


def PSNR(original, contrast):
    original = original * 255.0
    contrast = contrast * 255.0
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def save_single_image(img, img_path):
    if np.shape(img)[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * 255
    cv2.imwrite(img_path, img)


def pixel_unshuffle(batch_input, shuffle_scale=2, device=paddle.set_device('gpu')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2D(1, 1, 2, 2, bias_attr=False)
    conv1.weight.set_value(paddle.to_tensor(np.array([[1, 0], [0, 0]], dtype=\
        'float32').reshape([1, 1, 2, 2])))

    conv2 = nn.Conv2D(1, 1, 2, 2, bias_attr=False)
    conv2.weight.set_value(paddle.to_tensor(np.array([[0, 1], [0, 0]], dtype=\
        'float32').reshape([1, 1, 2, 2]))) 

    conv3 = nn.Conv2D(1, 1, 2, 2, bias_attr=False)
    conv3.weight.set_value(paddle.to_tensor(np.array([[0, 0], [1, 0]], dtype=\
        'float32').reshape([1, 1, 2, 2]))) 

    conv4 = nn.Conv2D(1, 1, 2, 2, bias_attr=False)
    conv4.weight.set_value(paddle.to_tensor(np.array([[0, 0], [0, 1]], dtype=\
        'float32').reshape([1, 1, 2, 2]))) 

    Unshuffle = paddle.ones((batch_size, 4, height // 2, width // 2)
        ).requires_grad_(False)
    for i in range(num_channels):
        each_channel = batch_input[:, i:i + 1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch2paddle.concat((first_channel, second_channel,
            third_channel, fourth_channel), dim=1)
        Unshuffle = torch2paddle.concat((Unshuffle, result), dim=1)
    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle#.detach()


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1 + int(0.15 * w), 1 + int(0.15 * h), int(0.85 * w),
        int(0.85 * h)))
    return region


def calc_pasnr_from_folder(src_path, dst_path):
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split('_')[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, '{}_source.png'.format(label))
        dst = os.path.join(dst_path, '{}_target.png'.format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)
        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr
    psnr /= num_image
    return psnr


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +
        C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
reconstruction_function = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch2paddle.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


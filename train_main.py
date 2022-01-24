import argparse
import logging
import os
import random
import numpy as np
import paddle
from train import *

from Net.MBCNN import *
parser = argparse.ArgumentParser()
parser.add_argument('--traindata_path', type=str, \
                         default=r'/home/aistudio/data/moire_train_dataset_1',
                         help='vit_patches_size, default is 16')
parser.add_argument('--testdata_path', type=str, default=\
                                    '/home/aistudio/data/moire_train_dataset', help=\
                                    'vit_patches_size, default is 16')
parser.add_argument('--testmode_path', type=str, default=\
                                    '/data/下载的压缩包/文献数据集/moire_testA_dataset', help=\
                                    'vit_patches_size, default is 16')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_worker', type=int, default=0, help=\
                                    'number of workers')
parser.add_argument('--batchsize', type=int, default=16, help='mini batch size')
parser.add_argument('--max_epoch', type=int, default=34000, help=\
                                    'number of max_epoch')
parser.add_argument('--save_every', type=int, default=1, help=\
                                    'saving period for pretrained weight ')
parser.add_argument('--name', type=str, default='MBCNN', help=\
                                    'name for this experiment rate')
parser.add_argument('--psnr_axis_min', type=int, default=10, help=\
                                    'mininum line for psnr graph')
parser.add_argument('--psnr_axis_max', type=int, default=70, help=\
                                    'maximum line for psnr graph')
parser.add_argument('--psnrfolder', type=str, default=\
                                    'psnrfoler path was not configured', help=\
                                    'psnrfoler path, define it first!!')
parser.add_argument('--device', type=str, default='cuda or cpu', help=\
                                    'device, define it first!!')   
parser.add_argument('--save_prefix', type=str, default=\
                                    'databse4/jhkim/PTHfolder/', help='saving folder directory')
parser.add_argument('--bestperformance', type=float, default=0.0, help=\
                                    'saving folder directory')
parser.add_argument('--Train_pretrained_path', type=str, default=\
                                    '/home/aistudio/work/epoch36857_psnr38_4493.pdiparams', help='saving folder directory')
parser.add_argument('--Test_pretrained_path', type=str, help=\
                                    'saving folder directory')
args = parser.parse_args()

if __name__ == '__main__':
    net = My_MBCNN(64)
    para = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Model {}:params:{:4f}M'.format(net.__name__, para * 4 / 1000 /
        1000))
    train(args, net)

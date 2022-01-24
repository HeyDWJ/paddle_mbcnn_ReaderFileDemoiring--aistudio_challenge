from x2paddle import torch2paddle
import argparse
import os
import time
import numpy as np
import paddle
from paddle import nn
import paddle.optimizer as optim
from x2paddle.torch2paddle import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im
from Util.util_collections import save_single_image
from Util.util_collections import PSNR
from Util.util_collections import Time2Str
from Net.LossNet import L1_LOSS
from Net.LossNet import L1_Advanced_Sobel_Loss
from dataset.dataset import *
from dataset.Meter import AverageValueMeter
from skimage.metrics import peak_signal_noise_ratio
import math


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

def train(args, model):
    args.device = paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')
    args.save_prefix = args.save_prefix + Time2Str() + '_MBCNN_AIMData_Loss=sum,LR=0.3'
    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)
    print('torch devices = \t\t\t', args.device)
    print('save_path = \t\t\t\t', args.save_prefix)
    args.pthfoler = os.path.join(args.save_prefix, '1pth_folder/')
    args.psnrfolder = os.path.join(args.save_prefix, '1psnr_folder/')
    if not os.path.exists(args.pthfoler):
        os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder):
        os.makedirs(args.psnrfolder)
    Moiredata_train = My_Moire_dataset_RR_2_4(args.traindata_path)
    train_dataloader = DataLoader(Moiredata_train, batch_size=args.batchsize, \
                                    shuffle=True, num_workers=args.num_worker, drop_last=True)
    Moiredata_test = My_Moire_dataset_test_val_1(args.testdata_path, crop=False)

    def yolo_dataset_collate(batch):
        moire = [item[0] for item in batch]
        clear = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return moire, clear, label
    test_dataloader = DataLoader(Moiredata_test, batch_size=2, shuffle=True,
                                    num_workers=args.num_worker, drop_last=False, collate_fn=\
                                    yolo_dataset_collate)
    lr = 1e-8
    last_epoch = 0
    optimizer = torch2paddle.Adam(params=model.parameters(), lr=lr)
    list_psnr_output = []
    list_loss_output = []
    # model = paddle.DataParallel(model)
    # model = model.cuda()
    if args.Train_pretrained_path:
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

        # load method 2: 加载paddle上得到的模型
        # checkpoint = paddle.load(args.Train_pretrained_path)
        # model.load_state_dict(checkpoint['model'])
        # last_epoch = checkpoint['epoch']
        # optimizer_state = checkpoint['optimizer']
        # optimizer.set_state_dict(optimizer_state)
        # lr = checkpoint['lr']
        # list_psnr_output = checkpoint['list_psnr_output']
        # list_loss_output = checkpoint['list_loss_output']
    model.train()
    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()
    lr = 4e-4

    psnr_meter = AverageValueMeter()
    Loss_meter1 = AverageValueMeter()
    Loss_meter2 = AverageValueMeter()
    Loss_meter3 = AverageValueMeter()
    Loss_meter4 = AverageValueMeter()

    for epoch in range(args.max_epoch):
        print('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()
        if epoch < last_epoch:
            continue
        Loss_meter1.reset()
        Loss_meter2.reset()
        Loss_meter3.reset()
        Loss_meter4.reset()
        psnr_meter.reset()
        for ii, (moires, clears_list, labels) in tqdm(enumerate(train_dataloader)):
            moires = moires.cuda()
            clear4, clear3, clear2, clear1 = clears_list
            clear4 = clear4.to(args.device)
            clear3 = clear3.to(args.device)
            clear2 = clear2.to(args.device)
            clear1 = clear1.to(args.device)
            output4, output3, output2, output1 = model(moires)
            Loss_l1 = criterion_l1(output1, clear1)

            Loss_advanced_sobel_l1 = criterion_advanced_sobel_l1(output1, clear1)

            Loss_l12 = criterion_l1(output2, clear2)

            Loss_advanced_sobel_l12 = criterion_advanced_sobel_l1(output2, clear2)

            Loss_l13 = criterion_l1(output3, clear3)

            Loss_advanced_sobel_l13 = criterion_advanced_sobel_l1(output3, clear3)

            Loss_l14 = criterion_l1(output4, clear4)

            Loss_advanced_sobel_l14 = criterion_advanced_sobel_l1(output4, clear4)

            Loss1 = Loss_l1 + 0.25 * Loss_advanced_sobel_l1
            Loss2 = Loss_l12 + 0.25 * Loss_advanced_sobel_l12
            Loss3 = Loss_l13 + 0.25 * Loss_advanced_sobel_l13
            Loss4 = Loss_l14 + 0.25 * Loss_advanced_sobel_l14

            loss = Loss1 + Loss2 + Loss3 + Loss4

            loss_check1 = Loss1
            loss_check2 = Loss_l1
            loss_check3 = 0.25 * Loss_advanced_sobel_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            moires = tensor2im(moires)
            output1 = tensor2im(output1)
            clear1 = tensor2im(clear1)

            psnr = peak_signal_noise_ratio(output1, clear1)
            psnr_meter.add(psnr)
            Loss_meter1.add(loss.item())
            Loss_meter2.add(loss_check1.item())
            Loss_meter3.add(loss_check2.item())
            Loss_meter4.add(loss_check3.item())

        print('training set : \tPSNR = {:f}\t loss = {:f}\t Loss1(scale) = {:f} \t Loss_L1 = {:f} + Loss_sobel = {:f},\t '
            .format(psnr_meter.value()[0], Loss_meter1.value()[0],
            Loss_meter2.value()[0], Loss_meter3.value()[0], Loss_meter4.
            value()[0]))
        (psnr_output, loss_output1, loss_output2, loss_output3, loss_output4
            ) = my_val(model, test_dataloader, epoch, args)
        print(
            'Test set : \tloss = {:0.4f} \t Loss_1 = {:0.4f} \t Loss_L1 = {:0.4f} \t Loss_ASL = {:0.4f}'
            .format(loss_output1, loss_output2, loss_output3, loss_output4))
        print('Test set : \t' + '\x1b[30m \x1b[43m' + 'PSNR = {:0.4f}'.
            format(psnr_output) + '\x1b[0m' + '\tbest PSNR ={:0.4f}'.format
            (args.bestperformance))
        list_psnr_output.append(round(psnr_output, 5))
        list_loss_output.append(round(loss_output1, 5))
        if psnr_output > args.bestperformance:
            args.bestperformance = psnr_output
            file_name = (args.pthfoler +
                'Best_performance_{:}_statedict_epoch{:03d}_psnr{:}.pdiparams'
                .format(args.name, epoch + 1, round(psnr_output, 4)))
            paddle.save(model.state_dict(), file_name)
            print('\x1b[30m \x1b[42m' + 'PSNR WAS UPDATED! ' + '\x1b[0m')
        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            file_name = (args.pthfoler +
                'Best_performance_{:}_ckpt_epoch{:03d}_psnr_{:0.4f}_.tar'.
                format(args.name, epoch + 1, round(psnr_output, 4)))
            checkpoint = {'epoch': epoch + 1, 'optimizer': optimizer.
                state_dict(), 'model': model.state_dict(), 'lr': lr,
                'list_psnr_output': list_psnr_output, 'list_loss_output':
                list_loss_output}
            paddle.save(checkpoint, file_name)
            with open(args.save_prefix +
                '/1_PSNR_validation_set_output_psnr.txt', 'w') as f:
                f.write('psnr_output: {:}\n'.format(list_psnr_output))
            with open(args.save_prefix +
                '/1_Loss_validation_set_output_loss.txt', 'w') as f:
                f.write('loss_output: {:}\n'.format(list_loss_output))
            if epoch >= 1:
                plt.figure()
                plt.plot(range(1, epoch + 2, 1), list_psnr_output, 'r',
                    label='Validation_set')
                plt.xlabel('Epochs')
                plt.ylabel('PSNR')
                plt.axis([1, epoch + 1, args.psnr_axis_min, args.psnr_axis_max]
                    )
                plt.title('PSNR per epoch')
                plt.grid(linestyle='--', color='lavender')
                plt.legend(loc='lower right')
                plt.savefig(args.psnrfolder +
                    'PSNR_graph_{name}_{epoch}.png'.format(name=args.name,
                    epoch=epoch + 1))
                plt.close('all')
        if epoch == args.max_epoch - 1:
            file_name2 = args.pthfoler + '{0}_stdc_epoch{1}.pdiparams'.format(
                args.name, epoch + 1)
            paddle.save(model.state_dict(), file_name2)
        print('1 epoch spends:{:.2f}sec\t remain {:2d}:{:2d} hours'.format(
            time.time() - start, int((args.max_epoch - epoch) * (time.time(
            ) - start) // 3600), int((args.max_epoch - epoch) * (time.time(
            ) - start) % 3600 / 60)))
    return 'Training Finished!'


def val(model, dataloader, epoch, args):
    model.eval()
    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()

    psnr_output_meter = AverageValueMeter()

    loss_meter1 = AverageValueMeter()
    loss_meter2 = AverageValueMeter()
    loss_meter3 = AverageValueMeter()
    loss_meter4 = AverageValueMeter()

    psnr_output_meter.reset()
    loss_meter1.reset()
    loss_meter2.reset()
    loss_meter3.reset()
    loss_meter4.reset()
    image_train_path_demoire = '{0}/epoch_{1}_validation_set_{2}/'.format(args
        .save_prefix, epoch + 1, 'demoire')

    if not os.path.exists(image_train_path_demoire) and (epoch + 1
        ) % args.save_every == 0:

        os.makedirs(image_train_path_demoire)
    image_train_path_moire = '{0}/epoch_{1}_validation_set_{2}/'.format(args
        .save_prefix, 1, 'moire')

    image_train_path_clean = '{0}/epoch_{1}_validation_set_{2}/'.format(args
        .save_prefix, 1, 'clean')

    if not os.path.exists(image_train_path_moire):
        os.makedirs(image_train_path_moire)

    if not os.path.exists(image_train_path_clean):
        os.makedirs(image_train_path_clean)
    for ii, (val_moires, val_clears_list, labels) in tqdm(enumerate(dataloader)
        ):
        with paddle.no_grad():
            val_moires = val_moires.to(args.device)
            clear4, clear3, clear2, clear1 = val_clears_list

            clear4 = clear4.to(args.device)
            clear3 = clear3.to(args.device)
            clear2 = clear2.to(args.device)
            clear1 = clear1.to(args.device)

            output4, output3, output2, output1 = model(val_moires)
        loss_l1 = criterion_l1(output1, clear1)
        loss_advanced_sobel_l1 = criterion_advanced_sobel_l1(output1, clear1)
        Loss_l12 = criterion_l1(output2, clear2)
        Loss_advanced_sobel_l12 = criterion_advanced_sobel_l1(output2, clear2)
        Loss_l13 = criterion_l1(output3, clear3)
        Loss_advanced_sobel_l13 = criterion_advanced_sobel_l1(output3, clear3)
        Loss_l14 = criterion_l1(output4, clear4)
        Loss_advanced_sobel_l14 = criterion_advanced_sobel_l1(output4, clear4)

        Loss1 = loss_l1 + 0.25 * loss_advanced_sobel_l1
        Loss2 = Loss_l12 + 0.25 * Loss_advanced_sobel_l12
        Loss3 = Loss_l13 + 0.25 * Loss_advanced_sobel_l13
        Loss4 = Loss_l14 + 0.25 * Loss_advanced_sobel_l14

        loss = Loss1 + Loss2 + Loss3 + Loss4

        loss_meter1.add(loss.item())
        loss_meter2.add(Loss1.item())
        loss_meter3.add(loss_l1.item())
        loss_meter4.add(loss_advanced_sobel_l1.item())

        val_moires = tensor2im(val_moires)
        output1 = tensor2im(output1)
        clear1 = tensor2im(clear1)

        bs = val_moires.shape[0]
        if epoch != -1:
            for jj in range(bs):
                output, clear, moire, label = output1[jj], clear1[jj
                    ], val_moires[jj], labels[jj]

                psnr_output_individual = peak_signal_noise_ratio(output, clear)
                psnr_output_meter.add(psnr_output_individual)
                psnr_input_individual = peak_signal_noise_ratio(moire, clear)

                if (epoch + 1) % args.save_every == 0 or epoch == 0:
                    img_path = (
                        '{0}/{1}_epoch:{2:04d}_demoire_PSNR:{3:.4f}_demoire.png'
                        .format(image_train_path_demoire, label, epoch + 1,
                        psnr_output_individual))
                    save_single_image(output, img_path)

                if epoch == 0:
                    psnr_in_gt = peak_signal_noise_ratio(moire, clear)
                    img_path2 = '{0}/{1}_moire_{2:.4f}_moire.png'.format(
                        image_train_path_moire, label, psnr_in_gt)
                    img_path3 = '{0}/{1}_clean_.png'.format(
                        image_train_path_clean, label)
                    save_single_image(moire, img_path2)
                    save_single_image(clear, img_path3)

    return psnr_output_meter.value()[0], loss_meter1.value()[0
        ], loss_meter2.value()[0], loss_meter3.value()[0], loss_meter4.value()[
        0]


def my_val(model, dataloader, epoch, args):
    model.eval()
    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()
    psnr_output_meter = AverageValueMeter()
    loss_meter1 = AverageValueMeter()
    loss_meter2 = AverageValueMeter()
    loss_meter3 = AverageValueMeter()
    loss_meter4 = AverageValueMeter()
    psnr_output_meter.reset()
    loss_meter1.reset()
    loss_meter2.reset()
    loss_meter3.reset()
    loss_meter4.reset()
    image_train_path_demoire = '{0}/epoch_{1}_validation_set_{2}/'.format(args.save_prefix, epoch + 1, 'demoire')
    if not os.path.exists(image_train_path_demoire) and (epoch + 1) % args.save_every == 0:
        os.makedirs(image_train_path_demoire)
    image_train_path_moire = '{0}/epoch_{1}_validation_set_{2}/'.format(args.save_prefix, 1, 'moire')
    image_train_path_clean = '{0}/epoch_{1}_validation_set_{2}/'.format(args.save_prefix, 1, 'clean')
    if not os.path.exists(image_train_path_moire):
        os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean):
        os.makedirs(image_train_path_clean)
    for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
        with paddle.no_grad():
            output1 = []
            loss_l1 = 0
            loss_advanced_sobel_l1 = 0
            for i in range(len(val_moires)):
                val_moires[i] = val_moires[i].to(args.device)
                val_clears[i] = val_clears[i].to(args.device)
                output = test_model(model, val_moires[i])
                output1.append(output)
                loss_l1 += criterion_l1(output, val_clears[i])
        Loss1 = loss_l1
        loss = Loss1
        loss_meter1.add(loss.item())
        loss_meter2.add(Loss1.item())
        loss_meter3.add(loss_l1.item())
        loss_meter4.add(loss.item())
        val_moires = tensor2im(val_moires)
        output1 = tensor2im(output1)
        val_clears = tensor2im(val_clears)
        bs = len(val_moires)
        if epoch != -1:
            for jj in range(bs):
                output, clear, moire, label = output1[jj], val_clears[jj], val_moires[jj], labels[jj]
                output = tensor2im(output)
                clear = tensor2im(clear)
                moire = tensor2im(moire)
                psnr_output_individual = peak_signal_noise_ratio(output, clear)
                psnr_output_meter.add(psnr_output_individual)
                psnr_input_individual = peak_signal_noise_ratio(moire, clear)
                if (epoch + 1) % args.save_every == 0 or epoch == 0:
                    img_path = (
                        '{0}/{1}_epoch:{2:04d}_demoire_PSNR:{3:.4f}_demoire.png'
                        .format(image_train_path_demoire, label, epoch + 1,
                        psnr_output_individual))
                    save_single_image(output, img_path)
                if epoch == 0:
                    psnr_in_gt = peak_signal_noise_ratio(moire, clear)
                    img_path2 = '{0}/{1}_moire_{2:.4f}_moire.png'.format(
                        image_train_path_moire, label, psnr_in_gt)
                    img_path3 = '{0}/{1}_clean_.png'.format(
                        image_train_path_clean, label)
                    save_single_image(moire, img_path2)
                    save_single_image(clear, img_path3)
    return psnr_output_meter.value()[0], loss_meter1.value()[0
        ], loss_meter2.value()[0], loss_meter3.value()[0], loss_meter4.value()[0]


def test_model(model, val_moires):
    scale = 256
    shape = list(val_moires.shape)
    val_moires = val_moires.view(1, shape[0], shape[1], shape[2])
    image = paddle.zeros(shape).requires_grad_(False)
    x = shape[-2]
    x_i = x // scale
    y = shape[-1]
    y_i = y // scale
    for i in range(x_i):
        imgs = []
        for j in range(y_i):
            imgs.append(val_moires[:, :, i * scale:(i + 1) * scale, j *scale:(j + 1) * scale])
        imgs = torch2paddle.concat(imgs, dim=0)
        output4, output3, output2, output1 = model(imgs)
        for j in range(y_i):
            image[:, i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = output1[j]
        output4, output3, output2, output1 = model(val_moires[:, :, i *scale:(i + 1) * scale, -scale:])
        image[:, i * scale:(i + 1) * scale, -scale:] = output1[0]
    imgs = []
    for j in range(y_i):
        imgs.append(val_moires[:, :, -scale:, j * scale:(j + 1) * scale])
    imgs = torch2paddle.concat(imgs, dim=0)
    output4, output3, output2, output1 = model(imgs)
    for j in range(y_i):
        image[:, -scale:, j * scale:(j + 1) * scale] = output1[j]
    output4, output3, output2, output1 = model(val_moires[:, :, -scale:, -scale:])
    image[:, -scale:, -scale:] = output1[0]
    return image

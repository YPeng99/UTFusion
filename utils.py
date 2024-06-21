import argparse
import logging
import time
from datetime import timedelta

import torch
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='SwitchFusion-d6-lw120-nm', help='model name')
    parser.add_argument('--use_checkpoint', type=str, default='False', choices=['True', 'False'], help='use checkpoint')
    parser.add_argument('--restart', type=str, default='True', choices=['True', 'False'], help='restart')
    parser.add_argument('--batch_size', type=int, default=96, help='batch Size during training')
    parser.add_argument('--epoch', default=3000, type=int, help='epoch to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--warmup_ratio', default=0.01, type=float, help='initial warmup ratio')
    parser.add_argument('--device', type=str, default='cuda', help='specify devices')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log path')
    parser.add_argument('--train_data_dir', type=str, default='/home/data/SwitchFusion/train', help='data path')
    parser.add_argument('--test_data_dir', type=str, default='/home/data/SwitchFusion/test', help='data path')
    parser.add_argument('--tf_logs_dir', type=str, default='/home/mac-of-ypeng/tf-logs/SwitchFusion', help='tf_logs path')

    return parser.parse_args()


def get_logger(args):
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler_1 = logging.FileHandler(f'{args.log_dir}/{args.model}.log')
    file_handler_1.setLevel(logging.INFO)
    file_handler_1.setFormatter(formatter)
    file_handler_2 = logging.StreamHandler()
    file_handler_2.setLevel(logging.INFO)
    logger.addHandler(file_handler_1)
    logger.addHandler(file_handler_2)
    return logger


def denormalizer(channel_mean=[0.485, 0.456, 0.406], channel_std=[0.229, 0.224, 0.225]):
    '''去归一化'''
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    return transforms.Normalize(mean=MEAN, std=STD)


def get_time_dif(start_time):
    '''获取已使用时间'''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total : {total_num / 10 ** 6}M, Trainable : {trainable_num / 10 ** 6}M')
    # return {'Total': total_num, 'Trainable': trainable_num}


def dice_loss(predicted, target, smooth=1e-6):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()

    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)

    loss = 1.0 - dice_coefficient
    return loss

def rgb2ycbcr(img):
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    return Y[:,None,...], Cb[:,None,...], Cr[:,None,...]


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    return torch.cat([R, G, B], 1)


def fuse_cb_cr(Cb1, Cr1, Cb2, Cr2, tao=128, eps=1e-12):
    Cb = (Cb1 * torch.abs(Cb1 - tao) + Cb2 * torch.abs(Cb2 - tao)) / (torch.abs(Cb1 - tao) + torch.abs(Cb2 - tao) + eps)
    Cr = (Cr1 * torch.abs(Cr1 - tao) + Cr2 * torch.abs(Cr2 - tao)) / (torch.abs(Cr1 - tao) + torch.abs(Cb2 - tao) + eps)
    return Cb, Cr

def fuse_seq_cb_cr(Cb,Cr,tao=0.5, eps=1e-12):
    Cb_n = 0.0
    Cb_d = 0.0
    Cr_n = 0.0
    Cr_d = 0.0
    for b,r in zip(Cb,Cr):
        Cb_n += b * torch.abs(b-tao)
        Cr_n += r * torch.abs(r-tao)

        Cb_d += torch.abs(b-tao)
        Cr_d += torch.abs(r-tao)

    return Cb_n/(Cb_d+eps), Cr_n/(Cr_d+eps)
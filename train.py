import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import WarmupLinearSchedule
from torchvision import transforms
from data_loader import TrainDataset

from models.MFFLoss_Y import fusion_loss_mff
from models.UTFusion import UTFusion
from utils import get_logger, parse_args, get_time_dif, seed_everything


def train(args, model, train_dataloader, logger, checkpoint):
    now = time.strftime('%Y-%m-%d|%H:%M:%S', time.localtime())
    tf_logs_dir = os.path.join(args.tf_logs_dir, args.model, now)

    start_time = time.time()
    no_decay = ['norm.bias', 'norm.weight', 'bias']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.05, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.lr},
    ]

    optimizer = AdamW(optimizer_parameters)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.epoch * args.warmup_ratio,
                                     t_total=args.epoch)

    start = 0
    loss = torch.tensor(0.0)
    best_loss = torch.inf
    mff_loss = fusion_loss_mff().to(args.device)

    if checkpoint:
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        random.setstate(checkpoint['py_state'])
        np.random.set_state(checkpoint['np_state'])
        torch.set_rng_state(checkpoint['torch_cpu_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_gpu_state'])
        best_loss = checkpoint['best_loss']
        tf_logs_dir = checkpoint['tf_logs_dir']

    writer = SummaryWriter(tf_logs_dir)
    for epoch in range(start, args.epoch):
        model.train()

        total_loss_epoch = torch.tensor(0.0)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'[{epoch + 1}/{args.epoch}]: Learning rate: {lr:>5.8f}')
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, ncols=120)
        bar.set_description(f'[{epoch + 1}/{args.epoch}]')
        bar.set_postfix({'loss': f'{loss.item():>5.4f}', })

        for i, (S1, S2, fuse_scheme) in bar:
            S1 = S1.to(args.device, non_blocking=True)
            S2 = S2.to(args.device, non_blocking=True)
            fuse_scheme = fuse_scheme.to(args.device, non_blocking=True)

            fused = model((S1, S2), fuse_scheme)

            loss = mff_loss(S1, S2, fused, fuse_scheme)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_postfix({'loss': f'{loss.item():>5.4f}', })
            total_loss_epoch += loss.item()

        scheduler.step()
        total_loss_epoch = total_loss_epoch / len(train_dataloader)

        improve = ''
        if total_loss_epoch <= best_loss:
            best_loss = total_loss_epoch
            improve = '*'
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'py_state': random.getstate(),
                'np_state': np.random.get_state(),
                'torch_cpu_state': torch.get_rng_state(),
                'torch_gpu_state': torch.cuda.get_rng_state_all(),
                'best_loss': best_loss,
                'tf_logs_dir': tf_logs_dir
            }
            torch.save(state, args.log_dir + f'/{args.model}.ckpt')

        logger.info(f'[{epoch + 1}/{args.epoch}]: loss: {total_loss_epoch:>5.4f} {improve}')
        logger.info(f'[{epoch + 1}/{args.epoch}]: best_loss: {best_loss:>5.4f}')

        writer.add_scalar('loss/loss', total_loss_epoch, epoch)

        time_dif = get_time_dif(start_time)
        logger.info(f'time usage: {time_dif}')

    writer.close()


if __name__ == '__main__':
    # exit(0)
    # 固定随机种子
    seed_everything(415)

    # args预处理
    args = parse_args()
    args.use_checkpoint = True if args.use_checkpoint == 'True' else False
    args.restart = True if args.restart == 'True' else False

    # 日志
    logger = get_logger(args)
    logger.info('PARAMETER ...')
    logger.info(args)

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5,
                                  prefetch_factor=5, pin_memory=True, persistent_workers=True, drop_last=False)

    model = UTFusion(use_checkpoint=args.use_checkpoint)
    model.to(args.device)

    checkpoint = torch.load(os.path.join(args.log_dir, args.model + '.ckpt')) if args.restart else None
    try:
        train(args, model, train_dataloader, logger, checkpoint)
    except Exception as e:
        logger.error(e)
    finally:
        # os.system("/usr/bin/shutdown")
        pass

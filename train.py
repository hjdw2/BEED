import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os, sys, random
import shutil
import argparse
import time
import logging
import math

from resnet import *
from data import *

#import torchvision.models.utils as utils
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 training')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--mode', choices=['CE', 'KD', 'EED', 'BEED'], default='BEED')
    parser.add_argument('--data-dir', default='data', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--arch', metavar='ARCH', default='resnet',
                        help='model architecture')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')
    #distillation parameter
    parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=1.15, type=float, help='balancing coef for CE and KD')
    parser.add_argument('--beta', default=1.6, type=float, help='importance coef for ensemble teacher')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(args.arch, args.resume))
        run_test(args)

def run_test(args):
    model = ResNet()
    model = model.cuda()

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
            exit()

    test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()
    validate(args, test_loader, model, criterion)

def run_training(args):
    model = ResNet()
    model = model.cuda()
    best_prec = 0

    logging.info("=> Training Mode `{}`".format(args.mode))
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    train_loader = prepare_cifar100_train_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    MSEloss = nn.MSELoss(reduction='mean').cuda()

    end = time.time()
    model.train()
    A = args.alpha # loss balancing coef
    B = args.beta # importance coef for ensemble
    e_sum = args.beta**0 + args.beta**1 + args.beta**2 + args.beta**3
    for current_epoch in range(args.start_epoch, args.epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        total_losses = AverageMeter()
        Acc1 = AverageMeter()
        Acc2 = AverageMeter()
        Acc3 = AverageMeter()
        Acc4 = AverageMeter()

        adjust_learning_rate(args, optimizer, current_epoch)
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            target = target.squeeze().long().cuda()
            input = Variable(input).cuda()

            output1, output2, output3, output4, \
            feature1, feature2, feature3, feature4 = model(input)

            CE1 = criterion(output1, target)
            CE2 = criterion(output2, target)
            CE3 = criterion(output3, target)
            CE4 = criterion(output4, target)
            losses.update(CE4.item(), input.size(0))

            if args.mode == 'CE':
                total_loss = CE1 + CE2 + CE3 + CE4

            elif args.mode == 'KD':
                temp = output4 / args.temperature
                temp = torch.softmax(temp, dim=1)

                KD1 = kd_loss_function(output1, temp.detach(), args) * (args.temperature**2) / input.size()[0]
                KD2 = kd_loss_function(output2, temp.detach(), args) * (args.temperature**2) / input.size()[0]
                KD3 = kd_loss_function(output3, temp.detach(), args) * (args.temperature**2) / input.size()[0]

                FD1 = MSEloss(feature1, feature4.detach())
                FD2 = MSEloss(feature2, feature4.detach())
                FD3 = MSEloss(feature3, feature4.detach())

                total_loss = (CE1 + CE2 + CE3 + CE4) \
                     + 0.1 * (KD1 + KD2 + KD3) \
                     + 0.025 * (FD1 + FD2 + FD3)

            elif args.mode == 'EED' or 'BEED':
                output_e =  (B**0*output1/e_sum + B**1*output2/e_sum + B**2*output3/e_sum + B**3*output4/e_sum).detach()
                feature_e = (B**0*feature1/e_sum + B**1*feature2/e_sum + B**2*feature3/e_sum + B**3*feature4/e_sum).detach()

                temp = output_e / args.temperature
                temp = torch.softmax(temp, dim=1)

                KD1 = kd_loss_function(output1, temp.detach(), args) * (args.temperature**2) / input.size()[0]
                KD2 = kd_loss_function(output2, temp.detach(), args) * (args.temperature**2) / input.size()[0]
                KD3 = kd_loss_function(output3, temp.detach(), args) * (args.temperature**2) / input.size()[0]
                KD4 = kd_loss_function(output4, temp.detach(), args) * (args.temperature**2) / input.size()[0]

                FD1 = MSEloss(feature1, feature_e.detach())
                FD2 = MSEloss(feature2, feature_e.detach())
                FD3 = MSEloss(feature3, feature_e.detach())
                FD4 = MSEloss(feature4, feature_e.detach())

                if args.mode == 'EED':
                    total_loss = (CE1 + CE2 + CE3 + CE4) \
                               + (KD1 + KD1 + KD1 + KD4) \
                               + 0.1 * (FD1 + FD2 + FD3 + FD4)
                else:
                    total_loss = ((2-A**3) * CE1 + A**3 * KD1) \
                               + ((2-A**2) * CE2 + A**2 * KD2) \
                               + ((2-A**1) * CE3 + A**1 * KD3) \
                               + ((2-A**0) * CE4 + A**0 * KD4) \
                               + 0.1 * (FD1 + FD2 + FD3 + FD4)

            total_losses.update(total_loss.item(), input.size(0))

            prec1 = accuracy(output1.data, target, topk=(1,))
            Acc1.update(prec1[0], input.size(0))
            prec2 = accuracy(output2.data, target, topk=(1,))
            Acc2.update(prec2[0], input.size(0))
            prec3 = accuracy(output3.data, target, topk=(1,))
            Acc3.update(prec3[0], input.size(0))
            prec4 = accuracy(output4.data, target, topk=(1,))
            Acc4.update(prec4[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {Acc4.val:.3f} ({Acc4.avg:.3f})\t".format(
                                current_epoch,
                                i,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=total_losses,
                                Acc4=Acc4)
                )

        prec = validate(args, test_loader, model, criterion, current_epoch)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print("best: ", best_prec)
        if is_best:
            checkpoint_path = os.path.join(args.save_path, 'model_best.path.tar'.format(current_epoch))
            save_checkpoint({
                'epoch': current_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec,
                }, filename=checkpoint_path)
        torch.cuda.empty_cache()

def validate(args, test_loader, model, criterion, current_epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    Acc1 = AverageMeter()
    Acc2 = AverageMeter()
    Acc3 = AverageMeter()
    Acc4 = AverageMeter()
    Acce = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        target = target.squeeze().long().cuda()
        input = Variable(input).cuda()

        output1, output2, output3, output4, \
        feature1, feature2, feature3, feature4 = model(input)

        CE4 = criterion(output4, target)
        losses.update(CE4.item(), input.size(0))

        prec1 = accuracy(output1.data, target, topk=(1,))
        Acc1.update(prec1[0], input.size(0))
        prec2 = accuracy(output2.data, target, topk=(1,))
        Acc2.update(prec2[0], input.size(0))
        prec3 = accuracy(output3.data, target, topk=(1,))
        Acc3.update(prec3[0], input.size(0))
        prec4 = accuracy(output4.data, target, topk=(1,))
        Acc4.update(prec4[0], input.size(0))
        prece = accuracy((output1+output2+output3+output4).data, target, topk=(1,))
        Acce.update(prece[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    logging.info("Loss {loss.avg:.3f}\t"
                 "Exit1@1 {Acc1.avg:.3f}\t"
                 "Exit2@1 {Acc2.avg:.3f}\t"
                 "Exit3@1 {Acc3.avg:.3f}\t"
                 "Main@1 {Acc4.avg:.3f}\t"
                 "Ens@1 {Acce.avg:.3f}\t".format(
                    loss=losses,
                    Acc1=Acc1,
                    Acc2=Acc2,
                    Acc3=Acc3,
                    Acc4=Acc4,
                    Acce=Acce))

    model.train()
    return Acc4.avg

def kd_loss_function(output, target_output,args):
    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = F.kl_div(output_log_softmax, target_output, reduction='sum')
    return loss_kd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    if 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
    elif epoch >=180:
        lr = args.lr * (args.step_ratio ** 3)
    else:
        lr = args.lr

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def save_checkpoint(state, filename):
    torch.save(state, filename)

if __name__ == '__main__':
    main()

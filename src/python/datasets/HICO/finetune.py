"""
Created on Feb 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import argparse
import time

import numpy as np
import torch.utils.data
import torch.autograd
import torch
import torchvision

import metadata
import hico_config
import roi_feature_model


def main(args):
    best_prec1 = 0.0
    args.distributed = args.world_size > 1
    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size)

    # create model
    if args.feature_type == 'vgg':
        model = roi_feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif args.feature_type == 'resnet':
        model = roi_feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif args.feature_type == 'densenet':
        model = roi_feature_model.Densenet(num_classes=len(metadata.action_classes))
    input_imsize = (224, 224)

    if not args.distributed:
        if args.feature_type.startswith('alexnet') or args.feature_type.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(os.path.join(args.resume, 'model_best.pth')):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, 'model_best.pth'))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(args.resume, 'model_best.pth')))

    torch.backends.cudnn.benchmark = True

    # Data loading code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    train_dataset = roi_feature_model.HICO(args.data, input_imsize, transform, 'train')
    #val_dataset = roi_feature_model.HICO(args.data, input_imsize, transform, 'val')
    test_dataset = roi_feature_model.HICO(args.data, input_imsize, transform, 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch == 0 or epoch >= 5:
            # evaluate on validation set
            prec1 = validate(test_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print('Best precision: {:.03f}'.format(best_prec1))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.feature_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    test_prec = validate(test_loader, model, criterion, test=True)
    print('Testing precision: {:.04f}'.format(test_prec))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        feature, output = model(input_var)
        loss = criterion(output, target_var.squeeze(1))

        #################################################
        ###################  feature  ###################
        #################################################
        # feature = feature.data.cpu().numpy()
        # scipy.io.savemat(os.path.join(args.resume, '../features/train/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        feature, output = model(input_var)
        loss = criterion(output, target_var.squeeze(1))

        #################################################
        ###################  feature  ###################
        #################################################
        # feature = feature.data.cpu().numpy()
        # if test:
        #     scipy.io.savemat(os.path.join(args.resume, '../features/test/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})
        # else:
        #     scipy.io.savemat(os.path.join(args.resume, '../features/val/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    torch.save(state, os.path.join(args.resume, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.resume, filename), os.path.join(args.resume, 'model_best.pth'))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    output = torch.nn.Softmax(dim=-1)(output)
    correct_num = 0
    output_np = output.data.cpu().numpy()
    target_np = target.cpu().numpy()
    for batch_i in range(target.size()[0]):
        if np.argmax(output_np[batch_i, :]) == target_np[batch_i, 0]:
            correct_num += 1
    return float(correct_num) / target.size()[0]


def parse_arguments():
    paths = hico_config.Paths()
    feature_type = 'resnet'

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--feature-type', default=feature_type, help='feature_type')
    parser.add_argument('--data', metavar='DIR', default=paths.data_root, help='path to dataset')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/hico/finetune_{}'.format(feature_type)), type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

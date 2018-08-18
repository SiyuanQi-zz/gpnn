"""
Created on Feb 25, 2018

@author: Siyuan Qi

Description of the file.

"""

# System packages
import os
import time
import datetime
import argparse
import pickle

# Supporting libraries
import numpy as np
import torch
import torch.utils.data
import torch.autograd
import torch.optim

# Local sub-modules
import datasets
import units

# Local imports
import config
import logutil
import utils


def evaluation(output, target, human_nums, obj_nums, test=False):
    output = torch.nn.Sigmoid()(output)
    # print 'mean link:', np.mean(target.data.cpu().numpy())
    # print target.size(), torch.sum(target).data.cpu().numpy()
    # return torch.mean(torch.abs(output - target))

    error = 0
    batch_size = output.size()[0]
    for batch_i in range(batch_size):
        node_num = human_nums[batch_i] + obj_nums[batch_i]
        error += torch.mean(torch.abs(output[batch_i, :node_num, :node_num] - target[batch_i, :node_num, :node_num]))
    return error/batch_size


def criterion(output, target, human_nums, obj_nums):
    weight_mask = torch.autograd.Variable(torch.ones(target.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight if hasattr(args, 'link_weight') else 1.0
    weight_mask += target * link_weight
    # return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask).cuda()(output, target)

    loss = 0
    batch_size = output.size()[0]
    for batch_i in range(batch_size):
        node_num = human_nums[batch_i] + obj_nums[batch_i]
        loss += torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask[batch_i, :node_num, :node_num]).cuda()(output[batch_i, :node_num, :node_num], target[batch_i, :node_num, :node_num])

    return loss/batch_size


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = utils.get_vcoco_data(args)

    # Get data size and define model
    edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes = training_set[0]
    edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    model = units.LinkFunction('GraphConv', {'edge_feature_size': edge_feature_size, 'link_hidden_size': 256, 'link_hidden_layers': 3, 'link_relu': False})
    del edge_features, node_features, adj_mat, node_labels
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = torch.nn.MultiLabelSoftMarginLoss(size_average=True)
    if args.cuda:
        model = model.cuda()
        # criterion = criterion.cuda()

    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, evaluation, logger, args=args)
        # test on validation set
        epoch_error = validate(valid_loader, model, criterion, evaluation, logger, args=args)

        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 2:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.01:
                pass
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        if epoch % 2 == 1:
            print('Learning rate decrease')
            args.lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        is_best = epoch_error < best_epoch_error
        best_epoch_error = min(epoch_error, best_epoch_error)
        datasets.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                        'best_epoch_error': best_epoch_error, 'avg_epoch_error': avg_epoch_error,
                                        'optimizer': optimizer.state_dict(), },
                                       is_best=is_best, directory=args.resume)
        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error,  avg_epoch_error))

    # For testing
    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint
    validate(test_loader, model, criterion, evaluation, logger, args=args, test=True)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger, args=None):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    error_ratio = logutil.AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_ids, img_names, human_nums, obj_nums, classes) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = utils.to_variable(edge_features, args.cuda)
        edge_features = edge_features.permute(0, 3, 1, 2)
        target = utils.to_variable(adj_mat, args.cuda)
        output = model(edge_features)
        train_loss = criterion(output, target, human_nums, obj_nums)

        # Log
        losses.update(train_loss.data[0], edge_features.size(0))
        error_ratio.update(evaluation(output, target, human_nums, obj_nums).data[0], edge_features.size(0))

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, err=error_ratio))

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_error_ratio', error_ratio.avg)

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None, args=None, test=False):
    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    error_ratio = logutil.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_ids, img_names, human_nums, obj_nums, classes) in enumerate(val_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        edge_features = edge_features.permute(0, 3, 1, 2)
        target = utils.to_variable(adj_mat, args.cuda)
        output = model(edge_features)

        # Logs
        losses.update(criterion(output, target, human_nums, obj_nums).data[0], edge_features.size(0))
        error_ratio.update(evaluation(output, target, human_nums, obj_nums, test=test).data[0], edge_features.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, err=error_ratio))

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return losses.avg


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    feature_type = 'resnet'

    # Path settings
    parser = argparse.ArgumentParser(description='VCOCO dataset')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=paths.vcoco_data_root, help='data path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'vcoco/graph_{}'.format(feature_type)), help='log files path')
    parser.add_argument('--feature-type', default=feature_type, help='feature_type')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/vcoco/graph_{}'.format(feature_type)), help='path to latest checkpoint')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--link-weight', type=float, default=5, metavar='N',
                        help='Loss weight of existing edges')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.8, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

"""
Created on Oct 03, 2017

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


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()
    # args.resume = None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = utils.get_cad_data(args)

    # Get data size and define model
    edge_features, node_features, adj_mat, node_labels, sequence_id = training_set[0]
    model = units.LinkFunction('GraphConvLSTM', {'edge_feature_size': edge_features.shape[0], 'link_hidden_size': 1024, 'link_hidden_layers': 2})
    # model = units.LinkFunction('GraphConv', {'edge_feature_size': edge_features.shape[0], 'link_hidden_size': 1024, 'link_hidden_layers': 3})
    del edge_features, node_features, adj_mat, node_labels
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    evaluation = lambda output, target: torch.mean(torch.abs(output - target))
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

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
        if len(epoch_errors) == 15:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.03:
                args.lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

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
    validate(test_loader, model, criterion, evaluation, args=args)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger, args=None):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    error_ratio = logutil.AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, node_nums) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        target = utils.to_variable(adj_mat, args.cuda)
        output = model(utils.to_variable(edge_features, args.cuda))
        train_loss = criterion(output, target)

        # Log
        losses.update(train_loss.data[0], edge_features.size(0))
        error_ratio.update(evaluation(output, target).data[0], edge_features.size(0))

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # if i % args.log_interval == 0 and i > 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Error Ratio {err.val:.4f} ({err.avg:.4f})'
        #           .format(epoch, i, len(train_loader), batch_time=batch_time,
        #                   data_time=data_time, loss=losses, err=error_ratio))

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_error_ratio', error_ratio.avg)

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None, args=None):
    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    error_ratio = logutil.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, node_nums) in enumerate(val_loader):
        target = utils.to_variable(adj_mat, args.cuda)
        output = model(utils.to_variable(edge_features, args.cuda))

        # Logs
        losses.update(criterion(output, target).data[0], edge_features.size(0))
        error_ratio.update(evaluation(output, target).data[0], edge_features.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.log_interval == 0 and i > 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Error Ratio {err.val:.4f} ({err.avg:.4f})'
        #           .format(i, len(val_loader), batch_time=batch_time,
        #                   loss=losses, err=error_ratio))

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    # Path settings
    parser = argparse.ArgumentParser(description='CAD 120 dataset')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'cad120/graph'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/cad120/graph'), help='path to latest checkpoint')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=2e-5, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.8, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

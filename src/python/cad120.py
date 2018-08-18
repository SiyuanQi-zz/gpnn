"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import argparse
import time
import datetime

import numpy as np
import torch
import torch.autograd
import sklearn.metrics

import datasets
import units
import models

import config
import logutil
import utils


def evaluation(pred_node_labels, node_labels):
    np_pred_node_labels = pred_node_labels.data.cpu().numpy()
    np_node_labels = node_labels.data.cpu().numpy()
    predictions = list()
    ground_truth = list()

    error_count = 0
    total_nodes = 0
    for batch_i in range(np_pred_node_labels.shape[0]):
        total_nodes += np_pred_node_labels.shape[1]
        pred_indices = np.argmax(np_pred_node_labels[batch_i, :, :], 1)
        indices = np.argmax(np_node_labels[batch_i, :, :], 1)

        predictions.extend(pred_indices)
        ground_truth.extend(indices)

        for node_i in range(np_pred_node_labels.shape[1]):
            if pred_indices[node_i] != indices[node_i]:
                error_count += 1
    return error_count/float(total_nodes), total_nodes, predictions, ground_truth


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = utils.get_cad_data(args)

    # Get data size and define model
    edge_features, node_features, adj_mat, node_labels, sequence_id = training_set[0]
    edge_feature_size, node_feature_size = edge_features.shape[0], node_features.shape[0]
    model_args = {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size, 'message_size': edge_feature_size, 'link_hidden_size': 1024, 'link_hidden_layers': 2, 'propagate_layers': 3, 'subactivity_classes': 10, 'affordance_classes': 12}
    model = models.GPNN_CAD(model_args)
    del edge_features, node_features, adj_mat, node_labels
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
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
        train(train_loader, model, criterion, optimizer, epoch, logger, args=args)
        # test on validation set
        epoch_error = validate(valid_loader, model, criterion, logger, args=args)

        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 10:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.01:
                pass
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        if epoch > 0 and epoch % 5 == 0:
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
    validate(test_loader, model, criterion, logger, args=args, test=True)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, criterion, optimizer, epoch, logger, args=None):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    subactivity_error_ratio = logutil.AverageMeter()
    affordance_error_ratio = logutil.AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, node_nums) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, args)
        train_loss = criterion(pred_node_labels, node_labels)

        # Log
        losses.update(train_loss.data[0], edge_features.size(0))
        error_rate, total_nodes, predictions, ground_truth = evaluation(pred_node_labels[:, [0], :], node_labels[:, [0], :])
        subactivity_error_ratio.update(error_rate, total_nodes)
        error_rate, total_nodes, predictions, ground_truth = evaluation(pred_node_labels[:, 1:, :], node_labels[:, 1:, :])
        affordance_error_ratio.update(error_rate, total_nodes)

        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_subactivity_error_ratio', subactivity_error_ratio.avg)
        logger.log_value('train_epoch_affordance_error_ratio', affordance_error_ratio.avg)

    print('Epoch: [{0}] Avg Subactivity Error Ratio {act_err.avg:.3f}; Avg Affordance Error Ratio {aff_err.avg:.3f}; Average Loss {loss.avg:.3f}; Batch Avg Time {b_time.avg:.3f}'
          .format(epoch, act_err=subactivity_error_ratio, aff_err=affordance_error_ratio, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, logger=None, args=None, test=False):
    if args.visualize:
        result_folder = os.path.join(args.tmp_root, 'results/CAD/figures/detection/')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()
    error_ratio = logutil.AverageMeter()
    subactivity_error_ratio = logutil.AverageMeter()
    affordance_error_ratio = logutil.AverageMeter()

    subact_predictions = list()
    subact_ground_truth = list()
    affordance_predictions = list()
    affordance_ground_truth = list()
    all_sequence_ids = list()
    all_node_nums = list()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, node_nums) in enumerate(val_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, args)

        # Logs
        losses.update(criterion(pred_node_labels, node_labels).data[0], edge_features.size(0))
        error_rate, total_nodes, predictions, ground_truth = evaluation(pred_node_labels, node_labels)
        error_ratio.update(error_rate, total_nodes)
        error_rate, total_nodes, predictions, ground_truth = evaluation(pred_node_labels[:, [0], :], node_labels[:, [0], :])
        subactivity_error_ratio.update(error_rate, total_nodes)
        subact_predictions.extend(predictions)
        subact_ground_truth.extend(ground_truth)
        error_rate, total_nodes, predictions, ground_truth = evaluation(pred_node_labels[:, 1:, :], node_labels[:, 1:, :])
        affordance_error_ratio.update(error_rate, total_nodes)
        affordance_predictions.extend(predictions)
        affordance_ground_truth.extend(ground_truth)
        all_sequence_ids.extend(sequence_ids)
        all_node_nums.extend(node_nums)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if args.visualize:
        utils.plot_all_activity_segmentations(all_sequence_ids, subact_predictions, subact_ground_truth, result_folder)
        utils.plot_all_affordance_segmentations(all_sequence_ids, all_node_nums, affordance_predictions, affordance_ground_truth, result_folder)

        # Plot confusion matrices
        confusion_matrix = sklearn.metrics.confusion_matrix(subact_ground_truth, subact_predictions,
                                                            labels=range(len(datasets.cad_metadata.subactivities)))
        utils.plot_confusion_matrix(confusion_matrix, datasets.cad_metadata.subactivities, normalize=True, title='',
                              filename=os.path.join(result_folder, 'confusion_subactivity.pdf'))

        confusion_matrix = sklearn.metrics.confusion_matrix(affordance_ground_truth, affordance_predictions,
                                                            labels=range(len(datasets.cad_metadata.affordances)))
        utils.plot_confusion_matrix(confusion_matrix, datasets.cad_metadata.affordances, normalize=True, title='',
                              filename=os.path.join(result_folder, 'confusion_affordance.pdf'))

    subact_micro_result = sklearn.metrics.precision_recall_fscore_support(subact_ground_truth, subact_predictions, labels=range(10), average='micro')
    subact_macro_result = sklearn.metrics.precision_recall_fscore_support(subact_ground_truth, subact_predictions, labels=range(10), average='macro')
    aff_micro_result = sklearn.metrics.precision_recall_fscore_support(affordance_ground_truth, affordance_predictions, labels=range(12), average='micro')
    aff_macro_result = sklearn.metrics.precision_recall_fscore_support(affordance_ground_truth, affordance_predictions, labels=range(12), average='macro')
    if test:
        print('Subactivity detection micro evaluation:', subact_micro_result)
        print('Subactivity detection macro evaluation:', subact_macro_result)
        print('Affordance detection micro evaluation:', aff_micro_result)
        print('Affordance detection macro evaluation:', aff_macro_result)

    print(' * Avg Subactivity Error Ratio {act_err.avg:.3f}; Avg Affordance Error Ratio {aff_err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(act_err=subactivity_error_ratio, aff_err=affordance_error_ratio, loss=losses))
    print(' * Subactivity F1 Score {:.3f}; Affordance F1 Score {:.3f};'.format(subact_macro_result[2], aff_macro_result[2]))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_subactivity_error_ratio', subactivity_error_ratio.avg)
        logger.log_value('test_epoch_affordance_error_ratio', affordance_error_ratio.avg)
        logger.log_value('test_epoch_subactivity_f1_detection', subact_macro_result[2])
        logger.log_value('test_epoch_affordance_f1_detection', aff_macro_result[2])

    # return error_ratio.avg
    # return subactivity_error_ratio.avg+affordance_error_ratio.avg
    return 2.0-(subact_macro_result[2] + aff_macro_result[2])
    # return 1.0 - aff_macro_result[2]


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
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'cad120/parsing'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/cad120/parsing'), help='path to latest checkpoint')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize final results')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=5e-5, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.8, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.8)')
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

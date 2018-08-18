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
from datasets.HICO import metadata
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy


action_class_num = 117
hoi_class_num = 600


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False):
    np_pred_node_labels = pred_node_labels.data.cpu().numpy()
    np_pred_node_labels_exp = np.exp(np_pred_node_labels)
    np_pred_node_labels = np_pred_node_labels_exp/(np_pred_node_labels_exp+1)  # overflows when x approaches np.inf
    np_node_labels = node_labels.data.cpu().numpy()

    new_y_true = np.empty((2 * len(det_indices), action_class_num))
    new_y_score = np.empty((2 * len(det_indices), action_class_num))
    for y_i, (batch_i, i, j) in enumerate(det_indices):
        new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
        new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
        new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
        new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :]

    y_true = np.vstack((y_true, new_y_true))
    y_score = np.vstack((y_score, new_y_score))
    return y_true, y_score


def weighted_loss(output, target):
    weight_mask = torch.autograd.Variable(torch.ones(target.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight if hasattr(args, 'link_weight') else 1.0
    weight_mask += target * link_weight
    return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask).cuda()(output, target)


def loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num=[], obj_num=[]):
    np_pred_adj_mat = pred_adj_mat.data.cpu().numpy()
    det_indices = list()
    batch_size = pred_adj_mat.size()[0]
    loss = 0
    for batch_i in range(batch_size):
        valid_node_num = human_num[batch_i] + obj_num[batch_i]
        np_pred_adj_mat_batch = np_pred_adj_mat[batch_i, :, :]

        if len(human_num) != 0:
            human_interval = human_num[batch_i]
            obj_interval = human_interval + obj_num[batch_i]
        max_score = np.max([np.max(np_pred_adj_mat_batch), 0.01])
        mean_score = np.mean(np_pred_adj_mat_batch)

        batch_det_indices = np.where(np_pred_adj_mat_batch > 0.5)
        for i, j in zip(batch_det_indices[0], batch_det_indices[1]):
            # check validity for H-O interaction instead of O-O interaction
            if len(human_num) != 0:
                if i < human_interval and j < obj_interval:
                    if j >= human_interval:
                        det_indices.append((batch_i, i, j))

        loss = loss + weighted_loss(pred_node_labels[batch_i, :valid_node_num].view(-1, action_class_num), node_labels[batch_i, :valid_node_num].view(-1, action_class_num))
    return det_indices, loss


def compute_mean_avg_prec(y_true, y_score):
    try:
        avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
        mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
    except ValueError:
        mean_avg_prec = 0

    return mean_avg_prec


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader, img_index = utils.get_hico_data(args)

    # Get data size and define model
    edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num = training_set[0]

    edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    message_size = int(edge_feature_size/2)*2
    model_args = {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size, 'message_size': message_size, 'link_hidden_size': 512, 'link_hidden_layers': 2, 'link_relu': False, 'update_hidden_layers': 1, 'update_dropout': False, 'update_bias': True, 'propagate_layers': 3, 'hoi_classes': action_class_num, 'resize_feature_to_message_size': False}
    model = models.GPNN_HICO(model_args)
    del edge_features, node_features, adj_mat, node_labels
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss(size_average=True)
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(size_average=True)
    if args.cuda:
        model = model.cuda()
        mse_loss = mse_loss.cuda()
        multi_label_loss = multi_label_loss.cuda()

    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()
        # train for one epoch
        train(train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, logger)
        # test on validation set
        epoch_error = validate(valid_loader, model, mse_loss, multi_label_loss, logger)
        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 2:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.005:
                print('Learning rate decrease')
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        if epoch % 5 == 0 and epoch > 0:
            args.lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        is_best = True
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

    # validate(test_loader, model, mse_loss, multi_label_loss, test=True)
    gen_test_result(args, test_loader, model, mse_loss, multi_label_loss, img_index)

    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, logger):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    # switch to train mode
    model.train()

    end_time = time.time()

    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_num, obj_num) in enumerate(train_loader):

        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num, obj_num)

        # Log and back propagate
        if len(det_indices) > 0:
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score)

        losses.update(loss.data[0], edge_features.size()[0])
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    print('Epoch: [{0}] Avg Mean Precision {map:.4f}; Average Loss {loss.avg:.4f}; Avg Time x Batch {b_time.avg:.4f}'
          .format(epoch, map=mean_avg_prec, loss=losses, b_time=batch_time))


def validate(val_loader, model, mse_loss, multi_label_loss, logger=None, test=False):
    if args.visualize:
        result_folder = os.path.join(args.tmp_root, 'results/HICO/detections/', 'top'+str(args.vis_top_k))
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_num, obj_num) in enumerate(val_loader):

        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num, obj_num)

        # Log
        if len(det_indices) > 0:
            losses.update(loss.data[0], len(det_indices))
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=test)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    print(' * Average Mean Precision {mean_avg_prec:.4f}; Average Loss {loss.avg:.4f}'
          .format(mean_avg_prec=mean_avg_prec, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    return 1.0 - mean_avg_prec

def get_indices(pred_adj_mat, pred_node_labels, human_num, obj_num, det_class, det_box):

    # Normalize adjacency matrix
    pred_adj_mat_prob = torch.nn.Sigmoid()(pred_adj_mat)
    np_pred_adj_mat = pred_adj_mat_prob.data.cpu().numpy()
    # Normalize label outputs
    pred_node_labels_prob = torch.nn.Sigmoid()(pred_node_labels)
    np_pred_node_labels = pred_node_labels_prob.data.cpu().numpy()

    hois = list()
    threshold1 = 0
    threshold2 = 0
    threshold3 = 0
    for h_idx in range(human_num):
        bbox_h = det_box[h_idx]
        h_class = det_class[h_idx]
        for a_idx in range(len(metadata.action_classes)):
            if np_pred_node_labels[h_idx, a_idx] < threshold1:
                continue
            max_score = -np.inf
            selected = -1

            for o_idx in range(human_num + obj_num):
                obj_name = det_class[o_idx]
                if a_idx not in metadata.obj_actions[obj_name]:
                    continue
                if np_pred_node_labels[h_idx, a_idx] < threshold2:
                    continue
                if np_pred_adj_mat[h_idx, o_idx] < threshold3:
                    continue
                score = np_pred_adj_mat[h_idx, o_idx] * np_pred_node_labels[h_idx, a_idx] * np_pred_node_labels[o_idx, a_idx]
                bbox_o = det_box[o_idx]
                rel_idx = metadata.action_to_obj_idx(obj_name, a_idx)
                hois.append((h_class, obj_name, a_idx, (bbox_h, bbox_o, rel_idx, score), (np_pred_node_labels[h_idx, a_idx], np_pred_node_labels[o_idx, a_idx], np_pred_adj_mat[h_idx, o_idx])))

    return hois


def gen_test_result(args, test_loader, model, mse_loss, multi_label_loss, img_index):
    filtered_hoi = dict()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    total_idx = 0
    obj_stats = dict()

    filtered_hoi = dict()

    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_nums, obj_nums) in enumerate(test_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)
        if sequence_ids[0] is 'HICO_test2015_00000396':
            break

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args)

        for batch_i in range(pred_adj_mat.size()[0]):
            sequence_id = sequence_ids[batch_i]
            hois_test = get_indices(pred_adj_mat[batch_i], pred_node_labels[batch_i], human_nums[batch_i], obj_nums[batch_i],
                        det_classes[batch_i], det_boxes[batch_i])
            hois_gt = get_indices(adj_mat[batch_i], node_labels[batch_i], human_nums[batch_i],
                                    obj_nums[batch_i],
                                    det_classes[batch_i], det_boxes[batch_i])
            for hoi in hois_test:
                _, o_idx, a_idx, info, _ = hoi
                if o_idx not in filtered_hoi.keys():
                    filtered_hoi[o_idx] = dict()
                if sequence_id not in filtered_hoi[o_idx].keys():
                    filtered_hoi[o_idx][sequence_id] = list()
                filtered_hoi[o_idx][sequence_id].append(info)

        print("finished generating result from " + sequence_ids[0] + " to " + sequence_ids[-1])

    for obj_idx, save_info in filtered_hoi.items():
        obj_start, obj_end = metadata.obj_hoi_index[obj_idx]
        obj_arr = np.empty((obj_end - obj_start + 1, len(img_index)), dtype=np.object)
        for row in range(obj_arr.shape[0]):
            for col in range(obj_arr.shape[1]):
                obj_arr[row][col] = []
        for id, data_info in save_info.items():
            col_idx = img_index.index(id)
            for pair in data_info:
                row_idx = pair[2]
                bbox_concat = np.concatenate((pair[0], pair[1], [pair[3]]))
                if len(obj_arr[row_idx][col_idx]) > 0:
                    obj_arr[row_idx][col_idx] = np.vstack((obj_arr[row_idx][col_idx], bbox_concat))
                else:
                    obj_arr[row_idx][col_idx] = bbox_concat
        sio.savemat(os.path.join(args.tmp_root, 'results', 'HICO', 'detections_' + str(obj_idx).zfill(2) + '.mat'), {'all_boxes': obj_arr})
        print('finished saving for ' + str(obj_idx))

    return


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    # Path settings
    parser = argparse.ArgumentParser(description='HICO dataset')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=paths.hico_data_root, help='data path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'hico/parsing'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/hico/parsing'), help='path to latest checkpoint')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize final results')
    parser.add_argument('--vis-top-k', type=int, default=1, metavar='N', help='Top k results to visualize')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--link-weight', type=float, default=100, metavar='N',
                        help='Loss weight of existing edges')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-5, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

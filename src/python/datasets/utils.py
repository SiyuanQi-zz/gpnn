"""
Created on Oct 04, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil

import numpy as np
import torch


def collate_fn_cad(batch):
    edge_features, node_features, adj_mat, node_labels, sequence_id = batch[0]
    max_node_num = np.max(np.array([[adj_mat.shape[0]] for (edge_features, node_features, adj_mat, node_labels, sequence_id) in batch]))
    edge_feature_len = edge_features.shape[0]
    node_feature_len = node_features.shape[0]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[1]
    del edge_features, node_features, adj_mat, node_labels

    edge_features_batch = np.zeros((len(batch), edge_feature_len, max_node_num, max_node_num))
    node_features_batch = np.zeros((len(batch), node_feature_len, max_node_num))
    adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), max_node_num))

    sequence_ids = list()
    node_nums = list()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_id) in enumerate(batch):
        node_num = adj_mat.shape[0]
        edge_features_batch[i, :, :node_num, :node_num] = edge_features
        node_features_batch[i, :, :node_num] = node_features
        adj_mat_batch[i, :node_num, :node_num] = adj_mat
        if node_label_dim > 1:
            node_labels_batch[i, :node_num, :] = node_labels
        else:
            node_labels_batch[i, :node_num] = node_labels
        sequence_ids.append(sequence_id)
        node_nums.append(node_num)

    edge_features_batch = torch.FloatTensor(edge_features_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    adj_mat_batch = torch.FloatTensor(adj_mat_batch)
    node_labels_batch = torch.FloatTensor(node_labels_batch)

    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, sequence_ids, node_nums


def collate_fn_hico(batch):
    edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num = batch[0]
    max_node_num = np.max(np.array([[adj_mat.shape[0]] for (edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num) in batch]))

    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[1]
    node_label_dim = node_labels.ndim
    if node_label_dim > 1:
        node_label_len = node_labels.shape[1]
    del edge_features, node_features, adj_mat, node_labels

    edge_features_batch = np.zeros((len(batch), max_node_num, max_node_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num))
    sequence_ids = list()
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), max_node_num))
    classes_batch = list()
    boxes_batch = list()
    human_nums = list()
    obj_nums = list()

    for i, (edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num) in enumerate(batch):

        node_num = adj_mat.shape[0]
        edge_features_batch[i, :node_num, :node_num, :] = edge_features
        node_features_batch[i, :node_num, :] = node_features
        adj_mat_batch[i, :node_num, :node_num] = adj_mat
        if node_label_dim > 1:
            node_labels_batch[i, :node_num, :] = node_labels
        else:
            node_labels_batch[i, :node_num] = node_labels
        sequence_ids.append(sequence_id)

        boxes_batch.append(det_boxes)
        classes_batch.append(det_classes)
        human_nums.append(human_num)
        obj_nums.append(obj_num)


    edge_features_batch = torch.FloatTensor(edge_features_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    adj_mat_batch = torch.FloatTensor(adj_mat_batch)
    node_labels_batch = torch.FloatTensor(node_labels_batch)

    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, sequence_ids, classes_batch, boxes_batch, human_nums, obj_nums


def collate_fn_vcoco(batch):
    edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes = batch[0]
    max_node_num = np.max(np.array([[adj_mat.shape[0]] for (edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes) in batch]))
    edge_feature_len = edge_features.shape[2]
    node_feature_len = node_features.shape[1]
    node_label_dim = node_labels.ndim
    node_role_num = node_roles.shape[1]
    if node_label_dim > 1:
        node_label_len = node_labels.shape[1]
    del edge_features, node_features, adj_mat, node_labels

    edge_features_batch = np.zeros((len(batch), max_node_num, max_node_num, edge_feature_len))
    node_features_batch = np.zeros((len(batch), max_node_num, node_feature_len))
    adj_mat_batch = np.zeros((len(batch), max_node_num, max_node_num))
    if node_label_dim > 1:
        node_labels_batch = np.zeros((len(batch), max_node_num, node_label_len))
    else:
        node_labels_batch = np.zeros((len(batch), max_node_num))
    node_roles_batch = np.zeros((len(batch), max_node_num, node_role_num))
    img_names = list()
    img_ids = list()
    boxes_batch = list()
    human_nums = list()
    obj_nums = list()
    classes_batch = list()

    for i, (edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes) in enumerate(batch):
        node_num = adj_mat.shape[0]
        edge_features_batch[i, :node_num, :node_num, :] = edge_features
        node_features_batch[i, :node_num, :] = node_features
        adj_mat_batch[i, :node_num, :node_num] = adj_mat
        if node_label_dim > 1:
            node_labels_batch[i, :node_num, :] = node_labels
        else:
            node_labels_batch[i, :node_num] = node_labels
        node_roles_batch[i, :node_num, :] = node_roles
        img_names.append(img_name)
        img_ids.append(img_id)
        boxes_batch.append(boxes)
        human_nums.append(human_num)
        obj_nums.append(obj_num)
        classes_batch.append(classes)

    edge_features_batch = torch.FloatTensor(edge_features_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    adj_mat_batch = torch.FloatTensor(adj_mat_batch)
    node_labels_batch = torch.FloatTensor(node_labels_batch)
    node_roles_batch = torch.FloatTensor(node_roles_batch)

    return edge_features_batch, node_features_batch, adj_mat_batch, node_labels_batch, node_roles_batch, boxes_batch, img_ids, img_names, human_nums, obj_nums, classes_batch


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def load_best_checkpoint(args, model, optimizer):
    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_epoch_error = checkpoint['best_epoch_error']
            try:
                avg_epoch_error = checkpoint['avg_epoch_error']
            except KeyError:
                avg_epoch_error = np.inf
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                model.cuda()
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            return args, best_epoch_error, avg_epoch_error, model, optimizer
        else:
            print("=> no best model found at '{}'".format(best_model_file))
    return None


def main():
    pass


if __name__ == '__main__':
    main()

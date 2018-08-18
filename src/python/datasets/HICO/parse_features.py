"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file.

"""

from __future__ import print_function
import os
import time
import pickle

import numpy as np
import scipy.io

import hico_config
import metadata


def parse_classes(det_classes):
    obj_nodes = False
    human_num = 0
    obj_num = 0
    for i in range(det_classes.shape[0]):
        if not obj_nodes:
            if det_classes[i] == 1:
                human_num += 1
            else:
                obj_nodes = True
                obj_num += 1
        else:
            if det_classes[i] > 1:
                obj_num += 1
            else:
                break

    node_num = human_num + obj_num
    edge_num = det_classes.shape[0] - node_num
    return human_num, obj_num, edge_num


def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))


def compute_area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def get_node_index(classname, bbox, det_classes, det_boxes, node_num):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in range(node_num):
        # print(classname, metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]])
        if classname == metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]]:
            # check bbox overlap
            intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node, :]))
            iou = intersection_area/(compute_area(bbox)+compute_area(det_boxes[i_node, :])-intersection_area)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i_node
    return max_iou_index

def read_features(data_root, tmp_root, bbox, list_action):
    # roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    roi_size = 49  # VGG fully connected feature
    # hoi_class_num = 600
    action_class_num = 117
    # feature_path = os.path.join(data_root, 'processed', 'features_background_49')
    save_data_path = os.path.join(data_root, 'processed', 'hico_data_background_49')
    feature_path = os.path.join(data_root, 'processed', 'features_background_49')
    #save_data_path = os.path.join(data_root, 'processed', 'hico_data_roi_vgg')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    for i_image in range(bbox['filename'].shape[1]):
        filename = os.path.splitext(bbox['filename'][0, i_image][0])[0]
        print(filename)

        try:
            det_classes = np.load(os.path.join(feature_path, '{}_classes.npy'.format(filename)))
            det_boxes = np.load(os.path.join(feature_path, '{}_boxes.npy'.format(filename)))
            det_features = np.load(os.path.join(feature_path, '{}_features.npy'.format(filename)))
        except IOError:
            continue

        human_num, obj_num, edge_num = parse_classes(det_classes)
        node_num = human_num + obj_num
        assert edge_num == human_num * obj_num

        edge_features = np.zeros((human_num+obj_num, human_num+obj_num, roi_size))
        node_features = np.zeros((node_num, roi_size*2))
        adj_mat = np.zeros((human_num+obj_num, human_num+obj_num))
        node_labels = np.zeros((node_num, action_class_num))

        # Node features
        for i_node in range(node_num):
            # node_features[i_node, :] = np.reshape(det_features[i_node, ...], roi_size)
            if i_node < human_num:
                node_features[i_node, :roi_size] = np.reshape(det_features[i_node, ...], roi_size)
            else:
                node_features[i_node, roi_size:] = np.reshape(det_features[i_node, ...], roi_size)

        # Edge features
        i_edge = 0
        for i_human in range(human_num):
            for i_obj in range(obj_num):
                edge_features[i_human, human_num + i_obj, :] = np.reshape(det_features[node_num + i_edge, ...], roi_size)
                edge_features[human_num + i_obj, i_human, :] = edge_features[i_human, human_num + i_obj, :]
                i_edge += 1

        # Adjacency matrix and node labels
        for i_hoi in range(bbox['hoi'][0, i_image]['id'].shape[1]):
            try:
                classname = 'person'
                x1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y2'][0, 0][0, 0]
                human_index = get_node_index(classname, [x1, y1, x2, y2], det_classes, det_boxes, node_num)

                hoi_id = bbox['hoi'][0, i_image]['id'][0, i_hoi][0, 0]
                classname = list_action['nname'][hoi_id, 0][0]
                x1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y2'][0, 0][0, 0]
                obj_index = get_node_index(classname, [x1, y1, x2, y2], det_classes, det_boxes, node_num)

                action_id = metadata.hoi_to_action[hoi_id]
                if human_index != -1 and obj_index != -1:
                    adj_mat[human_index, obj_index] = 1
                    adj_mat[obj_index, human_index] = 1
                    node_labels[human_index, action_id] = 1
                    node_labels[obj_index, action_id] = 1
            except IndexError:
                pass

        instance = dict()
        instance['human_num'] = human_num
        instance['obj_num'] = obj_num
        instance['img_name'] = filename
        instance['boxes'] = det_boxes
        instance['classes'] = det_classes
        # instance['edge_features'] = edge_features
        # instance['node_features'] = node_features
        instance['adj_mat'] = adj_mat
        instance['node_labels'] = node_labels
        np.save(os.path.join(save_data_path, '{}_edge_features'.format(filename)), edge_features)
        np.save(os.path.join(save_data_path, '{}_node_features'.format(filename)), node_features)
        pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))

def read_features_(data_root, tmp_root, bbox, list_action):
    # roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    roi_size = 1000  # VGG fully connected feature
    feature_size = 49
    feature_type = 'resnet'
    action_class_num = len(metadata.action_classes)
    # feature_path = os.path.join(data_root, 'processed', 'features_background_49')
    save_data_path = os.path.join(data_root, 'processed', 'hico_data_background_49')
    det_feature_path = os.path.join(data_root, 'processed', 'features_background_49')
    roi_feature_path = os.path.join(data_root, 'features_{}'.format(feature_type))
    #save_data_path = os.path.join(data_root, 'processed', feature_type)

    img_list_file = os.path.join(data_root, '../../tmp/hico', 'trainvaltest.txt')
    image_list = list()
    with open(img_list_file) as f:
        for line in f.readlines():
            image_list.append(line.strip())

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    for i_image in range(bbox['filename'].shape[1]):
        img_name = os.path.splitext(bbox['filename'][0, i_image][0])[0]
        if img_name not in image_list:
            print('skipping ' + img_name)
            continue

        print(img_name)

        try:
            bbox_features = np.load(os.path.join(roi_feature_path, '{}_features.npy'.format(img_name)))
            det_classes = np.load(os.path.join(det_feature_path, '{}_classes.npy'.format(img_name)))
            det_boxes = np.load(os.path.join(det_feature_path, '{}_boxes.npy'.format(img_name)))
            det_features = np.load(os.path.join(det_feature_path, '{}_features.npy'.format(img_name)))
        except IOError:
            print('-----Image indexing error!!!!!----')
            continue

        human_num, obj_num, edge_num = parse_classes(det_classes)
        node_num = human_num + obj_num
        assert edge_num == human_num * obj_num

        edge_features = np.zeros((human_num+obj_num, human_num+obj_num, roi_size))
        node_features = np.zeros((node_num, roi_size + feature_size * 2))
        adj_mat = np.zeros((human_num+obj_num, human_num+obj_num))
        node_labels = np.zeros((node_num, action_class_num))

        # Node features
        for i_node in range(node_num):
            # node_features[i_node, :] = np.reshape(det_features[i_node, ...], roi_size)
            node_features[i_node, :roi_size] = np.reshape(bbox_features[i_node, ...], roi_size)
            if i_node < human_num:
                node_features[i_node, roi_size:roi_size + feature_size] = np.reshape(det_features[i_node, ...], feature_size)
            else:
                node_features[i_node, roi_size + feature_size:] = np.reshape(det_features[i_node, ...], feature_size)

        # Edge features
        i_edge = 0
        for i_human in range(human_num):
            for i_obj in range(obj_num):
                edge_features[i_human, human_num + i_obj, :] = np.reshape(bbox_features[node_num + i_edge, ...], roi_size)
                edge_features[human_num + i_obj, i_human, :] = edge_features[i_human, human_num + i_obj, :]
                i_edge += 1

        # Adjacency matrix and node labels
        for i_hoi in range(bbox['hoi'][0, i_image]['id'].shape[1]):
            try:
                classname = 'person'
                x1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y2'][0, 0][0, 0]
                human_index = get_node_index(classname, [x1, y1, x2, y2], det_classes, det_boxes, node_num)

                hoi_id = bbox['hoi'][0, i_image]['id'][0, i_hoi][0, 0]
                classname = list_action['nname'][hoi_id, 0][0]
                x1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y2'][0, 0][0, 0]
                obj_index = get_node_index(classname, [x1, y1, x2, y2], det_classes, det_boxes, node_num)

                action_id = metadata.hoi_to_action[hoi_id]
                if human_index != -1 and obj_index != -1:
                    adj_mat[human_index, obj_index] = 1
                    adj_mat[obj_index, human_index] = 1
                    node_labels[human_index, action_id] = 1
                    node_labels[obj_index, action_id] = 1
            except IndexError:
                pass

        instance = dict()
        instance['human_num'] = human_num
        instance['obj_num'] = obj_num
        instance['img_name'] = img_name
        instance['boxes'] = det_boxes
        instance['classes'] = det_classes
        # instance['edge_features'] = edge_features
        # instance['node_features'] = node_features
        instance['adj_mat'] = adj_mat
        instance['node_labels'] = node_labels
        np.save(os.path.join(save_data_path, '{}_edge_features'.format(img_name)), edge_features)
        np.save(os.path.join(save_data_path, '{}_node_features'.format(img_name)), node_features)
        pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(img_name)), 'wb'))


def collect_data(paths):
    anno_bbox = scipy.io.loadmat(os.path.join(paths.data_root, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']
    list_action = anno_bbox['list_action']

    read_features(paths.data_root, paths.tmp_root, bbox_train, list_action)
    read_features(paths.data_root, paths.tmp_root, bbox_test, list_action)


def main():
    paths = hico_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

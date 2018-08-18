"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

# from __future__ import print_function
import os
import time
import pickle
import warnings

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import vcoco_config
import vsrl_eval
import vsrl_utils as vu
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
            if det_classes[i] > 1 or det_classes[i] == -1:
                obj_num += 1
            else:
                break

    node_num = human_num + obj_num
    edge_num = det_classes.shape[0] - node_num
    return human_num, obj_num, edge_num


def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))


def compute_area(box):
    side1 = box[2]-box[0]
    side2 = box[3]-box[1]
    if side1 > 0 and side2 > 0:
        return side1 * side2
    else:
        return 0.0


def compute_iou(box1, box2):
    intersection_area = compute_area(get_intersection(box1, box2))
    iou = intersection_area / (compute_area(box1) + compute_area(box2) - intersection_area)
    return iou


def get_node_index(bbox, det_boxes, index_list):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in index_list:
        # check bbox overlap
        iou = compute_iou(bbox, det_boxes[i_node, :])
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i_node
    return max_iou_index


def parse_features(paths, imageset):
    roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    # roi_size = 4096  # VGG fully connected feature
    roi_size = 1000  # ResNet fully connected feature
    feature_size = 49
    feature_type = 'resnet'
    action_class_num = len(metadata.action_classes)
    no_action_index = metadata.action_index['none']
    no_role_index = metadata.role_index['none']
    feature_path = os.path.join(paths.data_root, 'features_{}'.format(feature_type))
    det_feature_path = os.path.join(paths.data_root, 'features_deformable')
    save_data_path = os.path.join(paths.data_root, 'processed', feature_type)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    coco = vu.load_coco()
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset))
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)

    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
    all_results = list()
    unique_image_ids = list()
    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        if image_id not in unique_image_ids:
            try:
                bbox_features = np.load(os.path.join(feature_path, '{}_features.npy'.format(filename)))
                det_classes = np.load(os.path.join(det_feature_path, '{}_classes.npy'.format(filename)))
                det_boxes = np.load(os.path.join(det_feature_path, '{}_boxes.npy'.format(filename)))
                det_features = np.load(os.path.join(det_feature_path, '{}_features.npy'.format(filename)))
            except IOError:
                warnings.warn('Features and detection results missing for {}'.format(filename))
                continue

            human_num, obj_num, edge_num = parse_classes(det_classes)
            node_num = human_num + obj_num
            assert edge_num == human_num * obj_num

            unique_image_ids.append(image_id)
            edge_features = np.zeros((human_num+obj_num, human_num+obj_num, roi_size))
            node_features = np.zeros((node_num, roi_size+feature_size*2))
            adj_mat = np.zeros((human_num+obj_num, human_num+obj_num))
            node_labels = np.zeros((node_num, action_class_num))
            node_roles = np.zeros((node_num, 3))
            node_labels[:, no_action_index] = 1
            node_roles[:, no_role_index] = 1

            # Node features
            for i_node in range(node_num):
                # node_features[i_node, :] = np.reshape(det_features[i_node, ...], roi_size)
                node_features[i_node, :roi_size] = np.reshape(bbox_features[i_node, ...], roi_size)
                if i_node < human_num:
                    node_features[i_node, roi_size:roi_size+feature_size] = np.reshape(det_features[i_node, ...], feature_size)
                else:
                    node_features[i_node, roi_size+feature_size:] = np.reshape(det_features[i_node, ...], feature_size)

            # Edge features
            i_edge = 0
            for i_human in range(human_num):
                for i_obj in range(obj_num):
                    edge_features[i_human, human_num + i_obj, :] = np.reshape(bbox_features[node_num + i_edge, ...], roi_size)
                    edge_features[human_num + i_obj, i_human, :] = edge_features[i_human, human_num + i_obj, :]
                    i_edge += 1
        else:
            saved_instance = pickle.load(open(os.path.join(save_data_path, '{}.p'.format(filename)), 'rb'))
            edge_features = np.load(os.path.join(save_data_path, '{}_edge_features.npy').format(filename))
            node_features = np.load(os.path.join(save_data_path, '{}_node_features.npy').format(filename))
            adj_mat = saved_instance['adj_mat']
            node_labels = saved_instance['node_labels']
            node_roles = saved_instance['node_roles']
            human_num = saved_instance['human_num']
            obj_num = saved_instance['obj_num']
            det_boxes = saved_instance['boxes']
            det_classes = saved_instance['classes']

        # Ground truth labels: adj_mat, node_labels, node_roles
        for x in vcoco_all:
            if x['label'][i_image, 0] == 1:
                try:
                    action_index = metadata.action_index[x['action_name']]

                    role_bbox = x['role_bbox'][i_image, :] * 1.
                    role_bbox = role_bbox.reshape((-1, 4))
                    bbox = role_bbox[0, :]
                    human_index = get_node_index(bbox, det_boxes, range(human_num))
                    if human_index == -1:
                        warnings.warn('human detection missing')
                        continue
                    assert human_index < human_num
                    node_labels[human_index, action_index] = 1
                    node_labels[human_index, no_action_index] = 0

                    for i_role in range(1, len(x['role_name'])):
                        bbox = role_bbox[i_role, :]
                        if np.isnan(bbox[0]):
                            continue
                        obj_index = get_node_index(bbox, det_boxes, range(human_num, human_num+obj_num))
                        if obj_index == -1:
                            warnings.warn('object detection missing')
                            continue
                        assert obj_index >= human_num
                        node_labels[obj_index, action_index] = 1
                        node_labels[obj_index, no_action_index] = 0
                        node_roles[obj_index, metadata.role_index[x['role_name'][i_role]]] = 1
                        node_roles[obj_index, no_role_index] = 0
                        adj_mat[human_index, obj_index] = 1
                        adj_mat[obj_index, human_index] = 1
                except IndexError:
                    warnings.warn('Labels missing for {}'.format(filename))
                    pass

        instance = dict()
        instance['img_id'] = image_id
        instance['human_num'] = human_num
        instance['obj_num'] = obj_num
        instance['img_name'] = filename
        instance['boxes'] = det_boxes
        instance['classes'] = det_classes
        instance['adj_mat'] = adj_mat
        instance['node_labels'] = node_labels
        instance['node_roles'] = node_roles
        np.save(os.path.join(save_data_path, '{}_edge_features'.format(filename)), edge_features)
        np.save(os.path.join(save_data_path, '{}_node_features'.format(filename)), node_features)
        pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))

        if i_image == len(image_ids) - 1 - image_ids[::-1].index(image_id):
            append_result(all_results, node_labels, node_roles, int(image_ids[i_image]), det_boxes, human_num, obj_num, adj_mat)

    print 'total image', len(unique_image_ids), 'total results', len(all_results)
    vcocoeval = get_vcocoeval(paths, imageset)
    vcoco_evaluation(paths, vcocoeval, imageset, all_results)


def visualize_roi(paths, imageset, filename, roi):
    image_path = os.path.join(paths.data_root, '../v-coco/coco/images', '{}2014'.format(imageset), filename)
    assert os.path.exists(image_path)
    original_img = scipy.misc.imread(image_path, mode='RGB')
    roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
    plt.imshow(roi_image)
    plt.show()


def append_result(all_results, node_labels, node_roles, image_id, boxes, human_num, obj_num, adj_mat):
    for i in range(human_num):
        if node_labels[i, metadata.action_index['none']] > 0.5:
            continue
        instance_result = dict()
        instance_result['image_id'] = image_id
        instance_result['person_box'] = boxes[i, :]
        for action_index, action in enumerate(metadata.action_classes):
            if action == 'none' or node_labels[i, action_index] < 0.5:
                continue
            result = instance_result.copy()
            result['{}_agent'.format(action)] = node_labels[i, action_index]
            for role in metadata.action_roles[action][1:]:
                role_index = metadata.role_index[role]
                action_role_key = '{}_{}'.format(action, role)
                best_score = -np.inf
                for j in range(human_num, human_num+obj_num):
                    if adj_mat[i, j] > 0.5:
                        action_role_score = (node_labels[j, action_index] + node_roles[j, role_index])/2  # TODO: how to evaluate action-role score
                        if action_role_score > best_score:
                            best_score = action_role_score
                            obj_info = np.append(boxes[j, :], action_role_score)
                if best_score > 0:
                    result[action_role_key] = obj_info
            all_results.append(result)


def get_vcocoeval(paths, imageset):
    return vsrl_eval.VCOCOeval(os.path.join(paths.data_root, '..', 'v-coco/data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(paths.data_root, '..', 'v-coco/data/instances_vcoco_all_2014.json'),
                               os.path.join(paths.data_root, '..', 'v-coco/data/splits/vcoco_{}.ids'.format(imageset)))


def vcoco_evaluation(args, vcocoeval, imageset, all_results):
    det_file = os.path.join(args.eval_root, '{}_detections.pkl'.format(imageset))
    pickle.dump(all_results, open(det_file, 'wb'))
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)


def collect_data(paths):
    imagesets = ['train', 'val', 'test']
    for imageset in imagesets:
        parse_features(paths, imageset)
        # break


def main():
    start_time = time.time()
    paths = vcoco_config.Paths()
    paths.eval_root = '/home/siyuan/projects/papers/cvpr2018/tmp/evaluation/vcoco/features'
    if not os.path.exists(paths.eval_root):
        os.makedirs(paths.eval_root)
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

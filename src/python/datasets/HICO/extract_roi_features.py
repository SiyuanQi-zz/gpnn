"""
Created on Oct 12, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle
import warnings

import numpy as np
import scipy.misc
import cv2
import torch
import torch.autograd
import torchvision.models
import matplotlib.pyplot as plt

import hico_config
import roi_pooling
import roi_feature_model
import metadata


def get_model(paths, feature_type):
    if feature_type == 'vgg':
        feature_network = roi_feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif feature_type == 'resnet':
        feature_network = roi_feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif feature_type == 'densenet':
        feature_network = roi_feature_model.Densenet(num_classes=len(metadata.action_classes))
    else:
        raise ValueError('feature type not recognized')

    if feature_type.startswith('alexnet') or feature_type.startswith('vgg'):
        feature_network.features = torch.nn.DataParallel(feature_network.features)
        feature_network.cuda()
    else:
        feature_network = torch.nn.DataParallel(feature_network).cuda()

    checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'hico', 'finetune_{}'.format(feature_type))
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    checkpoint = torch.load(best_model_file)
    feature_network.load_state_dict(checkpoint['state_dict'])
    return feature_network


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths, feature_type):
    hico_path = paths.data_root
    hico_voc_path = os.path.join(hico_path, 'Deformable-ConvNets/data/hico/VOC2007')
    feature_path = os.path.join(hico_path, 'features1_{}'.format(feature_type))

    # image_list_file = os.path.join(hico_voc_path, 'ImageSets/Main/test.txt')
    # det_res_path = os.path.join(hico_path, 'Deformable-ConvNets/output/rfcn_dcn/hico/hico_detect/2007_test',
    #                             'hico_detect_test_detections.pkl')

    image_list_file = os.path.join(hico_voc_path, 'ImageSets/Main/trainvaltest.txt')
    det_res_path = os.path.join(hico_path, 'Deformable-ConvNets/output/rfcn_dcn/hico/hico_detect/2007_trainvaltest',
                                'hico_detect_trainvaltest_detections.pkl')

    classes = ['__background__',  # always index 0
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
               'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

    return hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file


def get_valid_roi(original_img, roi):
    roi[0] = min(original_img.shape[1] - 1, max(0, roi[0]))
    roi[1] = min(original_img.shape[0] - 1, max(0, roi[1]))
    roi[2] = min(original_img.shape[1] - 1, max(0, roi[2]))
    roi[3] = min(original_img.shape[0] - 1, max(0, roi[3]))
    return roi

def extract_features(paths):
    feature_type = 'resnet'
    input_h, input_w = 224, 224

    hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file = get_info(paths, feature_type)
    
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    
    image_list = list()
    with open(image_list_file) as f:
        for line in f.readlines():
            image_list.append(line.strip())

    feature_network = get_model(paths, feature_type)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    # Read detection results
    with open(det_res_path, 'r') as f:
        # detection results: [class_num][img_num][detection_num][x1, y1, x2, y2, score]
        det_res = pickle.load(f)

    total_idx = 0
    skip_idx = 0
    used_img_list = list()
    for i_image, img_name in enumerate(image_list):
        print(img_name)
        total_idx += 1
        # Extracted bounding boxes and classes
        det_boxes_all = np.empty((0, 4))
        det_classes_all = list()
        for c in range(1, len(classes)):
            for detection in det_res[c][i_image]:
                if detection[4] > 0.7:
                    det_boxes_all = np.vstack((det_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
                    det_classes_all.append(c)
        if len(det_classes_all) == 0:
            print(' skipping')
            skip_idx += 1
            continue
        used_img_list.append(img_name + '\n')
        edge_classes = list()
        for person_i, person_c in enumerate(det_classes_all):
            if person_c == 1:
                for obj_i, obj_c in enumerate(det_classes_all):
                    if obj_c == 1:
                        continue
                    combined_box = combine_box(det_boxes_all[person_i, :], det_boxes_all[obj_i, :])
                    det_boxes_all = np.vstack((det_boxes_all, combined_box))
                    edge_classes.append(0)
        det_classes_all.extend(edge_classes)

        # Get image feature by applying VGG to ROI (roi_vgg)
        image_path = os.path.join(hico_voc_path, 'JPEGImages', img_name + '.jpg')
        assert os.path.exists(image_path)
        original_img = scipy.misc.imread(image_path, mode='RGB')

        if feature_type == 'vgg':
            roi_features = np.zeros((det_boxes_all.shape[0], 4096))
        elif feature_type == 'resnet':
            roi_features = np.zeros((det_boxes_all.shape[0], 1000))
        elif feature_type == 'densenet':
            roi_features = np.zeros((det_boxes_all.shape[0], 1000))
        else:
            raise ValueError('feature type not recognized')

        for i_box in range(det_boxes_all.shape[0]):
            roi = det_boxes_all[i_box, :].astype(int)
            roi_image = original_img[roi[1]:roi[3]+1, roi[0]:roi[2]+1, :]
            # plt.imshow(roi_image)
            # plt.show()
            roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
            feature, _ = feature_network(roi_image)
            roi_features[i_box, ...] = feature.data.cpu().numpy()

        np.save(os.path.join(feature_path, '{}_classes'.format(img_name)), det_classes_all)
        np.save(os.path.join(feature_path, '{}_boxes'.format(img_name)), det_boxes_all)
        np.save(os.path.join(feature_path, '{}_features'.format(img_name)), roi_features)
    print(total_idx, skip_idx, total_idx - skip_idx)
    with open('/home/baoxiongjia/Projects/ECCV2018/trainvaltest.txt', 'w') as f:
        f.writelines(used_img_list)

def main():
    paths = hico_config.Paths()
    extract_features(paths)


if __name__ == '__main__':
    main()

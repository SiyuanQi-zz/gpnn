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

from pycocotools.coco import COCO
import vsrl_utils as vu
import vcoco_config
import roi_pooling
import feature_model
import metadata


def get_model(paths, feature_type):
    if feature_type == 'vgg':
        feature_network = feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif feature_type == 'resnet':
        feature_network = feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif feature_type == 'densenet':
        feature_network = feature_model.Densenet(num_classes=len(metadata.action_classes))
    else:
        raise ValueError('feature type not recognized')

    if feature_type.startswith('alexnet') or feature_type.startswith('vgg'):
        feature_network.features = torch.nn.DataParallel(feature_network.features)
        feature_network.cuda()
    else:
        feature_network = torch.nn.DataParallel(feature_network).cuda()

    checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'vcoco', 'finetune_{}'.format(feature_type))
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    checkpoint = torch.load(best_model_file)
    feature_network.load_state_dict(checkpoint['state_dict'])
    return feature_network


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths, imageset, feature_type):
    vcoco_feature_path = paths.data_root
    vcoco_path = os.path.join(vcoco_feature_path, '../v-coco')

    prefix = 'instances' if 'test' not in imageset else 'image_info'
    coco = COCO(os.path.join(vcoco_path, 'coco', 'annotations', prefix + '_' + imageset + '2014.json'))
    image_list = coco.getImgIds()
    image_list = coco.loadImgs(image_list)

    det_res_path = os.path.join('/home/siyuan/data/HICO/hico_20160224_det/Deformable-ConvNets/output/rfcn_dcn/vcoco/vcoco_detect2/{}2014'.format(imageset),
                                'COCO_{}2014_detections.pkl'.format(imageset))
    feature_path = os.path.join(vcoco_feature_path, 'features_{}'.format(feature_type))

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

    return vcoco_path, det_res_path, feature_path, classes, image_list


def extract_features(paths, imageset, vcoco_imageset):
    feature_type = 'resnet'
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    det_feature_path = os.path.join(paths.data_root, 'features_deformable')

    vcoco_path, det_res_path, feature_path, classes, image_list = get_info(paths, imageset, feature_type)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
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

    coco_from_vcoco = vu.load_coco()
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(vcoco_imageset))
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco_from_vcoco)
    vcoco_image_ids = vcoco_all[0]['image_id'][:, 0].astype(int)

    for i_image, img_info in enumerate(image_list):
        img_id = img_info['id']
        indices_in_vcoco = np.where(vcoco_image_ids == img_id)[0].tolist()
        if len(indices_in_vcoco) == 0:
            continue

        img_name = img_info['file_name']
        print(img_name)

        # # Extracted bounding boxes and classes
        # det_boxes_all = np.empty((0, 4))
        # det_classes_all = list()
        # for c in range(1, len(classes)):
        #     for detection in det_res[c][i_image]:
        #         if detection[4] > 0.7:
        #             det_boxes_all = np.vstack((det_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
        #             det_classes_all.append(c)
        # if len(det_classes_all) == 0:
        #     continue
        #
        # edge_classes = list()
        # for person_i, person_c in enumerate(det_classes_all):
        #     if person_c == 1:
        #         for obj_i, obj_c in enumerate(det_classes_all):
        #             if obj_c == 1:
        #                 continue
        #             combined_box = combine_box(det_boxes_all[person_i, :], det_boxes_all[obj_i, :])
        #             det_boxes_all = np.vstack((det_boxes_all, combined_box))
        #             edge_classes.append(0)
        # det_classes_all.extend(edge_classes)

        try:
            det_classes_all = np.load(os.path.join(det_feature_path, '{}_classes.npy'.format(img_name)))
            det_boxes_all = np.load(os.path.join(det_feature_path, '{}_boxes.npy'.format(img_name)))
        except IOError:
            continue

        # Read image
        image_path = os.path.join(vcoco_path, 'coco/images', '{}2014'.format(imageset), img_name)
        assert os.path.exists(image_path)
        original_img = scipy.misc.imread(image_path, mode='RGB')

        # Get image feature by applying network to ROI (roi_vgg)
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
            # if det_classes_all[i_box] < 0:
            #     plt.imshow(roi_image)
            #     plt.show()
            roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
            feature, _ = feature_network(roi_image)
            roi_features[i_box, ...] = feature.data.cpu().numpy()

        np.save(os.path.join(feature_path, '{}_classes'.format(img_name)), det_classes_all)
        np.save(os.path.join(feature_path, '{}_boxes'.format(img_name)), det_boxes_all)
        np.save(os.path.join(feature_path, '{}_features'.format(img_name)), roi_features)
        # break


def main():
    paths = vcoco_config.Paths()
    imagesets = [('val', 'test'), ('train', 'train'), ('train', 'val')]
    for imageset, vcoco_imageset in imagesets:
        extract_features(paths, imageset, vcoco_imageset)
        # break


if __name__ == '__main__':
    main()

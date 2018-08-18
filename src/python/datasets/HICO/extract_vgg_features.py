"""
Created on Oct 12, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle

import numpy as np
import scipy.misc
import cv2
import torch
import torch.autograd
import torchvision.models
import matplotlib.pyplot as plt

import hico_config
import roi_pooling


class Vgg16(torch.nn.Module):
    def __init__(self, last_layer=0, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential()
        for x in range(len(pretrained_vgg.features)):
            self.features.add_module(str(x), pretrained_vgg.features[x])

        self.classifier = torch.nn.Sequential()
        self.classifier.add_module(str(0), pretrained_vgg.classifier[0])
        # self.classifier.add_module(str(1), pretrained_vgg.classifier[1])
        # for x in range(len(pretrained_vgg.classifier)-last_layer):
        #     print pretrained_vgg.classifier[x]
        #     self.classifier.add_module(str(x), pretrained_vgg.classifier[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_info(paths):
    hico_path = paths.data_root
    hico_voc_path = os.path.join(hico_path, 'Deformable-ConvNets/data/hico/VOC2007')
    feature_path = os.path.join(hico_path, 'processed', 'features_roi_vgg')

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


def get_model():
    vgg16 = Vgg16(last_layer=1).cuda()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vgg16, transform


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def extract_features(paths):
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file = get_info(paths)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    image_list = list()
    with open(image_list_file) as f:
        for line in f.readlines():
            image_list.append(line.strip())

    vgg16, transform = get_model()

    # Read detection results
    with open(det_res_path, 'r') as f:
        # detection results: [class_num][img_num][detection_num][x1, y1, x2, y2, score]
        det_res = pickle.load(f)

    for i_image, img_name in enumerate(image_list):
        print(img_name)

        # Extracted bounding boxes and classes
        det_boxes_all = np.empty((0, 4))
        det_classes_all = list()
        for c in range(1, len(classes)):
            for detection in det_res[c][i_image]:
                if detection[4] > 0.7:
                    det_boxes_all = np.vstack((det_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
                    det_classes_all.append(c)
        if len(det_classes_all) == 0:
            continue

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
        # roi_features = np.empty((det_boxes_all.shape[0], 512, 7, 7))
        roi_features = np.empty((det_boxes_all.shape[0], 4096))
        for i_box in range(det_boxes_all.shape[0]):
            roi = det_boxes_all[i_box, :].astype(int)
            roi_image = original_img[roi[1]:roi[3]+1, roi[0]:roi[2]+1, :]
            roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
            roi_features[i_box, ...] = vgg16(roi_image).data.cpu().numpy()

        # # Get image feature by extracting ROI from VGG feature (vgg_roi)
        # image_path = os.path.join(hico_voc_path, 'JPEGImages', img_name + '.jpg')
        # assert os.path.exists(image_path)
        # original_img = scipy.misc.imread(image_path, mode='RGB')
        # scale_h = feature_size[0]/float(original_img.shape[0])
        # scale_w = feature_size[1]/float(original_img.shape[1])
        # transform_img = transform(cv2.resize(original_img, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
        # transform_img = torch.autograd.Variable(transform_img.unsqueeze(0)).cuda()
        # img_feature = vgg16(transform_img)
        #
        # roi_features = np.empty((det_boxes_all.shape[0], 512, 7, 7))
        # rois = np.copy(det_boxes_all)
        # rois[:, 0] *= scale_w
        # rois[:, 2] *= scale_w
        # rois[:, 1] *= scale_h
        # rois[:, 3] *= scale_h
        #
        # for i_box in range(rois.shape[0]):
        #     roi = rois[i_box, :].astype(int)
        #     roi_feature = adaptive_max_pool(img_feature[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)])
        #     roi_features[i_box, :, :, :] = roi_feature.data.cpu().numpy()

        np.save(os.path.join(feature_path, '{}_classes'.format(img_name)), det_classes_all)
        np.save(os.path.join(feature_path, '{}_boxes'.format(img_name)), det_boxes_all)
        np.save(os.path.join(feature_path, '{}_features'.format(img_name)), roi_features)


def main():
    paths = hico_config.Paths()
    extract_features(paths)
    import parse_features
    parse_features.main()


if __name__ == '__main__':
    main()

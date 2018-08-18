"""
Created on Feb 26, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import random

import numpy as np
import scipy.misc
import torch
import torchvision
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt

import metadata
import config

# TODO: Check if needed modification
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


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


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


def perturb_gt_box(box):
    new_box = box.copy()
    side1 = box[2] - box[0]
    side2 = box[3] - box[1]
    new_box = new_box + (np.random.rand(4) - 0.5) * np.array([side1, side2, side1, side2])/3
    return new_box


def perturb_box(box):
    #return box
    while True:
        new_box = perturb_gt_box(box)
        if compute_iou(new_box, box) > 0.7:
            return new_box


def get_valid_roi(original_img, roi):
    roi[0] = min(original_img.shape[1]-1, max(0, roi[0]))
    roi[1] = min(original_img.shape[0]-1, max(0, roi[1]))
    roi[2] = min(original_img.shape[1]-1, max(0, roi[2]))
    roi[3] = min(original_img.shape[0]-1, max(0, roi[3]))
    return roi



class HICO(torch.utils.data.Dataset):
    def __init__(self, root, input_imsize, transform, imageset):
        self.imageset = 'train' if imageset == 'train' else 'test'

        # result for deformable convnet? img feature
        anno_file = os.path.join(root, '{}_annotations.mat'.format(imageset))
        ld = sio.loadmat(anno_file)
        gt_anno = ld['gt_all']

        data = dict()
        data['img_ids'] = list()
        data['bbxs'] = list()
        data['actions'] = list()

        for hoi_idx, hoi in enumerate(gt_anno):
            for img_idx, bboxes in enumerate(hoi):
                if bboxes.size != 0:
                    for row in bboxes:
                        data['img_ids'].append(img_idx)
                        data['bbxs'].append(row)
                        data['actions'].append(metadata.hoi_to_action[hoi_idx])
            print('finished for ' + str(hoi_idx))
        np.save("anno_tune.npy", data)

        self.hico_path = root
        self.imsize = input_imsize
        self.transform = transform

        image_list = list()
        with open(os.path.join(config.Paths().project_root, '{}_all.txt'.format(imageset))) as f:
            for line in f.readlines():
                image_list.append(line.strip())

        self.img_files = [image_list[x] for x in data['img_ids']]
        self.bbxs = data['bbxs']
        self.actions = data['actions']

    def __getitem__(self, index):
        action_i, image_i = self.actions[index], self.img_files[index]
        bbxs = self.bbxs[index]
        h_bbx = bbxs[:4]
        o_bbx = bbxs[4:]
        perturbed_h_box = perturb_box(h_bbx)
        perturbed_o_box = perturb_box(o_bbx)
        roi = combine_box(perturbed_h_box, perturbed_o_box)

        roi = roi.astype(np.int)
        if 'test' in image_i:
            dir = 'test'
        else:
            dir = 'train'
        image_path = os.path.join(self.hico_path, 'images', '{}2015'.format(dir), '{}.jpg'.format(image_i))
        assert os.path.exists(image_path)

        original_img = scipy.misc.imread(image_path, mode='RGB')
        obj1 = original_img[h_bbx[1]:h_bbx[3]+1, h_bbx[0]:h_bbx[2]+1, :]
        obj2 = original_img[o_bbx[1]:o_bbx[3] + 1, o_bbx[0]:o_bbx[2] + 1, :]

        roi = get_valid_roi(original_img, roi)
        roi_image = original_img[roi[1]:roi[3]+1, roi[0]:roi[2]+1, :]

        # plt.imshow(roi_image)
        # plt.show()
        # print(metadata.action_classes[action_i])

        roi_image = cv2.resize(roi_image, self.imsize, interpolation=cv2.INTER_LINEAR)
        if random.random() > 0.5:
            roi_image = np.fliplr(roi_image).copy()

        roi_image = self.transform(roi_image)

        label = torch.LongTensor([action_i])

        return roi_image, label

    def __len__(self):
        return len(self.img_files)


class Vgg16(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(),
            # torch.nn.Linear(4096, num_classes),
            # torch.nn.Sigmoid()
        )
        self.last_layer = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.last_layer(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet152, self).__init__()
        # self.learn_modules = torch.nn.Sequential()
        # pretrained_resnet = torchvision.models.resnet152(pretrained=True)
        # for i, m in enumerate(pretrained_resnet.modules()):
        #     if isinstance(m, torch.nn.Linear):
        #         break
        #     self.learn_modules.add_module(str(i), m)
        self.learn_modules = torchvision.models.resnet152(pretrained=True)
        self.fc_ = torch.nn.Linear(1000, 200)
        self.fc = torch.nn.Linear(200, num_classes)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.ReLU(True),
        #     # torch.nn.Dropout(),
        #     torch.nn.Linear(1000, num_classes),
        # )
        # self._initialize_weights()

    def forward(self, x):
        x = self.learn_modules(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Densenet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Densenet, self).__init__()
        self.learn_modules = torchvision.models.densenet161(pretrained=True)
        self.fc = torch.nn.Linear(1000, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.learn_modules(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def main():
    # input_imsize = (244, 244)
    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                              std=[0.229, 0.224, 0.225])
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     normalize,
    # ])
    pass


if __name__ == '__main__':
    main()

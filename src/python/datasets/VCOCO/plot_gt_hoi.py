"""
Created on Mar 04, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import cv2

import numpy as np
import scipy.misc
import vsrl_utils as vu
import matplotlib.pyplot as plt

import vcoco_config


def plot_box_with_label(img, box, color, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, tuple(box[:2].tolist()), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, tuple(box[:2].tolist()), tuple(box[2:].tolist()), color)
    return img


def plot_set(paths, imageset):
    imageset = imageset
    vcoco_imageset = 'val' if imageset == 'test' else 'train'
    vcoco_path = os.path.join(paths.data_root, '..', 'v-coco')
    image_folder = os.path.join(vcoco_path, 'coco/images', '{}2014'.format(vcoco_imageset))
    result_folder = os.path.join(paths.tmp_root, 'results/VCOCO/detections/gt')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    coco = vu.load_coco()
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset))
    image_ids = vcoco_all[0]['image_id']
    image_info_list = coco.loadImgs(ids=image_ids[:, 0].tolist())
    image_ann_count = dict()

    for i_action, vcoco in enumerate(vcoco_all):
        vcoco = vu.attach_gt_boxes(vcoco, coco)
        action_name = vcoco['action_name']
        positive_indices = np.where(vcoco['label'] == 1)[0].tolist()
        for image_i in positive_indices:
            # img_id = vcoco['image_id'][image_i, 0]
            img_name = image_info_list[image_i]['file_name']
            image_path = os.path.join(image_folder, img_name)
            assert os.path.exists(image_path)
            img = scipy.misc.imread(image_path, mode='RGB')

            role_bbox = vcoco['role_bbox'][image_i, :] * 1.
            role_bbox = role_bbox.reshape((-1, 4))
            plot_box_with_label(img, role_bbox[0, :].astype(int), (255, 0, 0), action_name)
            for j in range(1, len(vcoco['role_name'])):
                if not np.isnan(role_bbox[j, 0]):
                    role = vcoco['role_name'][j]
                    plot_box_with_label(img, role_bbox[j, :].astype(int), (0, 255, 0), role)

            if img_name not in image_ann_count:
                image_ann_count[img_name] = 0
            else:
                image_ann_count[img_name] += 1

            # plot ground truth annotation
            plt.imshow(img)
            plt.axis('off')
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            filename, ext = os.path.splitext(img_name)
            plt.savefig(os.path.join(result_folder, '{}_gt_{:02d}{}'.format(filename, image_ann_count[img_name], ext)),
                        bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            # copy original image file
            shutil.copy(image_path, os.path.join(result_folder, img_name))


def main():
    paths = vcoco_config.Paths()
    imagesets = ['test']
    for imageset in imagesets:
        plot_set(paths, imageset)


if __name__ == '__main__':
    main()

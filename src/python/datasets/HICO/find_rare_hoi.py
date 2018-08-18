"""
Created on Oct 22, 2017

@author: Siyuan Qi

Description of the file.

"""

import os

import numpy as np
import scipy.io

import hico_config


def collect_hoi_stats(bbox):
    stats = np.zeros(600)
    for idx in range(bbox['filename'].shape[1]):
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]
            stats[int(hoi_id)-1] += 1

    return stats


def split_testing_set(paths, bbox, stats):
    feature_path = os.path.join(paths.data_root, 'processed', 'features_background_49')

    rare_set = list()
    non_rare_set = list()
    for idx in range(bbox['filename'].shape[1]):
        filename = str(bbox['filename'][0, idx][0])
        filename = os.path.splitext(filename)[0] + '\n'

        try:
            det_classes = np.load(os.path.join(feature_path, '{}_classes.npy'.format(filename.strip())))
        except IOError:
            continue

        rare = False
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]

            if stats[int(hoi_id)-1] < 10:
                rare_set.append(filename)
                rare = True
                continue
        if not rare:
            non_rare_set.append(filename)

    with open(os.path.join(paths.tmp_root, 'hico', 'test_rare.txt'), 'w') as f:
        f.writelines(rare_set)

    with open(os.path.join(paths.tmp_root, 'hico', 'test_non_rare.txt'), 'w') as f:
        f.writelines(non_rare_set)


def find_rare_hoi(paths):
    anno_bbox = scipy.io.loadmat(os.path.join(paths.data_root, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']
    list_action = anno_bbox['list_action']

    stats = collect_hoi_stats(bbox_train)
    split_testing_set(paths, bbox_test, stats)


def main():
    paths = hico_config.Paths()
    find_rare_hoi(paths)


if __name__ == '__main__':
    main()

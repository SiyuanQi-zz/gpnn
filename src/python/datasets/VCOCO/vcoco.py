"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle
import argparse
import warnings

import torch.utils.data
import numpy as np
import vsrl_utils as vu

import vcoco_config


class VCOCO(torch.utils.data.Dataset):
    def __init__(self, root, imageset):
        self.root = root
        self.coco = vu.load_coco()
        vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset))
        self.image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
        self.unique_image_ids = list(set(self.image_ids))

        # action_role = dict()
        # for i, x in enumerate(vcoco_all):
        #     action_role[x['action_name']] = x['role_name']
        # print 'action_role', action_role

    def __getitem__(self, index):
        img_name = self.coco.loadImgs(ids=[self.unique_image_ids[index]])[0]['file_name']
        try:
            data = pickle.load(open(os.path.join(self.root, '{}.p'.format(img_name)), 'rb'))
            edge_features = np.load(os.path.join(self.root, '{}_edge_features.npy').format(img_name))
            node_features = np.load(os.path.join(self.root, '{}_node_features.npy').format(img_name))
        except IOError:
            # warnings.warn('data missing for {}'.format(img_name))
            return self.__getitem__(0)

        img_id = data['img_id']
        adj_mat = data['adj_mat']
        node_labels = data['node_labels']
        node_roles = data['node_roles']
        boxes = data['boxes']
        human_num = data['human_num']
        obj_num = data['obj_num']
        classes = data['classes']
        return edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes

    def __len__(self):
        return len(self.unique_image_ids)


def main(args):
    start_time = time.time()

    subset = ['train', 'val', 'test']
    training_set = VCOCO(os.path.join(args.data_root, 'processed'), subset[0])
    print('{} instances.'.format(len(training_set)))
    edge_features, node_features, adj_mat, node_labels, node_roles, img_name = training_set[0]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def parse_arguments():
    paths = vcoco_config.Paths()
    parser = argparse.ArgumentParser(description='V-COCO dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

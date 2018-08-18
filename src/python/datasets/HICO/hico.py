"""
Created on Oct 02, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle
import argparse

import numpy as np
import torch.utils.data

import hico_config


class HICO(torch.utils.data.Dataset):
    def __init__(self, root, sequence_ids):
        self.root = root
        self.sequence_ids = sequence_ids

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        data = pickle.load(open(os.path.join(self.root, '{}.p'.format(sequence_id)), 'rb'))
        # edge_features = data['edge_features']
        # node_features = data['node_features']

        det_classes = data['classes']
        det_boxes = data['boxes']
        human_num = data['human_num']
        obj_num = data['obj_num']

        edge_features = np.load(os.path.join(self.root, '{}_edge_features.npy').format(sequence_id))
        node_features = np.load(os.path.join(self.root, '{}_node_features.npy').format(sequence_id))
        adj_mat = data['adj_mat']
        node_labels = data['node_labels']

        return edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num

    def __len__(self):
        return len(self.sequence_ids)


def main(args):
    start_time = time.time()

    subset = ['train', 'val', 'test']
    hico_voc_path = os.path.join(args.data_root, 'Deformable-ConvNets/data/hico/VOC2007')
    with open(os.path.join(hico_voc_path, 'ImageSets/Main', '{}.txt'.format(subset[0]))) as f:
        filenames = [line.strip() for line in f.readlines()]

    training_set = HICO(args.tmp_root, filenames[:5])
    print('{} instances.'.format(len(training_set)))
    edge_features, node_features, adj_mat, node_labels = training_set[0]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def parse_arguments():
    paths = hico_config.Paths()
    parser = argparse.ArgumentParser(description='HICO dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

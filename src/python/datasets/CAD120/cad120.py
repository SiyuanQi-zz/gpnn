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

import cad120_config
import metadata


class CAD120(torch.utils.data.Dataset):
    features = None

    def __init__(self, feature_data_path, sequence_ids):
        if not self.__class__.features:
            self.__class__.features = pickle.load(open(feature_data_path, 'rb'))
        self.data = list()
        self.sequence_ids = list()
        for sequence_id, sequence_features in self.__class__.features.items():
            if sequence_id in sequence_ids:
                self.data.extend(sequence_features)
                self.sequence_ids.extend([sequence_id for _ in range(len(sequence_features))])

        self.max_node_label_len = np.max([len(metadata.subactivities), len(metadata.affordances)])

    def __getitem__(self, index):
        edge_features = self.data[index]['edge_features']
        node_features = self.data[index]['node_features']
        adj_mat = self.data[index]['adj_mat']
        node_labels = self.data[index]['node_labels'].astype(np.int32)

        node_num = node_labels.shape[0]
        one_hot_node_labels = np.zeros((node_num, self.max_node_label_len))
        for v in range(node_num):
            one_hot_node_labels[v, node_labels[v]] = 1

        return np.transpose(edge_features, (2, 0, 1)), np.transpose(node_features, (1, 0)), adj_mat, one_hot_node_labels, self.sequence_ids[index]

    def __len__(self):
        return len(self.data)


def main(args):
    start_time = time.time()
    sequence_ids = pickle.load(open(os.path.join(args.tmp_root, 'cad120_data_list.p'), 'rb'))
    train_num = 80
    val_num = 20
    test_num = 25
    sequence_ids = np.random.permutation(sequence_ids)

    training_set = CAD120(args.tmp_root, sequence_ids[:train_num])
    print('{} instances.'.format(len(training_set)))
    edge_features, node_features, adj_mat, node_labels = training_set[0]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def parse_arguments():
    paths = cad120_config.Paths()
    parser = argparse.ArgumentParser(description='CAD 120 dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

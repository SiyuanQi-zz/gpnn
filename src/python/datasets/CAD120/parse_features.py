"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle

import numpy as np

import cad120_config
import metadata


def parse_colon_seperated_features(colon_seperated):
    f_list = [int(x.split(':')[1]) for x in colon_seperated]
    return f_list


def read_features(segments_feature_path, filename):
    data = dict()
    filename_base = os.path.basename(filename)
    sequence_id = filename_base.split('_')[0]
    segment_index = int(os.path.splitext(filename_base)[0].split('_')[1])

    # Spatial features
    with open(filename) as f:
        first_line = f.readline().strip()
        object_num = int(first_line.split(' ')[0])
        object_object_num = int(first_line.split(' ')[1])
        skeleton_object_num = int(first_line.split(' ')[2])

        edge_features = np.zeros((1+object_num, 1+object_num, 800))
        node_features = np.zeros((1+object_num, 810))
        adj_mat = np.zeros((1+object_num, 1+object_num)) if segment_index == 1 else np.eye(1+object_num)
        node_labels = np.ones((1+object_num)) * -1

        stationary_index = metadata.affordance_index['stationary']
        null_index = metadata.subactivity_index['null']

        # Object feature
        for _ in range(object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o_id = int(colon_seperated[1])
            node_labels[o_id] = int(colon_seperated[0]) - 1
            node_features[o_id, 630:] = np.array(parse_colon_seperated_features(colon_seperated[2:]))

        # Skeleton feature
        line = f.readline()
        colon_seperated = [x.strip() for x in line.strip().split(' ')]
        node_labels[0] = int(colon_seperated[0]) - 1
        node_features[0, :630] = parse_colon_seperated_features(colon_seperated[2:])

        # Object-object feature
        for _ in range(object_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o1_id, o2_id = int(colon_seperated[2]), int(colon_seperated[3])

            if int(node_labels[o1_id]) != stationary_index and int(node_labels[o2_id]) != stationary_index:
                adj_mat[o1_id, o2_id] = 1
            edge_features[o1_id, o2_id, 400:600] = parse_colon_seperated_features(colon_seperated[4:])

        # Skeleton-object feature
        for _ in range(skeleton_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            s_o_id = int(colon_seperated[2])
            edge_features[0, s_o_id, :400] = parse_colon_seperated_features(colon_seperated[3:])
            edge_features[s_o_id, 0, :400] = edge_features[0, s_o_id, :400]

            if int(node_labels[0]) != null_index and int(node_labels[s_o_id]) != stationary_index:
                adj_mat[0, s_o_id] = 1
                adj_mat[s_o_id, 0] = 1

    # Temporal features
    if segment_index == 1:
        for node_i in range(edge_features.shape[0]):
            edge_features[node_i, node_i, 600:] = 0
    else:
        with open(os.path.join(segments_feature_path, '{}_{}_{}.txt'.format(sequence_id, segment_index-1, segment_index)), 'r') as f:
            first_line = f.readline().strip()
            object_object_num = int(first_line.split(' ')[0])
            skeleton_skeleton_num = int(first_line.split(' ')[1])
            assert skeleton_skeleton_num == 1

            # Object-object temporal feature
            for _ in range(object_object_num):
                line = f.readline()
                colon_seperated = [x.strip() for x in line.strip().split(' ')]
                o_id = int(colon_seperated[2])
                edge_features[o_id, o_id, 760:] = np.array(parse_colon_seperated_features(colon_seperated[3:]))

            # Skeleton-object temporal feature
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            node_features[0, 600:760] = parse_colon_seperated_features(colon_seperated[3:])

    # Return data as a dictionary
    data['edge_features'] = edge_features
    data['node_features'] = node_features
    data['adj_mat'] = adj_mat
    data['node_labels'] = node_labels

    return data


def collect_data(paths):
    if not os.path.exists(paths.tmp_root):
        os.makedirs(paths.tmp_root)
    segments_files_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'segments_svm_format')
    segments_feature_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'features_binary_svm_format')

    data = dict()
    sequence_ids = list()
    # date_selection = ['1204142227', '0510175411']
    for sequence_path_file in os.listdir(segments_files_path):
        sequence_id = os.path.splitext(sequence_path_file)[0]
        # if sequence_id not in date_selection:
        #     continue
        data[sequence_id] = list()
        sequence_ids.append(sequence_id)

        with open(os.path.join(segments_files_path, sequence_path_file)) as f:
            first_line = f.readline()
            segment_feature_num = int(first_line.split(' ')[0])

            for _ in range(segment_feature_num):
                segment_feature_filename = f.readline().strip()
                segment_data = read_features(segments_feature_path, os.path.join(segments_feature_path, os.path.basename(segment_feature_filename)))
                data[sequence_id].append(segment_data)

    pickle.dump(data, open(os.path.join(paths.tmp_root, 'cad120_data.p'), 'wb'))
    pickle.dump(sequence_ids, open(os.path.join(paths.tmp_root, 'cad120_data_list.p'), 'wb'))


def main():
    paths = cad120_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

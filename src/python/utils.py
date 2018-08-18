"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle
import itertools

import numpy as np
import torch
import torch.utils.data
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2

import datasets
import visualization_utils
# import vsrl_eval
# import vsrl_utils as vu


def to_variable(v, use_cuda):
    if use_cuda:
        v = v.cuda()
    return torch.autograd.Variable(v)


def get_cad_data(args, prediction=False):
    sequence_ids = pickle.load(open(os.path.join(args.tmp_root, 'cad120', 'cad120_data_list.p'), 'rb'))
    train_num, val_num, test_num = 80, 20, 25
    sequence_ids = np.random.permutation(sequence_ids)

    if prediction:
        data_path = os.path.join(args.tmp_root, 'cad120', 'cad120_data_prediction.p')
    else:
        data_path = os.path.join(args.tmp_root, 'cad120', 'cad120_data.p')

    training_set = datasets.CAD120(data_path, sequence_ids[:train_num])
    valid_set = datasets.CAD120(data_path, sequence_ids[train_num:train_num+val_num])
    testing_set = datasets.CAD120(data_path, sequence_ids[-test_num:])

    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=datasets.utils.collate_fn_cad,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, collate_fn=datasets.utils.collate_fn_cad,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=datasets.utils.collate_fn_cad,
                                              batch_size=args.batch_size,
                                              num_workers=args.prefetch, pin_memory=True)
    print('Dataset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(valid_loader), len(test_loader)))
    return training_set, valid_set, testing_set, train_loader, valid_loader, test_loader


def get_label_bar(labels, height=10, width=50):
    label_bar = np.empty((height, width*len(labels)))
    for i, label in enumerate(labels):
        label_bar[:, i*width:(i+1)*width] = label
    return label_bar


def plot_segmentation(labels_list, classes, save_path=None):
    bar_height, bar_width = 10, 50

    fig = plt.figure(figsize=(len(classes), 1))
    gridspec = matplotlib.gridspec.GridSpec(len(labels_list), 1)
    gridspec.update(wspace=0.5, hspace=0.01)  # set the spacing between axes.
    for plt_idx, labels in enumerate(labels_list):
        label_bar = get_label_bar(labels, height=bar_height, width=bar_width)
        ax = plt.subplot(gridspec[plt_idx])
        plt.imshow(label_bar, vmin=0, vmax=len(classes), cmap=plt.get_cmap('hsv'))
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ticks = [classes[label] for label in labels_list[plt_idx]]
        ax.set_xticks([i*bar_width for i in range(len(labels))])
        ax.set_xticklabels(ticks, ha='left')
        ax.set_yticklabels([])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    else:
        plt.show()


def plot_all_activity_segmentations(all_sequence_ids, subact_predictions, subact_ground_truth, result_folder):
    result_folder = os.path.join(result_folder, 'segmentation')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    last_sequence_id = None
    seq_subact_pred = list()
    seq_subact_gt = list()
    for i, sequence_id in enumerate(all_sequence_ids):
        if sequence_id != last_sequence_id:
            if len(seq_subact_pred) > 0:
                plot_segmentation([seq_subact_gt, seq_subact_pred], classes=datasets.cad_metadata.subactivities, save_path=os.path.join(result_folder, '{}_action.png'.format(last_sequence_id)))
            last_sequence_id = sequence_id
            seq_subact_pred = list()
            seq_subact_gt = list()
        seq_subact_pred.append(subact_predictions[i])
        seq_subact_gt.append(subact_ground_truth[i])

    if len(seq_subact_pred) > 0:
        plot_segmentation([seq_subact_gt, seq_subact_pred], classes=datasets.cad_metadata.subactivities,
                          save_path=os.path.join(result_folder, '{}_action.png'.format(last_sequence_id)))


def plot_all_affordance_segmentations(all_sequence_ids, all_node_nums, aff_predictions, aff_ground_truth, result_folder):
    result_folder = os.path.join(result_folder, 'segmentation')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    last_sequence_id = None
    previous_obj_num = 0
    obj_num = 0
    seq_aff_pred = list()
    seq_aff_gt = list()
    for i, sequence_id in enumerate(all_sequence_ids):
        if sequence_id != last_sequence_id:
            if len(seq_aff_pred) > 0:
                for obj_i in range(obj_num):
                    plot_segmentation([seq_aff_gt[obj_i], seq_aff_pred[obj_i]], classes=datasets.cad_metadata.affordances, save_path=os.path.join(result_folder, '{}_affordance_{}.png'.format(last_sequence_id, obj_i)))
            last_sequence_id = sequence_id
            obj_num = all_node_nums[i]-1
            seq_aff_pred = [list() for _ in range(obj_num)]
            seq_aff_gt = [list() for _ in range(obj_num)]
        for obj_i in range(obj_num):
            seq_aff_pred[obj_i].append(aff_predictions[previous_obj_num+obj_i])
            seq_aff_gt[obj_i].append(aff_ground_truth[previous_obj_num+obj_i])
        previous_obj_num += obj_num

    if len(seq_aff_pred) > 0:
        for obj_i in range(obj_num):
            plot_segmentation([seq_aff_gt[obj_i], seq_aff_pred[obj_i]], classes=datasets.cad_metadata.affordances,
                              save_path=os.path.join(result_folder, '{}_action_{}.png'.format(last_sequence_id, obj_i)))


def plot_confusion_matrix(cm, classes, filename=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    ax = plt.gca()
    ax.tick_params(axis=u'both', which=u'both', length=0)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]), verticalalignment='center', horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def get_hico_data(args):
    np.random.seed(0)
    validation_ratio = 0.25
    with open(os.path.join(args.tmp_root, 'hico', 'trainval.txt')) as f:
        filenames = [line.strip() for line in f.readlines()]
        filenames = np.random.permutation(filenames)
        train_filenames = filenames[int(len(filenames)*validation_ratio):]
        val_filenames = filenames[:int(len(filenames)*validation_ratio)]

    with open(os.path.join(args.tmp_root, 'hico', 'test.txt')) as f:
        test_filenames = [line.strip() for line in f.readlines()]
    with open(os.path.join(args.tmp_root, 'hico', 'img_index.txt')) as f:
        img_index = [line.strip() for line in f.readlines()]

    root = os.path.join(args.data_root, 'processed', 'hico_data_background_49')
    training_set = datasets.HICO(root, train_filenames[:])
    valid_set = datasets.HICO(root, val_filenames[:])
    testing_set = datasets.HICO(root, test_filenames[:])

    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=datasets.utils.collate_fn_hico,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, collate_fn=datasets.utils.collate_fn_hico,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=datasets.utils.collate_fn_hico,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    print('Dataset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(valid_loader), len(test_loader)))
    return training_set, valid_set, testing_set, train_loader, valid_loader, test_loader, img_index


def get_vcoco_data(args):
    root = os.path.join(args.data_root, 'processed', args.feature_type)
    training_set = datasets.VCOCO(root, 'train')
    valid_set = datasets.VCOCO(root, 'val')
    testing_set = datasets.VCOCO(root, 'test')
    train_loader = torch.utils.data.DataLoader(training_set, collate_fn=datasets.utils.collate_fn_vcoco,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, collate_fn=datasets.utils.collate_fn_vcoco,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_set, collate_fn=datasets.utils.collate_fn_vcoco,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    print('Dataset sizes: {} training, {} validation, {} testing.'.format(len(train_loader), len(valid_loader), len(test_loader)))
    return training_set, valid_set, testing_set, train_loader, valid_loader, test_loader


def parse_result(result, coco, img_dir, result_folder, img_result_id):
    img_id = result['image_id']
    img_name = coco.loadImgs(ids=[img_id])[0]['file_name']
    image_path = os.path.join(img_dir, img_name)
    assert os.path.exists(image_path)
    img = scipy.misc.imread(image_path, mode='RGB')

    for k, v in result.items():
        if k.endswith('agent'):
            person_box = result['person_box']
            action = k.split('_')[0]
            action_score = v
            if action_score < 0.5:
                return img_name, False
            label = '{}: {:.2f}'.format(action, action_score)
            plot_box_with_label(img, person_box.astype(int), (255, 0, 0), label)

    for role, color in zip(['obj', 'instr'], [(0, 255, 0), (0, 0, 255)]):
        action_role_key = '{}_{}'.format(action, role)
        if action_role_key in result:
            obj_box = result[action_role_key][:4]
            hoi_score = result[action_role_key][4]
            obj_name = result['{}_class'.format(role)]
            if obj_name < 0:
                continue
            obj_name = datasets.vcoco_metadata.coco_classes[obj_name]
            label = '{} as {}: {:.2f}'.format(obj_name, role, hoi_score)
            plot_box_with_label(img, obj_box.astype(int), color, label)

    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    filename, ext = os.path.splitext(img_name)
    plt.savefig(os.path.join(result_folder, '{}_result_{:02d}{}'.format(filename, img_result_id, ext)), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    return img_name, True


def plot_box_with_label(img, box, color, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, tuple(box[:2].tolist()), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, tuple(box[:2].tolist()), tuple(box[2:].tolist()), color)
    return img


def visualize_vcoco_result(args, result_folder, all_results):
    coco = vu.load_coco()
    img_dir = os.path.join(args.data_root, '../v-coco/coco/images', 'val2014')
    last_img_name = ''
    img_result_id = 0
    for result in all_results:
        img_name, saved = parse_result(result, coco, img_dir, result_folder, img_result_id)
        if img_name != last_img_name:
            last_img_name = img_name
            img_result_id = 0
        elif saved:
            img_result_id += 1


def main():
    pass


if __name__ == '__main__':
    main()

"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

coco_classes = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
action_class_ins = [500, 2163, 2448, 1054, 236, 356, 2055, 203, 376, 381, 228, 187,  262,  141,  161, 173, 317, 248, 310, 259, 269, 867, 70, 77, 23, 65, 212]
assert len(action_classes) == len(action_class_ins)
action_class_weight = [2000.0/float(ins) for ins in action_class_ins]

action_no_obj = ['point', 'run', 'smile', 'stand', 'walk']
action_one_obj = {'hold': 'obj', 'look': 'obj', 'carry': 'obj', 'throw': 'obj', 'catch': 'obj', 'kick': 'obj',
                  'read': 'obj',
                  'sit': 'instr', 'ride': 'instr', 'jump': 'instr', 'lay': 'instr', 'talk_on_phone': 'instr',
                  'work_on_computer': 'instr', 'ski': 'instr', 'surf': 'instr', 'skateboard': 'instr', 'drink': 'instr',
                  'point': 'instr', 'snowboard': 'instr'}

roles = ['none', 'obj', 'instr']
action_roles = {'point': ['agent', 'instr'], 'walk': ['agent'], 'jump': ['agent', 'instr'], 'snowboard': ['agent', 'instr'], 'carry': ['agent', 'obj'], 'run': ['agent'], 'work_on_computer': ['agent', 'instr'], 'cut': ['agent', 'instr', 'obj'], 'sit': ['agent', 'instr'], 'eat': ['agent', 'obj', 'instr'], 'talk_on_phone': ['agent', 'instr'], 'smile': ['agent'], 'ski': ['agent', 'instr'], 'kick': ['agent', 'obj'], 'surf': ['agent', 'instr'], 'hit': ['agent', 'instr', 'obj'], 'read': ['agent', 'obj'], 'drink': ['agent', 'instr'], 'lay': ['agent', 'instr'], 'catch': ['agent', 'obj'], 'hold': ['agent', 'obj'], 'throw': ['agent', 'obj'], 'look': ['agent', 'obj'], 'ride': ['agent', 'instr'], 'skateboard': ['agent', 'instr'], 'stand': ['agent']}


action_index = dict()
for a in action_classes:
    action_index[a] = action_classes.index(a)

role_index = dict()
for r in roles:
    role_index[r] = roles.index(r)


def main():
    pass


if __name__ == '__main__':
    main()

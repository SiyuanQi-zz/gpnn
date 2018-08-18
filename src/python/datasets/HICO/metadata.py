"""
Created on Oct 02, 2017

@author: Siyuan Qi

Description of the file.

"""


hico_classes = ['__background__',  # always index 0
                'airplane', 'apple', 'backpack', 'banana', 'baseball_bat', 'baseball_glove', 'bear', 'bed', 'bench',
                'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat',
                'cell_phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining_table', 'dog', 'donut', 'elephant',
                'fire_hydrant', 'fork', 'frisbee', 'giraffe', 'hair_drier', 'handbag', 'horse', 'hot_dog', 'keyboard',
                'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking_meter',
                'person', 'pizza', 'potted_plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink',
                'skateboard', 'skis', 'snowboard', 'spoon', 'sports_ball', 'stop_sign', 'suitcase', 'surfboard',
                'teddy_bear', 'tennis_racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic_light', 'train',
                'truck', 'tv', 'umbrella', 'vase', 'wine_glass', 'zebra']

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

coco_to_hico = [hico_classes.index(c) for c in coco_classes]
hico_to_coco = [coco_classes.index(c) for c in hico_classes]

hoi_classes = ['board', 'direct', 'exit', 'fly', 'inspect', 'load', 'ride', 'sit_on', 'wash', 'no_interaction',
                'carry', 'hold', 'inspect', 'jump', 'hop_on', 'park', 'push', 'repair', 'ride', 'sit_on', 'straddle',
                'walk', 'wash', 'no_interaction', 'chase', 'feed', 'hold', 'pet', 'release', 'watch', 'no_interaction',
                'board', 'drive', 'exit', 'inspect', 'jump', 'launch', 'repair', 'ride', 'row', 'sail', 'sit_on',
                'stand_on', 'tie', 'wash', 'no_interaction', 'carry', 'drink_with', 'hold', 'inspect', 'lick', 'open',
               'pour', 'no_interaction', 'board', 'direct', 'drive', 'exit', 'inspect', 'load', 'ride', 'sit_on',
               'wash', 'wave', 'no_interaction', 'board', 'direct', 'drive', 'hose', 'inspect', 'jump', 'load', 'park',
               'ride', 'wash', 'no_interaction', 'dry', 'feed', 'hold', 'hug', 'kiss', 'pet', 'scratch', 'wash',
               'chase', 'no_interaction', 'carry', 'hold', 'lie_on', 'sit_on', 'stand_on', 'no_interaction', 'carry',
               'lie_on', 'sit_on', 'no_interaction', 'feed', 'herd', 'hold', 'hug', 'kiss', 'lasso', 'milk', 'pet',
               'ride', 'walk', 'no_interaction', 'clean', 'eat_at', 'sit_at', 'no_interaction', 'carry', 'dry', 'feed',
               'groom', 'hold', 'hose', 'hug', 'inspect', 'kiss', 'pet', 'run', 'scratch', 'straddle', 'train', 'walk',
               'wash', 'chase', 'no_interaction', 'feed', 'groom', 'hold', 'hug', 'jump', 'kiss', 'load', 'hop_on',
               'pet', 'race', 'ride', 'run', 'straddle', 'train', 'walk', 'wash', 'no_interaction', 'hold', 'inspect',
               'jump', 'hop_on', 'park', 'push', 'race', 'ride', 'sit_on', 'straddle', 'turn', 'walk', 'wash',
               'no_interaction', 'carry', 'greet', 'hold', 'hug', 'kiss', 'stab', 'tag', 'teach', 'lick',
               'no_interaction', 'carry', 'hold', 'hose', 'no_interaction', 'carry', 'feed', 'herd', 'hold', 'hug',
               'kiss', 'pet', 'ride', 'shear', 'walk', 'wash', 'no_interaction', 'board', 'drive', 'exit', 'load',
               'ride', 'sit_on', 'wash', 'no_interaction', 'control', 'repair', 'watch', 'no_interaction', 'buy',
               'cut', 'eat', 'hold', 'inspect', 'peel', 'pick', 'smell', 'wash', 'no_interaction', 'carry', 'hold',
               'inspect', 'open', 'wear', 'no_interaction', 'buy', 'carry', 'cut', 'eat', 'hold', 'inspect', 'peel',
               'pick', 'smell', 'no_interaction', 'break', 'carry', 'hold', 'sign', 'swing', 'throw', 'wield',
               'no_interaction', 'hold', 'wear', 'no_interaction', 'feed', 'hunt', 'watch', 'no_interaction', 'clean',
               'lie_on', 'sit_on', 'no_interaction', 'inspect', 'lie_on', 'sit_on', 'no_interaction', 'carry', 'hold',
               'open', 'read', 'no_interaction', 'hold', 'stir', 'wash', 'lick', 'no_interaction', 'cut', 'eat',
               'hold', 'smell', 'stir', 'wash', 'no_interaction', 'blow', 'carry', 'cut', 'eat', 'hold', 'light',
               'make', 'pick_up', 'no_interaction', 'carry', 'cook', 'cut', 'eat', 'hold', 'peel', 'smell', 'stir',
               'wash', 'no_interaction', 'carry', 'hold', 'read', 'repair', 'talk_on', 'text_on', 'no_interaction',
               'check', 'hold', 'repair', 'set', 'no_interaction', 'carry', 'drink_with', 'hold', 'inspect', 'pour',
               'sip', 'smell', 'fill', 'wash', 'no_interaction', 'buy', 'carry', 'eat', 'hold', 'make', 'pick_up',
               'smell', 'no_interaction', 'feed', 'hold', 'hose', 'hug', 'kiss', 'hop_on', 'pet', 'ride', 'walk',
               'wash', 'watch', 'no_interaction', 'hug', 'inspect', 'open', 'paint', 'no_interaction', 'hold', 'lift',
               'stick', 'lick', 'wash', 'no_interaction', 'block', 'catch', 'hold', 'spin', 'throw', 'no_interaction',
               'feed', 'kiss', 'pet', 'ride', 'watch', 'no_interaction', 'hold', 'operate', 'repair', 'no_interaction',
               'carry', 'hold', 'inspect', 'no_interaction', 'carry', 'cook', 'cut', 'eat', 'hold', 'make',
               'no_interaction', 'carry', 'clean', 'hold', 'type_on', 'no_interaction', 'assemble', 'carry', 'fly',
               'hold', 'inspect', 'launch', 'pull', 'no_interaction', 'cut_with', 'hold', 'stick', 'wash', 'wield',
               'lick', 'no_interaction', 'hold', 'open', 'read', 'repair', 'type_on', 'no_interaction', 'clean',
               'open', 'operate', 'no_interaction', 'control', 'hold', 'repair', 'no_interaction', 'buy', 'cut', 'eat',
               'hold', 'inspect', 'peel', 'pick', 'squeeze', 'wash', 'no_interaction', 'clean', 'hold', 'inspect',
               'open', 'repair', 'operate', 'no_interaction', 'check', 'pay', 'repair', 'no_interaction', 'buy',
               'carry', 'cook', 'cut', 'eat', 'hold', 'make', 'pick_up', 'slide', 'smell', 'no_interaction', 'clean',
               'hold', 'move', 'open', 'no_interaction', 'hold', 'point', 'swing', 'no_interaction', 'carry', 'cook',
               'cut', 'eat', 'hold', 'make', 'no_interaction', 'cut_with', 'hold', 'open', 'no_interaction', 'clean',
               'repair', 'wash', 'no_interaction', 'carry', 'flip', 'grind', 'hold', 'jump', 'pick_up', 'ride',
               'sit_on', 'stand_on', 'no_interaction', 'adjust', 'carry', 'hold', 'inspect', 'jump', 'pick_up',
               'repair', 'ride', 'stand_on', 'wear', 'no_interaction', 'adjust', 'carry', 'grind', 'hold', 'jump',
               'ride', 'stand_on', 'wear', 'no_interaction', 'hold', 'lick', 'wash', 'sip', 'no_interaction', 'block',
               'carry', 'catch', 'dribble', 'hit', 'hold', 'inspect', 'kick', 'pick_up', 'serve', 'sign', 'spin',
               'throw', 'no_interaction', 'hold', 'stand_under', 'stop_at', 'no_interaction', 'carry', 'drag', 'hold',
               'hug', 'load', 'open', 'pack', 'pick_up', 'zip', 'no_interaction', 'carry', 'drag', 'hold', 'inspect',
               'jump', 'lie_on', 'load', 'ride', 'stand_on', 'sit_on', 'wash', 'no_interaction', 'carry', 'hold',
               'hug', 'kiss', 'no_interaction', 'carry', 'hold', 'inspect', 'swing', 'no_interaction', 'adjust', 'cut',
               'hold', 'inspect', 'pull', 'tie', 'wear', 'no_interaction', 'hold', 'operate', 'repair',
               'no_interaction', 'clean', 'flush', 'open', 'repair', 'sit_on', 'stand_on', 'wash', 'no_interaction',
               'brush_with', 'hold', 'wash', 'no_interaction', 'install', 'repair', 'stand_under', 'stop_at',
               'no_interaction', 'direct', 'drive', 'inspect', 'load', 'repair', 'ride', 'sit_on', 'wash',
               'no_interaction', 'carry', 'hold', 'lose', 'open', 'repair', 'set', 'stand_under', 'no_interaction',
               'hold', 'make', 'paint', 'no_interaction', 'fill', 'hold', 'sip', 'toast', 'lick', 'wash',
               'no_interaction', 'feed', 'hold', 'pet', 'watch', 'no_interaction']

action_classes = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch',
                   'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble',
                   'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
                   'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect',
                   'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load',
                  'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
                  'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release',
                   'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at',
                   'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick', 'stir',
                   'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast',
                   'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']


hoi_to_action = [action_classes.index(c) for c in hoi_classes]


obj_hoi_index = [(0, 0), (161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576),
                 (32, 46), (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129),
                 (130, 146), (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584),
                 (353, 356), (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376), (225, 232),
                 (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305), (331, 336), (377, 383),
                 (484, 488), (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), (258, 264), (274, 283), (357, 363),
                 (419, 429), (306, 313), (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), (108, 111), (551, 558),
                 (195, 198), (384, 389), (394, 397), (435, 438),(364, 368), (284, 290), (390, 393), (408, 414), (547, 550),
                 (450, 453), (430, 434), (248, 252), (291, 295),(585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
                ]

obj_to_hoi = [hoi_classes[x[0] - 1: x[1]] for x in obj_hoi_index]
obj_actions = [[action_classes.index(y) for y in x] for x in obj_to_hoi]

def action_to_obj_idx(obj_class, action_hico):
    action_coco = action_classes[action_hico]
    obj_interval = obj_hoi_index[obj_class]
    hois = hoi_classes[obj_interval[0] - 1 : obj_interval[1]]
    return hois.index(action_coco)

def main():
    pass


if __name__ == '__main__':
    main()

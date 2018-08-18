"""
Created on Oct 02, 2017

@author: Siyuan Qi

Description of the file.

"""

activities = ['arranging_objects', 'picking_objects', 'taking_medicine', 'making_cereal', 'cleaning_objects', 'stacking_objects', 'having_meal', 'microwaving_food', 'unstacking_objects', 'taking_food']
subactivities = ['reaching', 'moving', 'pouring', 'eating', 'drinking', 'opening', 'placing', 'closing', 'null', 'cleaning']
objects = ['medcinebox', 'cup', 'bowl', 'box', 'milk', 'book', 'microwave', 'plate', 'remote', 'cloth']
affordances = ['movable', 'stationary', 'reachable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner']
align_subact_to_aff = [2, 0, 3, 2, 6, 7, 8, 9, 1, 10]

activity_index = dict()
for a in activities:
    activity_index[a] = activities.index(a)

subactivity_index = dict()
for s in subactivities:
    subactivity_index[s] = subactivities.index(s)

object_index = dict()
for o in objects:
    object_index[o] = objects.index(o)

affordance_index = dict()
for u in affordances:
    affordance_index[u] = affordances.index(u)


def main():
    pass


if __name__ == '__main__':
    main()

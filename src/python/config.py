"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        self.project_root = '/home/siyuan/projects/release/gpnn/'
        self.tmp_root = os.path.join(self.project_root, 'tmp')
        self.log_root = os.path.join(self.project_root, 'log')

        # self.cad_data_root = '/home/siyuan/data/CAD120/'
        self.hico_data_root = os.path.join(self.project_root, 'tmp', 'hico')
        # self.vcoco_data_root = '/media/siyuan/381ad97f-9c6a-4e12-be7e-1514562cce7e/data/v-coco-features/'


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

import numpy as np
import cv2
import time
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.multitask import MultitaskParams
import albumentations as A
from numba.typed import List
from models.centernet import ProcessImages as CenternetProcess
from models.semseg import ProcessImages as SemsegProcess
from models.depth import ProcessImages as DepthProcess


class ProcessImages(IPreProcessor):
    def __init__(self, params: MultitaskParams, start_augment = None):
        self.params: MultitaskParams = params
        self.cn_process = CenternetProcess(self.params.cn_params, start_augmentation=start_augment)
        self.semseg_process = SemsegProcess(self.params.semseg_params, start_augment=None)
        self.depth_process = DepthProcess(self.params.depth_params, do_augmentation=False)

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # start_time = time.time()
        # TODO: How to properly handle augmentation? probably save the "replay" from centernet process and apply it here for semseg and depth as well

        # lets start with processing the data for the centernet since it is the most complex
        raw_data, input_data, ground_truth, piped_params = self.cn_process.process(raw_data, input_data, ground_truth, piped_params)

        # add semseg process, do not overwrite the already generated ground_truth!
        semseg_gt = None
        raw_data, _, semseg_gt, piped_params = self.semseg_process.process(raw_data, input_data, semseg_gt, piped_params)

        # add semseg process, do not overwrite the already generated ground_truth!
        depth_gt = None
        raw_data, _, depth_gt, piped_params = self.depth_process.process(raw_data, input_data, depth_gt, piped_params)

        # concat the semseg and depth groundtruth to the generated centernet ground truth
        if self.params.TRAIN_CN:
            ground_truth = np.concatenate([ground_truth, semseg_gt, np.expand_dims(depth_gt, axis=-1)], axis=-1)
        else:
            ground_truth = np.concatenate([semseg_gt, np.expand_dims(depth_gt, axis=-1)], axis=-1)
        
        # elapsed_time = time.time() - start_time
        # print(str(elapsed_time) + " s")

        return raw_data, input_data, ground_truth, piped_params

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.multitask import MultitaskParams
from models.semseg.loss import SemsegLoss
from models.depth.loss import DepthLoss
from models.centernet.loss import CenternetLoss


class MultitaskLoss(Loss):
    def __init__(self, params: MultitaskParams, reduction=losses_utils.ReductionV2.AUTO, name=None, save_path=None):
        super().__init__(reduction=reduction, name=name)
        self.params = params
        self.cn_loss = CenternetLoss(params.cn_params)
        self.semseg_loss = SemsegLoss(self.params.semseg_params.CLASS_WEIGHTS)
        self.depth_loss = DepthLoss()

        if self.params.TRAIN_CN:
            self.cn_offset = {
                "y_true": [0, self.params.cn_params.mask_channels() + 1],
                "y_pred": [0, self.params.cn_params.mask_channels()],
            }
        else:
            self.cn_offset = {"y_true": [0, 0], "y_pred": [0, 0]}

        self.semseg_offset = {
            "y_true": [self.cn_offset["y_true"][1], self.cn_offset["y_true"][1] + len(SEMSEG_CLASS_MAPPING.items()) + 1],
            "y_pred": [self.cn_offset["y_pred"][1], self.cn_offset["y_pred"][1] + len(SEMSEG_CLASS_MAPPING.items())]
        }
        self.depth_offset = {
            "y_true": [self.semseg_offset["y_true"][1], self.semseg_offset["y_true"][1] + 1],
            "y_pred": [self.semseg_offset["y_pred"][1], self.semseg_offset["y_pred"][1] + 1]
        }

    def calc_depth(self, y_true, y_pred):
        y_true_depth = y_true[:, :, :, self.depth_offset["y_true"][0]:self.depth_offset["y_true"][1]]
        y_pred_depth = y_pred[:, :, :, self.depth_offset["y_pred"][0]:self.depth_offset["y_pred"][1]]
        return self.depth_loss.call(tf.squeeze(y_true_depth, axis=-1), y_pred_depth)

    def calc_semseg(self, y_true, y_pred):
        y_true_semseg = y_true[:, :, :, self.semseg_offset["y_true"][0]:self.semseg_offset["y_true"][1]]
        y_pred_semseg = y_pred[:, :, :, self.semseg_offset["y_pred"][0]:self.semseg_offset["y_pred"][1]]
        return self.semseg_loss.call(y_true_semseg, y_pred_semseg)

    def calc_centernet(self, y_true, y_pred):
        y_true_cn = y_true[:, :, :, self.cn_offset["y_true"][0]:self.cn_offset["y_true"][1]]
        y_pred_cn = y_pred[:, :, :, self.cn_offset["y_pred"][0]:self.cn_offset["y_pred"][1]]
        return self.cn_loss.call(y_true_cn, y_pred_cn)
    
    def call(self, y_true, y_pred):
        depth_loss_val = self.calc_depth(y_true, y_pred)
        semseg_loss_val = self.calc_semseg(y_true, y_pred)
        if self.params.TRAIN_CN:
            cn_loss_val = self.calc_centernet(y_true, y_pred)
            total_loss = depth_loss_val + semseg_loss_val + (cn_loss_val * 0.5)
        else:
            total_loss = depth_loss_val + semseg_loss_val
        return total_loss

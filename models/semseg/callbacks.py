import os
import cv2
import numpy as np
from numba.typed import List
from common.utils import to_3channel
from models.semseg import Params
from data.label_spec import SEMSEG_CLASS_MAPPING


class SaveSampleImages():
    def __init__(self, storage_path: str, params: Params, show_imgs: bool = True):
        self._params = params
        self._show_imgs = show_imgs
        self._storage_path = storage_path
        if not os.path.exists(self._storage_path):
            print("Storage folder does not exist yet, creating: " + self._storage_path)
            os.makedirs(self._storage_path)
        self.step_counter = 0

    def save_imgs(self, inp, y_true, y_pred):
        y_true_masked = y_true[:, :, :, :-1] # last channel has a ignore mask, thus cut it out
        inp_img = inp[0].numpy().astype(np.uint8)
        semseg_true = to_3channel(y_true_masked[0].numpy(), List(SEMSEG_CLASS_MAPPING.items()))
        semseg_pred = to_3channel(y_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items()))
        
        if self._show_imgs:
            cv2.imshow("input", inp_img)
            cv2.imshow("semseg_true", semseg_true)
            cv2.imshow("semseg_pred", semseg_pred)
            cv2.waitKey(1)

        self.step_counter += 1
        if self.step_counter % 100 == 0:
            cv2.imwrite(f"{self._storage_path}/semseg_input_{self.step_counter}.png", inp_img)
            cv2.imwrite(f"{self._storage_path}/semseg_gt_{self.step_counter}.png", semseg_true)
            cv2.imwrite(f"{self._storage_path}/semseg_pred_{self.step_counter}.png", semseg_pred)

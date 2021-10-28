import os
import sys
import json
import cv2
import time
import numpy as np
import tensorflow as tf
import pygame
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from numba.typed import List
from common.utils import to_3channel, Roi, cmap_depth
from data.label_spec import OD_CLASS_MAPPING, SEMSEG_CLASS_MAPPING
from models.centernet import post_processing


class ShowPygame():
    """
    Callback to show results in pygame window, custom callback called in custom train step
    """
    def __init__(self, storage_path: str, multitask_params):
        """
        :param storage_path: path to directory were the image data should be stored
        """
        self._params = multitask_params
        self._process_output = False # takes quite some time and slows down training!
        self._step_counter = 0
        self._display = pygame.display.set_mode((self._params.INPUT_WIDTH, int(self._params.INPUT_HEIGHT + self._params.MASK_HEIGHT*3)), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._storage_path = storage_path
        if not os.path.exists(self._storage_path):
            print("Storage folder does not exist yet, creating: " + self._storage_path)
            os.makedirs(self._storage_path)


        if self._params.TRAIN_CN:
            self.cn_offset = {
                "y_true": [0, self._params.cn_params.mask_channels() + 1],
                "y_pred": [0, self._params.cn_params.mask_channels()],
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

    def show(self, inp, y_true, y_pred):
        # start_time = time.time()

        # Slice data
        # --------------------------------
        y_true_depth = y_true[0][:, :, self.depth_offset["y_true"][0]:self.depth_offset["y_true"][1]] * 255.0
        y_pred_depth = y_pred[0][:, :, self.depth_offset["y_pred"][0]:self.depth_offset["y_pred"][1]] * 255.0
        y_true_semseg = y_true[0][:, :, self.semseg_offset["y_true"][0]:self.semseg_offset["y_true"][1]]
        y_pred_semseg = y_pred[0][:, :, self.semseg_offset["y_pred"][0]:self.semseg_offset["y_pred"][1]]
        y_true_cn = y_true[0][:, :, self.cn_offset["y_true"][0]:self.cn_offset["y_true"][1]]
        y_pred_cn = y_pred[0][:, :, self.cn_offset["y_pred"][0]:self.cn_offset["y_pred"][1]]

        # Display Input Image
        # --------------------------------
        show_inp_img = inp[0][0].numpy()
        show_inp_img = show_inp_img.astype(np.uint8)
        # get objects from y_pred
        if self._process_output and self._params.TRAIN_CN:
            roi = Roi()
            prediction = np.array(y_pred_cn)
            objects = post_processing.process_2d_output(prediction, roi, self._params, 0.2)

            for obj in objects:
                color = list(OD_CLASS_MAPPING.values())[obj["cls_idx"]]
                # top_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1] - obj["center_height"]))
                # bottom_left = (int(obj["bottom_left"][0]), int(obj["bottom_left"][1]))
                # bottom_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1]))
                # bottom_right = (int(obj["bottom_right"][0]), int(obj["bottom_right"][1]))
                # cv2.line(show_inp_img, bottom_left, bottom_center, (0, 255, 0) , 1) 
                # cv2.line(show_inp_img, bottom_center, bottom_right, (0, 255, 0) , 1) 
                # cv2.line(show_inp_img, bottom_center, top_center, (0, 255, 0) , 1)

                top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
                bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
                cv2.rectangle(show_inp_img, top_left, bottom_right, color, 1)

                cv2.circle(show_inp_img, (int(obj["center"][0]), int(obj["center"][1])), 2, color, 1)

        inp_img = cv2.cvtColor(show_inp_img, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self._display.blit(surface_img, (0, 0))

        if self._params.TRAIN_CN:
            # Object Detection
            # --------------------------------
            heatmap_true = np.array(y_true_cn[:, :, :1]) # needed because otherwise numba makes mimimi
            heatmap_true = to_3channel(heatmap_true, List([("object", (0, 0, 255))]), 0.01, True, False)
            weights = np.stack([y_true_cn[:, :, -1]]*3, axis=-1)
            heatmap_pred = np.array(y_pred_cn[:, :, :1])
            heatmap_pred = to_3channel(heatmap_pred, List([("object", (0, 0, 255))]), 0.01, True, False)
            # display heatmap y_true
            heatmap_true = cv2.cvtColor(heatmap_true, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            surface_y_true = pygame.surfarray.make_surface(heatmap_true)
            self._display.blit(surface_y_true, (0, self._params.INPUT_HEIGHT))
            # display heatmap y_pred
            heatmap_pred = cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            surface_y_pred = pygame.surfarray.make_surface(heatmap_pred)
            self._display.blit(surface_y_pred, (self._params.MASK_WIDTH, self._params.INPUT_HEIGHT))

        # Show Semseg
        # --------------------------------------
        semseg_true = cv2.cvtColor(to_3channel(y_true_semseg.numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self._display.blit(surface_y_true, (0, int(self._params.INPUT_HEIGHT + self._params.MASK_HEIGHT)))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred_semseg.numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self._display.blit(surface_y_pred, (self._params.MASK_WIDTH, int(self._params.INPUT_HEIGHT + self._params.MASK_HEIGHT)))

        # Show Depth
        # -------------------------------------
        surface_y_true = pygame.surfarray.make_surface(cmap_depth(np.squeeze(y_true_depth, axis=-1), vmin=0.1, vmax=255.0).swapaxes(0, 1))
        self._display.blit(surface_y_true, (0, int(self._params.INPUT_HEIGHT + self._params.MASK_HEIGHT * 2)))
        surface_y_pred = pygame.surfarray.make_surface(cmap_depth(np.squeeze(y_pred_depth, axis=-1), vmin=0.1, vmax=255.0).swapaxes(0, 1))
        self._display.blit(surface_y_pred, (self._params.MASK_WIDTH, int(self._params.INPUT_HEIGHT + self._params.MASK_HEIGHT * 2)))

        self._step_counter += 1
        if self._step_counter % 2000 == 0:
            pygame.image.save(self._display, f"{self._storage_path}/train_result_{self._step_counter}.png")

        pygame.display.flip()

        # elapsed_time = time.time() - start_time
        # print(str(elapsed_time) + " s")


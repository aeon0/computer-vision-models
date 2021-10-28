import numpy as np
import cv2
import math
import copy
import random
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from models.centernet import ProcessImages
from common.utils import resize_img
from data.label_spec import OD_CLASS_MAPPING, OD_CLASS_IDX
import albumentations as A


class CenterTrackerProcess(ProcessImages):
    def random_offset(self):
        abs_val = random.random() * 35 + 10
        if random.random() < 0.5:
            return -abs_val
        else:
            return abs_val

    def gen_prev_heatmap(self, shape, gt_2d_info, roi):
        prev_heatmap = np.zeros((*shape, 1))
        mask_width = self.params.INPUT_WIDTH // self.params.R
        mask_height = self.params.INPUT_HEIGHT // self.params.R
        for [center_x, center_y, width, height] in gt_2d_info:
            # skip with certain probability to simulate a false negative
            if random.random() > self.params.FN_PROB:
                # add fp with certain probability
                if random.random() < self.params.FP_PROB:
                    fp_center_x = int(center_x + self.random_offset())
                    fp_center_y = int(center_y + self.random_offset())
                    peak = ((random.random() * 0.8) + 0.2) ** 2
                    self.fill_heatmap(prev_heatmap, 0, fp_center_x, fp_center_y, width, height, mask_width, mask_height, peak)
                # jitter the true positive positions by gausian distribution
                noisy_center_x = int(center_x + np.random.normal() * self.params.POS_NOISE_WEIGHT * width)
                noisy_center_y = int(center_y + np.random.normal() * self.params.POS_NOISE_WEIGHT * height)
                peak = random.random() * 0.5 + 0.2 # usually higher peaks than fps
                self.fill_heatmap(prev_heatmap, 0, noisy_center_x, noisy_center_y, width, height, mask_width, mask_height, peak)

        return prev_heatmap

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        if isinstance(raw_data, list):
            # TODO: Use the t-1 frame
            _, input_data, ground_truth, _ = super().process(raw_data[0], input_data, ground_truth, piped_params)
            assert(False and "Not implemented yet")
        else:
            _, input_data, ground_truth, _ = super().process(raw_data, input_data, ground_truth, piped_params)
             # make input_data a dict since we will have multiple input channels
            input_data = { "img": input_data }

            # create previous heatmap
            prev_heatmap = self.gen_prev_heatmap((ground_truth.shape[0], ground_truth.shape[1]), piped_params["gt_2d_info"], piped_params["roi"])

            # copy as prev image
            prev_img = input_data["img"].copy()

            # Translate and scale the image as well as the heatmap to mimic a t-1 frame and add to input_data
            trans = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, always_apply=True, border_mode=cv2.BORDER_CONSTANT)
            transform = A.ReplayCompose(
                [trans],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
                additional_targets={'prev_heatmap': 'image'}
            )
            keypoints = list(np.array(piped_params["gt_2d_info"])[:,:2] * self.params.R)
            transformed = transform(image=prev_img, prev_heatmap=prev_heatmap, keypoints=keypoints)
            input_data["prev_img"] = transformed["image"]
            input_data["prev_heatmap"] = transformed["prev_heatmap"]

            # Add to ground_truth: Regression param offset to t-1 object using the scale and dx, dy changes from the previous image transformation
            # applied_params = transformed["replay"]["transforms"][0]["params"]
            # scale = applied_params["scale"]
            # shift_x = applied_params["dx"] * prev_img.shape[0]
            # shift_y = applied_params["dy"] * prev_img.shape[0]

            for i, [center_x, center_y, width, height] in enumerate(piped_params["gt_2d_info"]):
                if self.params.REGRESSION_FIELDS["track_offset"].active:
                    prev_center_x = transformed["keypoints"][i][0]
                    prev_center_y = transformed["keypoints"][i][1]

                    if prev_center_x < 0 or prev_center_x > prev_img.shape[1] or prev_center_y < 0 or prev_center_y > prev_img.shape[0]:
                        # previous center is outside of heatmap bounds, set offset to 0
                        offset_x = 0
                        offset_y = 0
                    else:
                        offset_x = prev_center_x - (center_x * self.params.R)
                        offset_y = prev_center_y - (center_y * self.params.R)
                    ground_truth[center_y][center_x][:][self.params.start_idx("track_offset"):self.params.end_idx("track_offset")] = [offset_x, offset_y]
                else:
                    assert(False and "I mean, why even use CenterTracker when we dont regress the t-1 track offset?")

        return raw_data, input_data, ground_truth, piped_params

import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.centernet import process_2d_output
from common.utils import Roi
from data.label_spec import OD_CLASS_MAPPING
from models.centernet.params import CenternetParams


class TestPosProcessing:
    def fill_obj(self, obj, nb_classes, info):
        obj[1:] = [
            *info["class"],
            *info["loc_off"],
            info["width"], info["height"],
            *info["bottom_left_off"],
            *info["bottom_right_off"],
            *info["bottom_center_off"],
            info["center_height"],
            info["radial_dist"],
            info["orientation"],
            info["obj_width"],
            info["obj_height"],
            info["obj_length"]
        ]

    def test_post_processing(self):
        nb_classes = 3
        params = CenternetParams(nb_classes)
        params.REGRESSION_FIELDS["class"].active = True
        params.REGRESSION_FIELDS["r_offset"].active = True
        params.REGRESSION_FIELDS["fullbox"].active = True
        params.REGRESSION_FIELDS["l_shape"].active = True
        params.REGRESSION_FIELDS["3d_info"].active = True
        params.R = 2.0

        output_mask = np.zeros((9, 11, params.mask_channels()))

        # Add obj 1
        testObj1 = output_mask[4][5]
        testObj1[0] = 0.8 # set class
        self.fill_obj(testObj1, nb_classes, {
            "class": [0.0, 1.0, 0.0], "width": 7, "height": 6, "loc_off": [0.1, 0.2], "bottom_left_off": [-2, 1], "bottom_right_off": [2, 1], "bottom_center_off": [0, 2],
            "center_height": 3.0, "radial_dist": 30.0, "orientation": 0.0, "obj_width": 1.5, "obj_height": 1.0, "obj_length": 2.0
        })

        roi = Roi()
        roi.scale = 0.25
        roi.offset_left = -5
        roi.offset_right = -5
        roi.offset_top = -3
        objects = process_2d_output(output_mask, roi, params)

        org_height = (output_mask.shape[0] * params.R) * (1.0 / roi.scale) - roi.offset_top
        org_width = (output_mask.shape[1] * params.R) * (1.0 / roi.scale) - roi.offset_left - roi.offset_right
        org_img = np.zeros((int(org_height), int(org_width), 3))
        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["cls_idx"]]
            top_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1] - obj["center_height"]))
            bottom_left = (int(obj["bottom_left"][0]), int(obj["bottom_left"][1]))
            bottom_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1]))
            bottom_right = (int(obj["bottom_right"][0]), int(obj["bottom_right"][1]))
            cv2.line(org_img, bottom_left, bottom_center, (0, 255, 0) , 1) 
            cv2.line(org_img, bottom_center, bottom_right, (0, 255, 0) , 1) 
            cv2.line(org_img, bottom_center, top_center, (0, 255, 0) , 1)

            top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
            bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
            cv2.rectangle(org_img, top_left, bottom_right, (255, 0, 0), 1)

            cv2.circle(org_img, (int(obj["center"][0]), int(obj["center"][1])), 2, (0, 0, 255), 1)

        f, (ax1) = plt.subplots(1, 1)
        ax1.imshow(cv2.cvtColor(org_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.show()

import numpy as np
import pytest
from models.centernet import CenternetLoss
import tensorflow as tf
from models.centernet.params import CenternetParams


class TestLoss():
    def setup_method(self):
        nb_classes = 3
        self.params = CenternetParams(nb_classes)
        self.params.REGRESSION_FIELDS["class"].active = True
        self.params.REGRESSION_FIELDS["r_offset"].active = True
        self.params.REGRESSION_FIELDS["fullbox"].active = True
        self.params.REGRESSION_FIELDS["l_shape"].active = True
        self.params.REGRESSION_FIELDS["radial_dist"].active = True
        self.params.REGRESSION_FIELDS["3d_info"].active = True

        # Object data
        self.cp_x = 1 
        self.cp_y = 1
        self.cls_idx = 1
        self.obj_data = {
            "class": [0.0, 1.0, 0.0], "r_offset": [0.2, 0.3], "width_px": 2.2, "height_px": 1.1,
            "bottom_left_off": [-2.0, 1.5], "bottom_right_off": [2.1, 1.2], "bottom_center_off": [0.5, 1.7],
            "radial_dist": 23.0, "radial_dist": 23.0, "orientation": 0.12, "obj_width": 1.8, "obj_height": 1.1, "obj_length": 2.9
        }

        # Create ground truth input
        self.mask_height = 7
        self.mask_width = 7
        self.channels = self.params.mask_channels()
        self.ground_truth = np.zeros((self.mask_height, self.mask_width, self.channels + 1))
        self.ground_truth[:, :, -1] = 1.0 # set weights
        # class with a bit of distribution to the right keypoint
        self.ground_truth[self.cp_y    ][self.cp_x][0] = 1.0
        self.ground_truth[self.cp_y + 1][self.cp_x][0] = 0.8
        # regression params
        self.ground_truth[self.cp_y][self.cp_x][1:] = [
            *self.obj_data["class"], *self.obj_data["r_offset"], self.obj_data["width_px"], self.obj_data["height_px"],
            *self.obj_data["bottom_left_off"], *self.obj_data["bottom_right_off"], *self.obj_data["bottom_center_off"],
            self.obj_data["radial_dist"], self.obj_data["radial_dist"], self.obj_data["orientation"], self.obj_data["obj_width"], self.obj_data["obj_height"], self.obj_data["obj_length"], 1.0
        ]

        # Create perfect prediction
        self.perfect_prediction = self.ground_truth.copy()
        self.perfect_prediction[self.cp_y + 1][self.cp_x][0] = 0.0

        # Create loss class
        self.loss = CenternetLoss(self.params)

    def test_no_loss(self):
        no_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([self.perfect_prediction])).numpy()
        assert no_loss < 0.0001

    def test_object_loss(self):
        # Test object loss: peak not quite at 1.0
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y][self.cp_x][0] = 0.8
        obj_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert obj_loss < 0.01
        # Test obj loss: one off peak vs random wrong peak
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y + 1][self.cp_x][0] = 1.0
        one_off_obj_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        prediction[self.cp_y + 1][self.cp_x][0] = 0.0 # reset previous peak
        prediction[self.cp_y + 5][self.cp_x][0] = 1.0
        wrong_peak_ojb_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert one_off_obj_loss < wrong_peak_ojb_loss

    def test_class_loss(self):
        # Test class loss: peak not quite at 1.0
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y][self.cp_x][1 + self.cls_idx] = 0.8
        class_loss = self.loss.class_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert class_loss < 0.01
        # Test class loss: one off peak vs random wrong peak
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y][self.cp_x][1 + self.cls_idx + 1] = 1.0
        class_loss = self.loss.class_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert class_loss > 0.5

    def test_fullbox_loss(self):
        # Test size loss: width
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        width_idx = self.params.start_idx("fullbox")
        height_idx = self.params.end_idx("fullbox") - 1
        fp[width_idx] = self.obj_data["width_px"] + 10
        small_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        fp[width_idx] = self.obj_data["width_px"] - 30
        large_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert large_width_loss > small_width_loss > 0
        # Test width method directly
        width_loss = self.loss.fullbox_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert width_loss > 0.0

        # Test size loss: height
        fp[width_idx] = self.obj_data["width_px"] # reset width to ground truth
        fp[height_idx] = self.obj_data["height_px"] + 10
        small_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        fp[height_idx] = self.obj_data["height_px"] - 30
        large_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert large_width_loss > small_width_loss > 0
        # Test height method directly
        height_loss = self.loss.fullbox_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert height_loss > 0.0

    def test_l_shape_loss(self):
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x][self.params.start_idx("l_shape"):self.params.end_idx("l_shape")]
        fp[0] = self.obj_data["bottom_left_off"][0] + 1
        fp[1] = self.obj_data["bottom_left_off"][1] + 1
        fp[2] = self.obj_data["bottom_right_off"][0] + 1
        fp[3] = self.obj_data["bottom_right_off"][1] + 1
        fp[4] = self.obj_data["bottom_center_off"][0] + 1
        fp[5] = self.obj_data["bottom_center_off"][1] + 1
        loss_val = self.loss.l_shape_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 1.0

    def test_radial_dist_loss(self):
        prediction = self.perfect_prediction.copy()
        radial_dist_idx = self.params.start_idx("3d_info")
        fp = prediction[self.cp_y][self.cp_x]
        fp[radial_dist_idx] = self.obj_data["radial_dist"] + 2.5
        loss_val = self.loss.radial_dist_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.1

    def test_orientation_loss(self):
        prediction = self.perfect_prediction.copy()
        orientation_idx = self.params.start_idx("3d_info") + 1
        fp = prediction[self.cp_y][self.cp_x]
        fp[orientation_idx] = self.obj_data["orientation"] + 1.5 # 90 deg error
        loss_val = self.loss.orientation_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.1

    def test_obj_dims_loss(self):
        prediction = self.perfect_prediction.copy()
        start_idx = self.params.start_idx("3d_info")
        fp = prediction[self.cp_y][self.cp_x]
        fp[start_idx + 2] = self.obj_data["obj_width"] + 0.5
        fp[start_idx + 3] = self.obj_data["obj_height"] + 0.5
        fp[start_idx + 4] = self.obj_data["obj_length"] + 0.5
        loss_val = self.loss.obj_dims_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.1

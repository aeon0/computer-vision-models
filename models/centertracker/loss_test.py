import numpy as np
import pytest
import tensorflow as tf
from models.centertracker.loss import CentertrackerLoss
from models.centertracker.params import CentertrackerParams
from models.centernet.loss_test import TestLoss


class TestLossCenterTracker(TestLoss):
    def setup_method(self):
        super().setup_method()
        # Change some members to work with centertracker
        self.params = CentertrackerParams(self.params.NB_CLASSES)
        self.params.REGRESSION_FIELDS["track_offset"].active = True
        self.loss = CentertrackerLoss(self.params)

        s = self.ground_truth.shape
        self.ground_truth = np.concatenate((self.ground_truth, np.zeros((s[0], s[1], 2))), axis=2)
        self.perfect_prediction = np.concatenate((self.perfect_prediction, np.zeros((s[0], s[1], 2))), axis=2)
        self.ground_truth[self.cp_y][self.cp_y][self.params.start_idx("track_offset"):self.params.end_idx("track_offset")] = [1.0, 2.0]
        self.perfect_prediction[self.cp_y][self.cp_y][self.params.start_idx("track_offset"):self.params.end_idx("track_offset")] = [1.0, 2.0]

    def test_track_offset_loss(self):
        no_loss = self.loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([self.perfect_prediction]), tf.float32)).numpy()
        assert no_loss < 0.0001
        # some loss
        kp = self.perfect_prediction[self.cp_y][self.cp_y][self.params.start_idx("track_offset"):self.params.end_idx("track_offset")]
        kp[0] = 0.0
        kp[1] = 3.0
        some_loss = self.loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([self.perfect_prediction]), tf.float32)).numpy()
        assert some_loss > 0.1
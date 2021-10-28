import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.losses import Loss


class SemsegLoss(Loss):
    def __init__(self, class_weights, reduction=losses_utils.ReductionV2.AUTO, name=None, save_path=None):
        super().__init__(reduction=reduction, name=name)
        self._class_weights = class_weights
        self.channels = len(class_weights)

    @staticmethod
    def focal_tversky(y_true, y_pred, pos_mask, class_weights):
        smooth = 1.0
        gamma = 0.75
        alpha = 0.7
        y_true_masked = y_true * pos_mask * class_weights
        y_pred_masked = y_pred * pos_mask * class_weights
        y_true_pos = tf.reshape(y_true_masked, [-1])
        y_pred_pos = tf.reshape(y_pred_masked, [-1])
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1.0 - y_pred_pos))
        false_pos = tf.reduce_sum((1.0 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
        return tf.math.pow((1.0 - pt_1), gamma)

    def call(self, y_true, y_pred):
        # pos mask is encoded in last channel of y_pred img
        pos_mask = y_true[:, :, :, -1]
        y_true_mask = y_true[:, :, :, :-1]
        pos_mask_stacked = tf.repeat(tf.expand_dims(pos_mask, axis=-1), self.channels, axis=-1)
        focal_loss = self.focal_tversky(y_true_mask, y_pred, pos_mask_stacked, self._class_weights)
        return focal_loss

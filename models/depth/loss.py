import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.depth.params import Params
import matplotlib.pyplot as plt
import pygame
import numpy as np
from tensorflow.python.keras.utils import losses_utils
from common.utils import cmap_depth


class DepthLoss(Loss):
    @staticmethod
    def calc(y_true, y_pred):
        pos_mask = tf.cast(tf.greater(y_true, 0.1), tf.float32)
        n = tf.reduce_sum(pos_mask)

        # Point-wise depth
        # l_depth = pos_mask * tf.abs(y_pred - y_true)
        # l_depth = tf.reduce_sum(l_depth) / n

        loss_mask = pos_mask * ((y_true - y_pred) / tf.maximum(tf.math.abs(y_true), 1e-10))
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(n, 0), lambda: loss_val / n, lambda: loss_val)
        return loss_val

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=-1)
        return self.calc(y_true, y_pred)


# TODO: Do this like in all other models by implementing a make_train_function() on the Model
# def _show_depthmaps(self, y_true, y_pred):
#     surface_y_true = pygame.surfarray.make_surface(cmap_depth(y_true[0], vmin=0.1, vmax=255.0).swapaxes(0, 1))
#     self.display.blit(surface_y_true, (0, 0))
#     surface_y_pred = pygame.surfarray.make_surface(cmap_depth(y_pred[0], vmin=0.1, vmax=255.0).swapaxes(0, 1))
#     self.display.blit(surface_y_pred, (640, 0))

#     if self.step_counter % 1000 == 0:
#         pygame.image.save(self.display, f"{self.save_path}/train_result_{self.step_counter}.png")

#     pygame.display.flip()

# self._show_depthmaps(y_true, y_pred)
# self.step_counter += 1
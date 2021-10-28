import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from tensorflow.python.keras.engine import data_adapter
from models.dmds import DmdsParams
from models.dmds.convert import create_dataset
from common.utils import tflite_convert, resize_img
from common.layers import encoder, upsample_block, bottle_neck_block
from models.depth.model import create_model as create_depth_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame
from tensorflow.python.eager import backprop


class DmdsModel(Model):
    def init_file_writer(self, log_dir):
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.train_step_counter = 0

        self.display = pygame.display.set_mode((640*2, 256*3), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    def _cmap(self, depth_map, max_val = 140.0, to_255 = True, swap_axes = True):
        npumpy_depth_map = depth_map.numpy()
        if max_val == "max":
            npumpy_depth_map /= np.amax(npumpy_depth_map)
        else:
            npumpy_depth_map /= max_val
        npumpy_depth_map = np.clip(npumpy_depth_map, 0.0, 1.0)
        if to_255:
            npumpy_depth_map *= 255.0
        if swap_axes:
            npumpy_depth_map = npumpy_depth_map.swapaxes(0, 1)
        depth_stack = [npumpy_depth_map.astype(np.uint8)]
        npumpy_depth_map = np.concatenate(depth_stack * 3, axis=-1)
        return npumpy_depth_map

    def train_step(self, data):
        def rgb_img(img):
            np_img = img.numpy()
            c0 = tf.unstack(np_img, axis=-1)
            return tf.stack([c0[2], c0[1], c0[0]], axis=-1)

        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            depth_model = self.get_layer("depth_model")
            motion_model = self.get_layer("motion_model")
            
            combined_input_depth_model = (tf.concat([input_data[1], input_data[0]], axis=0),)
            depth1 = depth_model(combined_input_depth_model, training=True)
            depth0 = tf.concat(tf.split(depth1, 2, axis=0)[::-1], axis=0)

            combined_input_motion_model = (
                tf.concat([input_data[0], input_data[1]], axis=0),
                tf.concat([input_data[1], input_data[0]], axis=0),
                depth0,
                depth1,
            )
            obj_tran, bg_tran, rot = motion_model(combined_input_motion_model, training=True)

            obj_tran_inv = tf.concat(tf.split(obj_tran, 2, axis=0)[::-1], axis=0)
            bg_tran_inv = tf.concat(tf.split(bg_tran, 2, axis=0)[::-1], axis=0)
            rot_inv = tf.concat(tf.split(rot, 2, axis=0)[::-1], axis=0)

            loss_val = self.custom_loss.calc(
                combined_input_motion_model[0], combined_input_motion_model[1],
                depth0, depth1,
                obj_tran, obj_tran_inv,
                bg_tran, bg_tran_inv,
                rot, rot_inv, tf.concat([input_data[2], input_data[2]], axis=0),
                gt[0], gt[1], self.train_step_counter)
            loss_dict = self.custom_loss.loss_vals.copy()

        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        # Display images
        surface_img0 = pygame.surfarray.make_surface(input_data[0][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img0, (0, 0))
        surface_img1 = pygame.surfarray.make_surface(input_data[1][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img1, (640, 0))

        surface_x0 = pygame.surfarray.make_surface(self._cmap(depth1[0], max_val="max"))
        self.display.blit(surface_x0, (0, 256))
        surface_weights = pygame.surfarray.make_surface(self._cmap(tf.expand_dims(self.custom_loss.depth_weights[0], axis=-1), max_val=1.0))
        self.display.blit(surface_weights, (640, 256))

        surface_warped = pygame.surfarray.make_surface(self.custom_loss.resampled_img1[0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_warped, (0, 256*2))
        surface_mask = pygame.surfarray.make_surface(self._cmap(self.custom_loss.warp_mask[0], max_val=1.0))
        self.display.blit(surface_mask, (640, 256*2))

        if self.train_step_counter % 100 == 0:
            pygame.image.save(self.display, f"{self.log_dir}/train_result_{self.train_step_counter}.png")

        pygame.display.flip()

        # Using the file writer, log images
        tf.summary.experimental.set_step(int(self.train_step_counter / 40))
        with self.file_writer.as_default():
            # tf.summary.image("img0", rgb_img(combined_input[0]), max_outputs=10)

            tf.summary.histogram("depth_hist", depth0)
            tf.summary.histogram("tran", bg_tran)
            tf.summary.histogram("rot", rot)

        self.train_step_counter += 1

        loss_dict["sum"] = loss_val
        return loss_dict
    
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        combined_input = (
            tf.concat([input_data[0], input_data[1]], axis=0),
            tf.concat([input_data[1], input_data[0]], axis=0),
            tf.concat([input_data[2], input_data[2]], axis=0)
        )
        y_pred = self(combined_input, training=True)

        depth0, depth1, obj_tran, bg_tran, rot, _ = y_pred

        loss_val = self.custom_loss.calc(
            combined_input[0], combined_input[1],
            depth0, depth1,
            obj_tran, obj_tran_inv,
            bg_tran, bg_tran_inv,
            rot, rot_inv, combined_input[2],
            gt[0], gt[1], self.train_step_counter)
        loss_dict = self.custom_loss.loss_vals.copy()

        loss_dict["sum"] = loss_val
        return loss_dict


class ScaleConstraint(tf.keras.constraints.Constraint):
    """The weight tensors will be constrained to not fall below constraint_minimum, this is used for the scale variables in DMLearner/motion_field_net."""

    def __init__(self, constraint_minimum=0.01):
        self.constraint_minimum = constraint_minimum

    def __call__(self, w):
        return tf.nn.relu(w - self.constraint_minimum) + self.constraint_minimum

    def get_config(self):
        return {'constraint_minimum': self.constraint_minimum}


def create_motion_model(input_height: int, input_width: int) -> tf.keras.Model:
    input_t0 = Input(shape=(input_height, input_width, 3))
    rescaled_input_t0 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(input_t0)

    input_t1 = Input(shape=(input_height, input_width, 3))
    rescaled_input_t1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(input_t1)

    depth_t0 = Input(shape=(input_height, input_width, 1))
    depth_t1 = Input(shape=(input_height, input_width, 1))

    inp_mm_t0 = Concatenate()([rescaled_input_t0, depth_t0])
    inp_mm_t1 = Concatenate()([rescaled_input_t1, depth_t1])
    mm_inp = Concatenate()([inp_mm_t0, inp_mm_t1])

    mm_conv1 = Conv2D(8, [3, 3], strides=2, padding="same", name='mm/conv1')(mm_inp)
    mm_conv1 = BatchNormalization()(mm_conv1)
    mm_conv1 = ReLU()(mm_conv1)
    mm_conv2 = Conv2D(16, [3, 3], strides=2, padding="same", name='mm/conv2')(mm_conv1)
    mm_conv2 = BatchNormalization()(mm_conv2)
    mm_conv2 = ReLU()(mm_conv2)
    mm_conv3 = Conv2D(32, [3, 3], strides=2, padding="same", name='mm/conv3')(mm_conv2)
    mm_conv3 = BatchNormalization()(mm_conv3)
    mm_conv3 = ReLU()(mm_conv3)
    mm_conv4 = Conv2D(64, [3, 3], strides=2, padding="same", name='mm/conv4')(mm_conv3)
    mm_conv4 = BatchNormalization()(mm_conv4)
    mm_conv4 = ReLU()(mm_conv4)
    mm_conv5 = Conv2D(128, [3, 3], strides=2, padding="same", name='mm/conv5')(mm_conv4)
    mm_conv5 = BatchNormalization()(mm_conv5)
    mm_conv5 = ReLU()(mm_conv5)
    mm_conv6 = Conv2D(196, [3, 3], strides=2, padding="same", name='mm/conv6')(mm_conv5)
    mm_conv6 = BatchNormalization()(mm_conv6)
    mm_conv6 = ReLU()(mm_conv6)

    bottleneck = AveragePooling2D([mm_conv6.shape[1], mm_conv6.shape[2]])(mm_conv6)
    background_translation = Conv2D(3, [1, 1], strides=1, activation=None, bias_initializer=None, name='mm/background_translation')(bottleneck)
    background_rotation = Conv2D(3, [1, 1], strides=1, activation=None, bias_initializer=None, name='mm/background_rotation')(bottleneck)

    resudial_translation = upsample_block("mm/up0_", mm_conv6, mm_conv5, 128)
    resudial_translation = upsample_block("mm/up1_", resudial_translation, mm_conv4, 64)
    resudial_translation = upsample_block("mm/up2_", resudial_translation, mm_conv3, 32)
    resudial_translation = upsample_block("mm/up3_", resudial_translation, mm_conv2, 16)
    resudial_translation = upsample_block("mm/up4_", resudial_translation, mm_conv1, 8)
    resudial_translation = Conv2D(3, [1, 1], strides=1, name='mm/resudial_translation')(resudial_translation)

    scaling = Conv2D(1, (1, 1), use_bias=False, kernel_constraint=ScaleConstraint(0.001), name='mm/scaling')
    rot_scaling = Conv2D(1, (1, 1), use_bias=False, kernel_constraint=ScaleConstraint(0.001), name='mm/rot_scaling')

    resudial_translation = Concatenate(axis=-1, name='mm/scaled_resudial_translation')([
        scaling(tf.expand_dims(resudial_translation[:, :, :, 0], axis=-1)),
        scaling(tf.expand_dims(resudial_translation[:, :, :, 1], axis=-1)),
        scaling(tf.expand_dims(resudial_translation[:, :, :, 2], axis=-1))])

    background_translation = Concatenate(axis=-1, name='mm/scaled_background_translation')([
        scaling(tf.expand_dims(background_translation[:, :, :, 0], axis=-1)),
        scaling(tf.expand_dims(background_translation[:, :, :, 1], axis=-1)),
        scaling(tf.expand_dims(background_translation[:, :, :, 2], axis=-1))])

    background_rotation = Concatenate(axis=-1, name='mm/scaled_background_rotation')([
        rot_scaling(tf.expand_dims(background_rotation[:, :, :, 0], axis=-1)),
        rot_scaling(tf.expand_dims(background_rotation[:, :, :, 1], axis=-1)),
        rot_scaling(tf.expand_dims(background_rotation[:, :, :, 2], axis=-1))])

    motion_model = Model(inputs=[input_t0, input_t1, depth_t0, depth_t1], outputs=[resudial_translation, background_translation, background_rotation] , name="motion_model")
    return motion_model

def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    intr = Input(shape=(3, 3))
    
    input_t0 = Input(shape=(input_height, input_width, 3))
    input_t1 = Input(shape=(input_height, input_width, 3))
    depth_t0 = Input(shape=(input_height, input_width, 1))

    # Depth Model
    # ------------------------------
    DepthModel = create_depth_model(input_height, input_width)
    depth_t1 = DepthModel([input_t1])

    # Motion Model
    # ------------------------------
    MotionModel = create_motion_model(input_height, input_width)
    motion_outputs = MotionModel([input_t0, input_t1, depth_t0, depth_t1])

    return DmdsModel(inputs=[input_t0, input_t1, depth_t0, intr], outputs=[depth_t1, *motion_outputs, intr])


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, params.LOAD_DEPTH_MODEL)
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))

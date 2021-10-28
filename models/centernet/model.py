import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers, regularizers, Model
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, Input, Concatenate
import numpy as np
from models.centernet import CenternetParams, create_dataset
from common.utils import tflite_convert
from common.layers import bottle_neck_block, upsample_block, encoder


def create_output_layer(x, params, namescope):
    output_layer_arr = []

    heatmap = Conv2D(16, (3, 3), padding="same", use_bias=False, name=f"{namescope}heatmap_conv2d")(x)
    heatmap = BatchNormalization(name=f"{namescope}heatmap_batchnorm")(heatmap)
    heatmap = ReLU(6.0)(heatmap)
    heatmap = Conv2D(1, (1, 1), padding="valid", activation=tf.nn.sigmoid, name=f"{namescope}heatmap")(heatmap)
    output_layer_arr.append(heatmap)

    # All other regerssion parameters are optional, but note that the order is important here and should be as in the OrderedDict REGRESSION_FIELDS
    if params.REGRESSION_FIELDS["class"].active:
        # Create object class regression
        obj_class = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}obj_class_conv2d")(x)
        obj_class = BatchNormalization(name=f"{namescope}obj_class_batchnorm")(obj_class)
        obj_class = ReLU(6.0)(obj_class)
        obj_class = Conv2D(params.REGRESSION_FIELDS["class"].size, (1, 1), padding="valid", activation=tf.nn.sigmoid, name=f"{namescope}obj_class")(obj_class)
        output_layer_arr.append(obj_class)

    if params.REGRESSION_FIELDS["r_offset"].active:
        # Create location offset due to R scaling
        offset = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}r_offset_conv2d")(x)
        offset = BatchNormalization(name=f"{namescope}r_offset_batchnorm")(offset)
        offset = ReLU(6.0)(offset)
        offset = Conv2D(params.REGRESSION_FIELDS["r_offset"].size, (1, 1), padding="valid", name=f"{namescope}r_offset")(offset)
        offset = ReLU(1.0)(offset)
        output_layer_arr.append(offset)

    if params.REGRESSION_FIELDS["fullbox"].active:
        # Create 2D output: fullbox (width, height)
        fullbox = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}fullbox_conv2d")(x)
        fullbox = BatchNormalization(name=f"{namescope}fullbox_batchnorm")(fullbox)
        fullbox = ReLU(6.0)(fullbox)
        fullbox = Conv2D(params.REGRESSION_FIELDS["fullbox"].size, (1, 1), padding="valid", name=f"{namescope}fullbox")(fullbox)
        fullbox = ReLU(1.0)(fullbox)
        output_layer_arr.append(fullbox)

    if params.REGRESSION_FIELDS["l_shape"].active:
        # Create 2.5D output: bottom_left_edge_point, bottom_center_edge_point, bottom_right_edge_point, center_height
        l_shape = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}lshape_conv2d")(x)
        l_shape = BatchNormalization(name=f"{namescope}lshape_batchnorm")(l_shape)
        l_shape = ReLU(6.0)(l_shape)
        l_shape = Conv2D(params.REGRESSION_FIELDS["l_shape"].size, (1, 1), padding="valid", name=f"{namescope}lshape")(l_shape)
        output_layer_arr.append(l_shape)

    if params.REGRESSION_FIELDS["radial_dist"].active:
        # Create radial distance output
        radial_dist = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}dist_conv2d")(x)
        radial_dist = BatchNormalization(name=f"{namescope}dist_batchnorm")(radial_dist)
        radial_dist = ReLU(6.0)(radial_dist)
        radial_dist = Conv2D(1, (1, 1), padding="valid", name=f"{namescope}dist")(radial_dist)
        radial_dist = ReLU(1.0)(radial_dist)
        output_layer_arr.append(radial_dist)

    if params.REGRESSION_FIELDS["3d_info"].active:
        # Create radial distance output
        radial_dist = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}dist_conv2d")(x)
        radial_dist = BatchNormalization(name=f"{namescope}dist_batchnorm")(radial_dist)
        radial_dist = ReLU(6.0)(radial_dist)
        radial_dist = Conv2D(1, (1, 1), padding="valid", name=f"{namescope}dist")(radial_dist)
        output_layer_arr.append(radial_dist)

        # Create orientation output
        orientation = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}orientation_conv2d")(x)
        orientation = BatchNormalization(name=f"{namescope}orientation_batchnorm")(orientation)
        orientation = ReLU(6.0)(orientation)
        orientation = Conv2D(1, (1, 1), padding="valid", name=f"{namescope}orientation")(orientation)
        output_layer_arr.append(orientation)

        # Create object dimensions in [m] (width, height, length)
        obj_dims = Conv2D(8, (3, 3), padding="same", use_bias=False, name=f"{namescope}dims_conv2d")(x)
        obj_dims = BatchNormalization(name=f"{namescope}dims_batchnorm")(obj_dims)
        obj_dims = ReLU(6.0)(obj_dims)
        obj_dims = Conv2D(3, (1, 1), padding="valid", name=f"{namescope}dims")(obj_dims)
        output_layer_arr.append(obj_dims)

    # Concatenate output
    output_layer = Concatenate(axis=3, name="centernet/out")(output_layer_arr)
    return output_layer

def create_layers(params, inp, inject_layers=[]):
    namescope = "centernet/"

    x, _ = encoder(8, inp, output_scaled_down=True, inject_layers=inject_layers, namescope=f"{namescope}")
    output_layer = create_output_layer(x, params, namescope)

    return output_layer

def create_model(params: CenternetParams, inject_layers=[]) -> tf.keras.Model:
    inp = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, params.CHANNELS))

    inject_layer_inputs = []
    for layer in inject_layers:
        inject_layer_inputs.append(Input(shape=layer.shape[1:]))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)

    output_layer = create_layers(params, inp_rescaled, inject_layers)

    # Create Model
    model = Model(inputs=[inp, *inject_layer_inputs], outputs=output_layer, name="centernet/model")
    return model


if __name__ == "__main__":
    params = CenternetParams(5)
    model = create_model(params)
    model.summary()
    plot_model(model, to_file="./tmp/centernet_model.png")
    tflite_convert.tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))

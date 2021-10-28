import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from models.depth import Params
from models.depth.convert import create_dataset
from common.utils import resize_img
from common.utils.tflite_convert import tflite_convert
from common.layers import encoder, upsample_block, bottle_neck_block


def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    inp_t0 = Input(shape=(input_height, input_width, 3))
    inp_t0_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp_t0)
    namescope = "depth/"

    x0, _ = encoder(16, inp_t0_rescaled, output_scaled_down=True, namescope=f"{namescope}")
    x0 = Conv2D(8, (3, 3), use_bias=False, padding="same", name=f"{namescope}depthhead_conv2d")(x0)
    x0 = BatchNormalization(name=f"{namescope}depthhead_batchnorm")(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(1, kernel_size=1, padding="same", use_bias=True, name=f"{namescope}out")(x0)

    depth_model = Model(inputs=[inp_t0], outputs=x0, activation="relu", name="depth_model")

    return depth_model


if __name__ == "__main__":
    params = Params()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/depth_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from common.layers import bottle_neck_block, upsample_block, encoder
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.centernet import create_layers as create_centernet
from models.semseg import create_layers as create_semseg
from models.multitask import MultitaskParams


def create_model(params: MultitaskParams) -> tf.keras.Model:
    inp = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)

    # Semseg & Depth Multitask Model
    # ------------------------------
    x = create_semseg(inp_rescaled, cut_head=True)

    semseg_map = Conv2D(8, (3, 3), padding="same", name=f"semseg/semseghead_conv2d", kernel_regularizer=l2(l=0.0001))(x)
    semseg_map = BatchNormalization(name=f"semseg/semseghead_batchnorm")(semseg_map)
    semseg_map = ReLU(6.0)(semseg_map)
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name=f"semseg/out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(semseg_map)

    depth_map = Conv2D(12, (3, 3), use_bias=False, padding="same", name=f"depthhead_conv2d")(x)
    depth_map = BatchNormalization(name=f"depthhead_batchnorm")(depth_map)
    depth_map = ReLU(6.0)(depth_map)
    depth_map = Conv2D(1, kernel_size=1, padding="same", activation="relu", name=f"depth_head/out")(depth_map)

    if not params.BASE_TRAINABLE:
        dummy_model = Model(inputs=[inp], outputs=[semseg_map, depth_map])
        for layer in dummy_model.layers:
            layer.trainable = False

    if params.TRAIN_CN:
        # Centernet Head
        # ------------------------------
        centernet_output = create_centernet(params.cn_params, inp_rescaled, inject_layers=[semseg_map, depth_map])
        out_layer = Concatenate()([centernet_output, semseg_map, depth_map])
    else:
        out_layer = Concatenate()([semseg_map, depth_map])

    return Model(inputs=[inp], outputs=out_layer, name="multitask/model")

if __name__ == "__main__":
    from data.label_spec import OD_CLASS_MAPPING
    from tensorflow.keras.utils import plot_model
    from models.multitask import create_dataset, MultitaskLoss
    from common.utils import tflite_convert, set_weights

    params = MultitaskParams(len(OD_CLASS_MAPPING.items()))
    model = create_model(params)

    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss = MultitaskLoss(params)
    model.compile(optimizer=opt, loss=loss)

    model.summary()
    plot_model(model, to_file="./tmp/multitask_model.png")
    # set_weights.set_weights("/home/computer-vision-models/trained_models/multitask_nuscenes_2021-03-10-075953/tf_model_11/keras.h5", model, get_layers=["semseg/model", "centernet/model"])
    tflite_convert.tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))

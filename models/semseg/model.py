import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import encoder
from models.semseg import Params


def create_model(params: Params) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: 
    :return: Semseg Keras Model
    """
    inp = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3))
    inp_scaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=-0.5)(inp)
    
    should_scale_down = params.INPUT_WIDTH != params.MASK_WIDTH
    x, _ = encoder(16, inp_scaled, output_scaled_down=should_scale_down, namescope="semseg/")
    
    x = Conv2D(8, (3, 3), padding="same", name=f"semseghead_conv2d", use_bias=False, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization(name=f"semseghead_batchnorm")(x)
    x = ReLU(6.0)(x)
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, activation="sigmoid", name=f"semseghead_out", kernel_regularizer=l2(l=0.0001))(x)
    
    model = Model(inputs=[inp], outputs=semseg_map, name="semseg/model")
    return model


# To test model creation and quickly check if edgetpu compiler compiles it correctly
# Note: The conversion might work after init but could fail after training, to be sure train for 1-2 epochs and try converting again
if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    from models.semseg import convert
    from common.utils import set_weights, tflite_convert
    
    params = Params()
    reduce_output_size = params.INPUT_WIDTH != params.MASK_WIDTH
    model = create_model(params)

    # set_weights.set_weights("...", model, force_resize=False)
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert.tflite_convert(model, "./tmp", True, True, convert.create_dataset(model.input.shape))

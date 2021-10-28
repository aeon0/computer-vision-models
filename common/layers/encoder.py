import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, LayerNormalization
from tensorflow.keras.regularizers import l2
from common.layers import bottle_neck_block, upsample_block


def encoder(filter_size: int, input_tensor: tf.Tensor, output_scaled_down: bool = False, do_upsample: bool = True, inject_layers=[], namescope: str = "encoder/"):
    x = input_tensor
    fms = []
    if not output_scaled_down:
        fms = [input_tensor]

    filters = (np.array([1, 2, 4, 6, 8, 12, 16]) * filter_size)

    # Downsample
    # ----------------------------
    x = Conv2D(filters[0], 5, use_bias=False, padding="same", name=f"{namescope}initial_downsample", strides=(2, 2))(x)
    x = BatchNormalization(name=f"{namescope}initial_batchnorm")(x)
    x = ReLU(6., name=f"{namescope}initial_acitvation")(x)
    fms.append(x)

    for layer in inject_layers:
        x = Concatenate()([x, layer])

    x = bottle_neck_block(f"{namescope}downsample_2/", x, filters[1])
    x = bottle_neck_block(f"{namescope}downsample_3/", x, filters[1], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_4/", x, filters[2])
    x = bottle_neck_block(f"{namescope}downsample_5/", x, filters[2], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_6/", x, filters[3])
    x = bottle_neck_block(f"{namescope}downsample_7/", x, filters[3], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_8/", x, filters[4])
    x = bottle_neck_block(f"{namescope}downsample_9/", x, filters[4], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_10/", x, filters[5])
    x = bottle_neck_block(f"{namescope}downsample_11/", x, filters[5], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_12/", x, filters[6])
    x = bottle_neck_block(f"{namescope}downsample_13/", x, filters[6], downsample = True)
    fms.append(x)

    # Alternative to more downsampling, dilated layers:
    # dilation_rates = [3, 6, 9, 12]
    # concat_tensors = []
    # x = Conv2D(filters, kernel_size=1, name=f"{namescope}start_dilation_1x1")(x)
    # concat_tensors.append(x)
    # for i, rate in enumerate(dilation_rates):
    #     x = bottle_neck_block(f"{namescope}dilation_{i}", x, filters)
    #     x = BatchNormalization(name=f"{namescope}dilation_batchnorm_{i}")(x)
    #     concat_tensors.append(x)

    # x = Concatenate(name=f"{namescope}dilation_concat")(concat_tensors)
    # fms.append(x)

    # Upsample
    # ----------------------------
    if do_upsample:
        for i in range(len(fms) - 2, -1, -1):
            fms[i] = Conv2D(filters[i], (3, 3), use_bias=False, padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
            fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
            fms[i] = ReLU(6.)(fms[i])
            x = upsample_block(f"{namescope}upsample_{i}/", x, fms[i], filters[i])

    return x, fms
# Check: https://coral.ai/docs/edgetpu/models-intro/#supported-operations for supported ops on EdgeTpu
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, Model
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, Input, Concatenate
import numpy as np
from models.centertracker.params import CentertrackerParams
import models.centernet as centernet
from tensorflow.keras.utils import plot_model


def create_model(params: CentertrackerParams):
    base_model = centernet.create_model(params)
    curr_img_input = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3), name="current_img")
    prev_img_input = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3), name="prev_img")
    prev_heatmap_input = Input(shape=(params.MASK_HEIGHT, params.MASK_WIDTH, 1), name="prev_heatmap")

    # upsample the heatmap to the same size as the images and make it compatiable to the base model input shape
    prev_heatmap = Conv2DTranspose(8, kernel_size=2, strides=(2, 2), use_bias=False, padding='same')(prev_heatmap_input)
    new_input = Concatenate(axis=3)([curr_img_input, prev_img_input, prev_heatmap])
    new_input = Conv2D(3, (3, 3), padding="same")(new_input)
    
    # output tensor
    output_tensor: tf.Tensor = base_model(new_input)
    # centernet/encoder_output_relu/Identity
    feature_map_tensor = output_tensor.graph.get_tensor_by_name("centernet/encoder_output/Identity:0")

    # All other regerssion parameters are optional, but note that the order is important here and should be as in the OrderedDict REGRESSION_FIELDS
    if params.REGRESSION_FIELDS["track_offset"].active:
        # Create location offset due to R scaling
        track_offset = Conv2D(32, (3, 3), padding="same", use_bias=False)(feature_map_tensor)
        track_offset = BatchNormalization()(track_offset)
        track_offset = ReLU()(track_offset)
        track_offset = Conv2D(params.REGRESSION_FIELDS["track_offset"].size, (1, 1), padding="valid", activation=None)(track_offset)
        output_tensor = Concatenate(axis=3)([output_tensor, track_offset])

    # Create Model
    input_dict = {
        "img": curr_img_input,
        "prev_img": prev_img_input,
        "prev_heatmap": prev_heatmap_input
    }
    model = Model(inputs=[*base_model.inputs, base_model.output], outputs=output_tensor, name="centernet")
    return model

params = CentertrackerParams(6)
model = create_model(params)
# model.summary()
plot_model(model, to_file="./centertrack.png")

# graph_def = output_tensor.graph.as_graph_def()
# for node in graph_def.node:
#     if "encoder" in node.name:
#         print(f"{node.name} {node.op}")

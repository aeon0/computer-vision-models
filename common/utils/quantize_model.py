import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, UpSampling2D, Concatenate, ReLU
from tensorflow.python.keras.engine import functional
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply
from tensorflow.keras.models import Model, clone_model


def quantize_model(model, ignore_layers = ()):
    ignore_layers += (Concatenate, ReLU, BatchNormalization, Conv2DTranspose, UpSampling2D, functional.Functional)
    def apply_quantization_annotation(layer):
        # Add all layers to the tuple that currently do not have any quantization support
        if not isinstance(layer, ignore_layers):
            return quantize_annotate_layer(layer)
        return layer

    annotated_model = clone_model(model, clone_function=apply_quantization_annotation)
    quant_aware_model = quantize_apply(annotated_model)
    return quant_aware_model

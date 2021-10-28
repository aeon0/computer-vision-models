import tensorflow as tf
import numpy as np
from tensorflow.python.eager import backprop


def layer_weights_have_same_size(layer1, layer2):
    weights1 = layer1.get_weights()
    weights2 = layer2.get_weights()
    if len(weights1) != len(weights2):
        return False
    for i in range(len(weights1)):
        if weights1[i].shape != weights2[i].shape:
            return False
    return True

def set_weights(base_model_path, new_model, force_resize=False, custom_objects = {}, get_layers = []):
    # Store names of base model layers of model in dict
    base_model = tf.keras.models.load_model(base_model_path, custom_objects=custom_objects, compile=False)
    base_layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    for get_layer in get_layers:
        try:
            additional_layers = dict([(layer.name, layer) for layer in base_model.get_layer(get_layer).layers])
            base_layer_dict = {**base_layer_dict, **additional_layers}
        except ValueError:
            print(f"Could not find layer: {get_layer}")

    # Loop through actual model and see if names are matching, set weights in case they are
    for i, layer in enumerate(new_model.layers):
        if layer.name in base_layer_dict:
            print(f"Setting weights for {layer.name} by name")
            try:
                weights = base_layer_dict[layer.name].get_weights()
                goal_size = layer.get_weights()
                for j in range(min(len(goal_size), len(weights))):
                    if goal_size[j].shape != weights[j].shape:
                        print(f"Need to resize from {weights[j].shape} to {goal_size[j].shape}")
                        weights[j] = np.resize(weights[j], goal_size[j].shape)
                layer.set_weights(weights)
            except ValueError as e:
                print(f"ValueError: {e}")
        elif len(base_model.layers) > i and layer_weights_have_same_size(base_model.layers[i], layer):
            print(f"Setting weights for {layer.name} by matching size")
            layer.set_weights(base_model.layers[i].get_weights())
        elif len(base_model.layers) > i and force_resize:
            print(f"Setting weights for {layer.name} by forcing resize")
            try:
                weights = base_model.layers[i].get_weights()
                goal_size = layer.get_weights()
                for j in range(len(goal_size)):
                    if goal_size[j].shape != weights[j].shape:
                        print(f"Need to resize from {weights[j].shape} to {goal_size[j].shape}")
                        weights[j] = np.resize(weights[j], goal_size[j].shape)
                layer.set_weights(weights)
            except (ValueError, IndexError) as e:
                print(f"ValueError: {e}")
        else:
            print(f"Not found: {layer.name}")
    return new_model

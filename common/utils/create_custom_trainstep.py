import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


def create_custom_trainstep(model: tf.keras.Model, save_imgs):
    """
    Create a custom training step in order to add a custom callback function
    """
    original_train_step = model.train_step
    def custom_train_step(original_data):
        # call custom callback function
        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = model(x, training=True)
        save_imgs(x, y_true, y_pred)
        # call original train step
        result = original_train_step(original_data)
        return result
    return custom_train_step

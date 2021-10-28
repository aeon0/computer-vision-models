import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks


class SaveToStorage(callbacks.Callback):
    """
    Callback to save Trainings & Results to a local storage
    """
    def __init__(
        self,
        storage_path: str,
        keras_model: Model,
        save_initial_weights: bool = True):
        """
        :param storage_path: path to directory were the data should be stored
        :param keras_model: keras model that should be saved to the training
        :param save_initial_weights: boolean to determine if weights should be saved initally before training,
                                    default = True
        """
        self._storage_path = storage_path

        if not os.path.exists(self._storage_path):
            print("Storage folder does not exist yet, creating: " + self._storage_path)
            os.makedirs(self._storage_path)

        self._save_initial_weights = save_initial_weights
        self._keras_model = keras_model
        self._curr_epoch = 0

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self._save_initial_weights:
            self.save(save_as_initial=True)

    def on_epoch_begin(self, epoch, logs=None):
        self._curr_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # TODO: Check if model is better now and only save weights in that case
        save_tf = True
        self.save(save_tf)

    def save(self, save_tf: bool = True, save_as_initial: bool = False):
        epoch = self._curr_epoch

        if epoch <= 0:
            stdout_origin = sys.stdout
            sys.stdout = open(self._storage_path + "/network_architecture.txt", "w")
            try:
                self._keras_model.summary()
            except ValueError:
                pass
            sys.stdout.close()
            sys.stdout = stdout_origin

        if save_tf:
            if save_as_initial:
                epoch = "init"

            # save in SaveModel format
            tf_export_dir = self._storage_path + "/tf_model_" + str(epoch)
            os.makedirs(tf_export_dir)
            self._keras_model.save(tf_export_dir, save_format="tf")
            # save in h5 format
            tf_export_dir_keras = tf_export_dir + "/keras.h5"
            self._keras_model.save(tf_export_dir_keras, save_format="h5")

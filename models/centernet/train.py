import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

from tensorflow.python.keras.engine import data_adapter
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Logger, Config, set_weights
from common.callbacks import SaveToStorage
from data.label_spec import OD_CLASS_MAPPING
from models.centernet import ProcessImages, CenternetParams, CenternetLoss, ShowPygame, create_model


def make_custom_callbacks(keras_model, show_pygame):
    original_train_step = keras_model.train_step
    def call_custom_callbacks(original_data):
        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = keras_model(x, training=True)
        result = original_train_step(original_data)
        # custom stuff called during training
        show_pygame.show_od(x, y_true, y_pred)
        return result
    return call_custom_callbacks

if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = CenternetParams(len(OD_CLASS_MAPPING))

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "labels", "nuscenes_train")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(90, 10),
        shuffle_data=True
    )

    train_gen = MongoDBGenerator(
        [collection_details],
        [train_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, [12, 24])]
    )
    val_gen = MongoDBGenerator(
        [collection_details],
        [val_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)]
    )

    # Create Model
    storage_path = "./trained_models/centernet_nuimages_nuscenes_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    opt = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07) 
    loss = CenternetLoss(params)

    model: tf.keras.models.Model = create_model(params)
    model.train_step = make_custom_callbacks(model, ShowPygame(storage_path + "/images", od_params=params))
    model.compile(optimizer=opt, loss=loss, metrics=[loss.obj_focal_loss, loss.class_loss, loss.fullbox_loss, loss.radial_dist_loss])

    if params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.LOAD_WEIGHTS, model)

    model.summary()
    model.run_eagerly = True

    # Train Model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, False), tensorboard_callback]
    params.save_to_storage(storage_path)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        use_multiprocessing=False,
        workers=3,
    )

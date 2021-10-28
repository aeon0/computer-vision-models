import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config, set_weights
from data.label_spec import OD_CLASS_MAPPING
from models.multitask import create_model, MultitaskParams, ProcessImages, MultitaskLoss, ShowPygame


def make_custom_callbacks(keras_model, show_pygame):
    original_train_step = keras_model.train_step
    def call_custom_callbacks(original_data):
        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = keras_model(x, training=True)
        result = original_train_step(original_data)
        # custom stuff called during training
        show_pygame.show(x, y_true, y_pred)
        return result
    return call_custom_callbacks

if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = MultitaskParams(len(OD_CLASS_MAPPING.items()))

    Config.add_config('./config.ini')
    con = ("local_mongodb", "labels", "nuscenes_train")

    td, vd = load_ids(
        con,
        data_split=(90, 10),
        shuffle_data=True
    )

    train_data = [td]
    val_data = [vd]
    collection_details = [con]

    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, [3, 5])],
        shuffle_data=True
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)],
        shuffle_data=True
    )

    # Create Model
    storage_path = "./trained_models/multitask_nuscenes_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss = MultitaskLoss(params)

    model: models.Model = create_model(params)
    model.train_step = make_custom_callbacks(model, ShowPygame(storage_path + "/images", params))
    metrics = [loss.calc_semseg, loss.calc_depth]
    if params.TRAIN_CN:
        metrics += [loss.calc_centernet, loss.cn_loss.obj_focal_loss, loss.cn_loss.class_loss, loss.cn_loss.fullbox_loss, loss.cn_loss.radial_dist_loss]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    if params.semseg_params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.semseg_params.LOAD_WEIGHTS, model)
    if params.TRAIN_CN and params.cn_params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.cn_params.LOAD_WEIGHTS, model)
    if params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.LOAD_WEIGHTS, model, get_layers=["semseg/model", "centernet/model"])

    model.summary()
    model.run_eagerly = True

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=3,
    )

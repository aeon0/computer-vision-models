import tensorflow as tf
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Logger, Config
from common.callbacks import SaveToStorage
from common.processors import AugmentImages
from data.label_spec import OD_CLASS_MAPPING
from models.centertracker import CenterTrackerProcess, CentertrackerLoss, CentertrackerParams, create_model

print("Using Tensorflow Version: " + tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4864)])


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = CentertrackerParams(len(OD_CLASS_MAPPING))
    params.REGRESSION_FIELDS["l_shape"].active = False
    params.REGRESSION_FIELDS["3d_info"].active = False

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "labels", "kitti")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(82, 18),
        shuffle_data=True
    )

    processors = [CenterTrackerProcess(params)]
    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=processors
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=processors
    )

    loss = CenterTrackerProcess(params)
    # metrics = [loss.class_focal_loss, loss.r_offset_loss, loss.fullbox_loss]
    metrics = []
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) 

    if params.LOAD_PATH is None:
        model: tf.keras.models.Model = create_model(params)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
    else:
        custom_objects = {"compute_loss": loss}
        model: tf.keras.models.Model = tf.keras.models.load_model(params.LOAD_PATH, compile=False)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary()

    # Train Model
    storage_path = "./trained_models/centertracker_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
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

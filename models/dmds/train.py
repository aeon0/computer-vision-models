import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config
from models.dmds import create_model, DmdsParams, ProcessImages
from models.dmds.loss import DmdsLoss


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = DmdsParams()

    # get one entry from the database
    Config.add_config('./config.ini')
    con = ("local_mongodb", "depth", "driving_stereo")
    scenes = [
        "2018-10-19-09-30-39",
        "2018-10-22-10-44-02",
        # "2018-10-23-08-34-04",
        # "2018-10-31-06-55-01",
        # "2018-10-27-10-02-04",
        # "2018-10-26-15-24-18",
        # "2018-10-27-08-54-23",
        # "2018-10-25-07-37-26",
        # "2018-10-24-14-13-21",
        # "2018-10-23-13-59-11",
        # "2018-10-12-07-57-23",
        # "2018-10-18-15-04-21",
        # "2018-10-17-14-35-33",
        # "2018-10-18-10-39-04",
        # "2018-10-30-13-45-14",
        # "2018-10-16-11-43-02",
        # "2018-07-27-11-39-31",
        # "2018-10-16-11-13-47",
        # "2018-07-24-14-31-18",
        # "2018-07-18-10-16-21",
        # "2018-07-16-15-37-46",
        # "2018-10-15-11-43-36",
        # "2018-10-16-07-40-57",
        # "2018-07-18-11-25-02",
        # "2018-10-17-15-38-01",
        # "2018-10-10-07-51-49",
        # These recs have cuts in them
        # "2018-08-17-09-45-58",
        # "2018-07-09-16-11-56",
        # "2018-07-16-15-18-53",
        # "2018-07-10-09-54-03",
        # "2018-10-11-17-08-31",
        # "2018-08-13-17-45-03",
        # "2018-08-13-15-32-19",
        # "2018-07-31-11-22-31",
        # "2018-07-31-11-07-48",
    ]
    train_data = []
    val_data = []
    collection_details = []

    # get ids
    for scene_token in scenes:
        td, vd = load_ids(
            con,
            data_split=(95, 5),
            shuffle_data=False,
            mongodb_filter={"scene_token": scene_token},
            sort_by={"timestamp": 1}
        )
        train_data.append(td)
        val_data.append(vd)
        collection_details.append(con)

    processors = [ProcessImages(params)]
    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=processors,
        data_group_size=2,
        continues_data_selection=True,
        shuffle_data=False
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=processors,
        data_group_size=2,
        continues_data_selection=True,
        shuffle_data=False
    )

    # Create Model
    opt = optimizers.Adam(lr=0.0006)

    model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)

    # custom_loss parameter only works because we override the compile() and train_step() of the tf.keras.Model
    model.compile(optimizer=opt, custom_loss=DmdsLoss(params))
    model.run_eagerly = True
    model.summary()

    if params.LOAD_PATH is not None:
        set_weights.set_weights(params.LOAD_PATH, model)

    if params.LOAD_DEPTH_MODEL is not None:
        set_weights.set_weights(params.LOAD_DEPTH_MODEL, model.get_layer("depth_model"))

    # Train model
    storage_path = "./trained_models/dmds_ds_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]
    model.init_file_writer(storage_path + "/images")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=3,
        # use_multiprocessing=True
    )

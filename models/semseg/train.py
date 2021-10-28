import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config, set_weights, create_custom_trainstep
from models.semseg import create_model, Params, SemsegLoss, ProcessImages, SaveSampleImages


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = Params()

    Config.add_config('./config.ini')
    collection_comma10k = ("local_mongodb", "labels", "comma10k")
    collection_mapillary_train = ("local_mongodb", "labels", "mapillary_training")
    collection_mapillary_val = ("local_mongodb", "labels", "mapillary_validation")

    # Create Data Generators
    train_data_comma10k, val_data_comma10k = load_ids(
        collection_comma10k,
        data_split=(91, 9),
        shuffle_data=True,
    )
    train_data_mapillary, _ = load_ids(
        collection_mapillary_train,
        data_split=(100, 0),
        shuffle_data=True,
    )
    _, val_data_mapillary = load_ids(
        collection_mapillary_val,
        data_split=(0, 100),
        shuffle_data=True,
    )

    train_gen = MongoDBGenerator(
        [collection_comma10k, collection_mapillary_train],
        [train_data_comma10k, train_data_mapillary],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, [3, 0])],
        shuffle_data=True
    )
    val_gen = MongoDBGenerator(
        [collection_comma10k, collection_mapillary_val],
        [val_data_comma10k, val_data_mapillary],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)],
        shuffle_data=True
    )

    # Create Model
    storage_path = "./trained_models/semseg_comma10k_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss = SemsegLoss(params.CLASS_WEIGHTS)

    reduce_output_size = params.INPUT_WIDTH != params.MASK_WIDTH
    model: models.Model = create_model(params)

    if params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.LOAD_WEIGHTS, model, custom_objects={"SemsegModel": tf.keras.Model}, force_resize=False)

    if params.QUANTIZE:
        model = tfmot.quantization.keras.quantize_model(model)
        if params.LOAD_WEIGHTS_QUANTIZED is not None:
            with tfmot.quantization.keras.quantize_scope():
                # TODO: there is still a bug in tfmot I belive with Default8BitConvTransposeQuantizeConfig object not found
                set_weights.set_weights(params.LOAD_WEIGHTS_QUANTIZED, model, custom_objects={"SemsegModel": tf.keras.Model}, force_resize=False)

    model.train_step = create_custom_trainstep(model, SaveSampleImages(storage_path + "/images", params).save_imgs)
    model.compile(optimizer=opt, loss=loss)

    model.summary()
    # for debugging custom loss or layers, set to True
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
        # use_multiprocessing=True
    )

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU'), True)

import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import pygame
import matplotlib.pyplot as plt
import argparse
import time
from pymongo import MongoClient
from common.utils import resize_img
from models.dmds.model import DmdsModel, ScaleConstraint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="depth", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="driving_stereo", help="MongoDB collection")
    parser.add_argument("--img_width", type=int, default=320, help="Width of image, must be model input")
    parser.add_argument("--img_height", type=int, default=128, help="Width of image, must be model input")
    parser.add_argument("--offset_bottom", type=int, default=0, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/path/to/tf_model_x/model_quant_edgetpu.tflite", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    # For debugging force a value here
    args.use_edge_tpu = False
    args.model_path = "/home/computer-vision-models/trained_models/dmds_ds_2021-02-16-231612/tf_model_1/keras.h5"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    is_tf_lite = args.model_path[-7:] == ".tflite"
    model = None
    interpreter = None
    input_details = None
    output_details = None

    if is_tf_lite:
        # Load the TFLite model and allocate tensors.
        if args.use_edge_tpu:
            print("Using EdgeTpu")
            interpreter = edgetpu.make_interpreter(args.model_path)
        else:
            print("Using TFLite")
            interpreter = tflite.Interpreter(args.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        with tfmot.quantization.keras.quantize_scope():
            custom_objects = {
                "DmdsModel": DmdsModel,
                "ScaleConstraint": ScaleConstraint
            }
            model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, custom_objects, compile=False)
        model.summary()
        print("Using Tensorflow")

    # alternative data source, mp4 video
    # cap = cv2.VideoCapture('/path/to/video.mp4')
    # cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    # while (cap.isOpened()):
    #     ret, img = cap.read()

    documents = collection.find({}).limit(1000)
    documents = list(documents)
    for i in range(0, len(documents)-1):
        decoded_img_t0 = np.frombuffer(documents[i]["img"], np.uint8)
        img_t0 = cv2.imdecode(decoded_img_t0, cv2.IMREAD_COLOR)
        img_t0, _ = resize_img(img_t0, args.img_width, args.img_height, args.offset_bottom)

        decoded_img_t1 = np.frombuffer(documents[i+1]["img"], np.uint8)
        img_t1 = cv2.imdecode(decoded_img_t1, cv2.IMREAD_COLOR)
        img_t1, _ = resize_img(img_t1, args.img_width, args.img_height, args.offset_bottom)

        intr = np.array([
            [375.0,  0.0, 160.0],
            [ 0.0, 375.0, 128.0],
            [ 0.0,   0.0,   1.0]
        ], dtype=np.float32)

        if is_tf_lite:
            interpreter.set_tensor(input_details[0]['index'], [img_t0])
            interpreter.set_tensor(input_details[1]['index'], [img_t1])
            #input_shape = input_details[0]['shape']
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            # output_data = interpreter.get_tensor(output_details[0]['index'])
        else:
            img_arr = [np.array([img_t0]), np.array([img_t1]), np.array([intr])]
            start_time = time.time()
            raw_result = model.predict(img_arr)
            elapsed_time = time.time() - start_time
            x0, x1, mm, tran, rot, intr = raw_result

        print(str(elapsed_time) + " s")

        f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)

        x0 = 1/x0
        x1 = 1/x1

        ax11.imshow(cv2.cvtColor(img_t0.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax21.imshow(x0[0], cmap="inferno", vmin=0, vmax=0.1)
        ax12.imshow(cv2.cvtColor(img_t1.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax22.imshow(x1[0], cmap="inferno", vmin=0, vmax=0.1)
        plt.show()

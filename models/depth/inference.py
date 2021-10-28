import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import time
from pymongo import MongoClient
from common.utils import resize_img, cmap_depth
import pygame
from pygame.locals import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="driving_stereo", help="MongoDB collection")
    parser.add_argument("--img_width", type=int, default=640, help="Width of image, must be model input")
    parser.add_argument("--img_height", type=int, default=256, help="Width of image, must be model input")
    parser.add_argument("--offset_bottom", type=int, default=0, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/path/to/tf_model_x/model_quant_edgetpu.tflite", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    # For debugging force a value here
    args.use_edge_tpu = True
    args.model_path = "/home/computer-vision-models/tmp/model_quant_edgetpu.tflite"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    display = pygame.display.set_mode((640, 256*2), pygame.HWSURFACE | pygame.DOUBLEBUF)
    is_tf_lite = args.model_path[-7:] == ".tflite"
    scale = 1.0
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
        # interpreter.resize_tensor_input(0, [2, 256, 640, 3])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        scale = output_details[0]["quantization"][0]
    else:
        print("Using Tensorflow GPU")
        model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
        model.summary()

    # cap = cv2.VideoCapture('/home/computer-vision-models/tmp/train.mp4')
    # cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    # while (cap.isOpened()):
    #     ret, img = cap.read()
    documents = collection.find({}).limit(1000)
    documents = list(documents)
    for i in range(0, len(documents)-1):
        decoded_img = np.frombuffer(documents[i]["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)

        img, _ = resize_img(img, args.img_width, args.img_height, args.offset_bottom)

        if is_tf_lite:
            interpreter.set_tensor(input_details[0]['index'], [img])
            #input_shape = input_details[0]['shape']
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            depth_map = interpreter.get_tensor(output_details[0]['index'])[0]
            depth_map = np.squeeze(depth_map)
        else:
            img_arr = np.array([img])
            start_time = time.time()
            raw_result = model.predict(img_arr)
            elapsed_time = time.time() - start_time
            depth_map = raw_result[0]

        print(str(elapsed_time) + " s")

        depth_map = np.squeeze(depth_map)
        depth_map = depth_map.astype(np.float32)
        pos_mask = np.where(depth_map > 1.0, 1.0, 0.0) 
        # to get the "true" depth in [m]
        # depth_map = (((depth_map * scale) / 22.0)**2 + 4.0)
        # or just scaling it to account for the quantization bias
        depth_map = depth_map * scale
        depth_map *= pos_mask
        depth_map = cv2.cvtColor(cmap_depth(depth_map, vmin=4.1, vmax=130.0), cv2.COLOR_BGR2RGB)

        # show on pygame window
        surface_input_img = pygame.surfarray.make_surface(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
        display.blit(surface_input_img, (0, 0))
        surface_depth = pygame.surfarray.make_surface(depth_map.swapaxes(0, 1))
        display.blit(surface_depth, (0, args.img_height))
        pygame.display.flip()

        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_SPACE:
                break

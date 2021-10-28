import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import pygame
from numba.typed import List
from pygame.locals import * 
import matplotlib.pyplot as plt
import argparse
import time
from pymongo import MongoClient
from common.utils import to_3channel, resize_img, Roi, cmap_depth
from data.label_spec import SEMSEG_CLASS_MAPPING, OD_CLASS_MAPPING
from models.multitask import MultitaskParams
from models.centernet import post_processing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuscenes_train", help="MongoDB collection")
    parser.add_argument("--img_width", type=int, default=320, help="Width of image, must be model input")
    parser.add_argument("--img_height", type=int, default=128, help="Width of image, must be model input")
    parser.add_argument("--offset_bottom", type=int, default=-120, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/path/to/tf_model_x/model_quant_edgetpu.tflite", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    # For debugging force a value here
    args.use_edge_tpu = True
    args.model_path = "/home/computer-vision-models/trained_models/multitask_nuscenes_2021-05-01-104246/tf_model_13/keras.h5"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]
    params = MultitaskParams(len(OD_CLASS_MAPPING.items()))

    is_tf_lite = args.model_path[-7:] == ".tflite"
    model = None
    interpreter = None
    input_details = None
    output_details = None
    scale = 1.0

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
        scale = output_details[0]["quantization"][0]
    else:
        with tfmot.quantization.keras.quantize_scope():
            model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
        model.summary()
        print("Using Tensorflow")

    # create pygame display to show images
    display = pygame.display.set_mode((args.img_width, args.img_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
    cn_offset     = [0,                params.cn_params.mask_channels()]
    semseg_offset = [cn_offset[1],     cn_offset[1] + len(SEMSEG_CLASS_MAPPING.items())]
    depth_offset  = [semseg_offset[1], semseg_offset[1] + 1]

    # alternative data source, mp4 video
    cap = cv2.VideoCapture('/home/computer-vision-models/tmp/train.mp4')
    cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
    while (cap.isOpened()):
        ret, img = cap.read()

    # documents = collection.find({}).limit(100)
    # for doc in documents:
    #     decoded_img = np.frombuffer(doc["img"], np.uint8)
    #     img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
        
        img, roi = resize_img(img, args.img_width, args.img_height, args.offset_bottom)

        if is_tf_lite:
            img_input = img
            interpreter.set_tensor(input_details[0]['index'], [img_input])
            #input_shape = input_details[0]['shape']
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
        else:
            img_arr = np.array([img])
            start_time = time.time()
            raw_result = model.predict(img_arr)
            elapsed_time = time.time() - start_time
            output_data = raw_result

        print(str(elapsed_time) + " s")

        print(f"{output_data.min()} - {output_data.max()}")
        output_data = output_data.astype(np.float32) * scale
        y_pred_depth = output_data[0][:, :, depth_offset[0]:depth_offset[1]]
        y_pred_depth *= 255.0
        y_pred_semseg = output_data[0][:, :, semseg_offset[0]:semseg_offset[1]]
        y_pred_cn = output_data[0][:, :, cn_offset[0]:cn_offset[1]]

        # input img + OD
        img = img.astype(np.uint8)
        roi = Roi()
        objects = post_processing.process_2d_output(y_pred_cn, roi, params.cn_params, 0.2)
        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["cls_idx"]]
            top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
            bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
            cv2.rectangle(img, top_left, bottom_right, color, 1)
            cv2.circle(img, (int(obj["center"][0]), int(obj["center"][1])), 2, color, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(img)
        display.blit(surface_img, (0, 0))
        # semseg
        semseg_pred = cv2.cvtColor(to_3channel(y_pred_semseg, List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        display.blit(surface_y_pred, (0, int(args.img_height)))
        # depth
        surface_y_pred = pygame.surfarray.make_surface(cmap_depth(np.squeeze(y_pred_depth, axis=-1), vmin=0.1, vmax=255.0).swapaxes(0, 1))
        display.blit(surface_y_pred, (params.MASK_WIDTH, int(args.img_height)))
        # center map
        heatmap_pred = cv2.cvtColor(to_3channel(y_pred_cn, List([("object", (0, 0, 255))]), 0.01, True, False), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(heatmap_pred)
        display.blit(surface_y_pred, (0, int(params.MASK_HEIGHT + args.img_height)))
        pygame.display.flip()

        # wait till space is pressed
        while False:
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_SPACE:
                break

    pygame.quit()

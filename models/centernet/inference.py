import tensorflow as tf
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import argparse
import time
from pymongo import MongoClient
from common.utils import to_3channel, resize_img
from data.label_spec import OD_CLASS_MAPPING
from models.centernet import process_2d_output, CenternetParams

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="kitti_test", help="MongoDB collection")
    parser.add_argument("--offset_bottom", type=int, default=-180, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/home/jo/git/computer-vision-models/trained_models/centernet_2021-01-30-152914/tf_model_9/keras.h5", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    # For debugging force a value here
    args.use_edge_tpu = True
    args.model_path = "/home/computer-vision-models/trained_models/centernet_nuimages_nuscenes_2021-04-18-14347/tf_model_17/keras.h5"

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
        model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
        model.summary()
        print("Using Tensorflow")

    # alternative data source, mp4 video
    cap = cv2.VideoCapture('/home/computer-vision-models/tmp/train.mp4')
    cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
    while (cap.isOpened()):
        ret, img = cap.read()
    # documents = collection.find({}).limit(20)
    # for doc in documents:
    #     decoded_img = np.frombuffer(doc["img"], np.uint8)
    #     img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)

        input_img, roi = resize_img(img, input_shape[2], input_shape[1], offset_bottom=args.offset_bottom)

        if is_tf_lite:
            input_img = input_img.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], [input_img])
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_mask = output_data[0]
        else:
            start_time = time.time()
            raw_result = model.predict(np.array([input_img]))
            elapsed_time = time.time() - start_time
            output_mask = raw_result[0]

        print(elapsed_time)

        # heatmap = to_3channel(output_mask, OD_CLASS_MAPPING, 0.2, True)
        # r = float(input_shape[1]) / float(output_shape[1])
        # # TODO: Create parameters from json file instead of using the default values
        # #       that way we dont have to remember the exact parameters for every model we train
        # params = CenternetParams(len(OD_CLASS_MAPPING))
        # # TODO: On inference, read the param file from the data folder
        # params.REGRESSION_FIELDS["l_shape"].active = False
        # params.REGRESSION_FIELDS["3d_info"].active = False

        # objects = process_2d_output(output_mask, roi, params, min_conf_value=0.2)
        # for obj in objects:
        #     color = list(OD_CLASS_MAPPING.values())[obj["cls_idx"]]
        #     color = (color[2], color[1], color[0])
        #     # fullbox
        #     top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
        #     bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
        #     cv2.rectangle(img, top_left, bottom_right, color, 1)
        #     # # 3d box
        #     # top_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1] - obj["center_height"]))
        #     # bottom_left = (int(obj["bottom_left"][0]), int(obj["bottom_left"][1]))
        #     # bottom_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1]))
        #     # bottom_right = (int(obj["bottom_right"][0]), int(obj["bottom_right"][1]))
        #     # cv2.line(img, bottom_left, bottom_center, color , 1)
        #     # cv2.line(img, bottom_center, bottom_right, color, 1)
        #     # cv2.line(img, bottom_center, top_center, color, 1)
        #     # circle at center
        #     cv2.circle(img, (int(obj["center"][0]), int(obj["center"][1])), 5, (0, 0, 255), 3)

        # print(str(elapsed_time) + " s")
        # cv2.imshow("Org Image with Objects", img)
        # cv2.imshow("Input Image", input_img)
        # cv2.imshow("Heatmap", heatmap)
        # cv2.waitKey(0)

import cv2
from pymongo import MongoClient
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from data.label_spec import Entry
import numpy as np
from numba import jit
from common.utils import resize_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload semseg data from comma10k dataset")
    parser.add_argument("--depth_map", type=str, help="Path to depth maps")
    parser.add_argument("--images", type=str, help="Path to images")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="driving_stereo", help="MongoDB collection")
    parser.add_argument("--resize", nargs='+', type=int, default=None, help="If set, will resize images and masks to [width, height, offset_bottom]")
    args = parser.parse_args()


    args.resize = [640, 256, 0]
    args.depth_map = "/home/jo/training_data/drivingstereo/depth_map"
    args.images = "/home/jo/training_data/drivingstereo/left_img"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    # calib_root_path = "/home/jo/Downloads/half-image-calib"
    # _, _, filenames = next(os.walk(calib_root_path))
    # for filename in filenames:
    #     with open(calib_root_path + "/" + filename) as f:
    #         # Create calibration matrix for P2 image from calibration data
    #         calib_lines = f.readlines()
    #         calib_lines = [line.strip().split(" ") for line in calib_lines if line.strip()]

    #         calib_101 = calib_lines[3]
    #         focal_length_101 = (float(calib_101[1]), float(calib_101[5]))
    #         pp_offset_101 = (float(calib_101[3]), float(calib_101[6]))
            
    #         calib_103 = calib_lines[11]
    #         focal_length_103 = (float(calib_103[1]), float(calib_103[5]))
    #         pp_offset_103 = (float(calib_103[3]), float(calib_103[6]))

    #         print(f"------- {filename} -------------")
    #         print(focal_length_101)
    #         print(focal_length_103)
    #         print(pp_offset_101)
    #         print(pp_offset_103)

    for folder in tqdm(next(os.walk(args.depth_map))[1]):
        curr_scene_depth_map_root = os.path.join(args.depth_map, folder)
        curr_scene_image_root = os.path.join(args.images, folder)
        _, _, filenames = next(os.walk(curr_scene_depth_map_root))
        filenames.sort() # since we have video frames
        timestamp = 0
        next_timestamp = 1
        for filename in tqdm(filenames):
            depth_file = os.path.join(curr_scene_depth_map_root, filename)
            img_file = os.path.join(curr_scene_image_root, filename[:-3] + "jpg")
            if os.path.isfile(img_file):
                img_data = cv2.imread(img_file)
                depth_data = cv2.imread(depth_file, -1) # load as is (uint16 grayscale img)
                # depth_data_org = depth_data

                if resize_img is not None:
                    depth_data, _ = resize_img(depth_data, args.resize[0], args.resize[1], args.resize[2], interpolation=cv2.INTER_NEAREST)
                    img_data, _ = resize_img(img_data, args.resize[0], args.resize[1], args.resize[2])

                img_bytes = cv2.imencode(".jpg", img_data)[1].tobytes()
                depth_bytes = cv2.imencode(".png", depth_data)[1].tobytes()

                entry = Entry(
                    img=img_bytes,
                    depth=depth_bytes,
                    content_type="image/jpg",
                    org_source="driving_stereo",
                    org_id=filename[:-3],
                    scene_token=folder,
                    timestamp=timestamp,
                    next_timestamp=next_timestamp
                )
                collection.insert_one(entry.get_dict())

                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                # ax2.imshow(cv2.cvtColor(depth_data, cv2.COLOR_BGR2RGB))
                # plt.show()

                timestamp += 1
                if next_timestamp is not None:
                    next_timestamp += 1
                    if next_timestamp == len(filenames):
                        next_timestamp = None
            else:
                print(f"ERROR: {img_file}")

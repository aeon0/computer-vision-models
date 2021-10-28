import os
import cv2
import json
import numpy as np
from pymongo import MongoClient
import argparse
from tqdm import tqdm
from numba import jit
from numba.typed import Dict
from data.label_spec import Entry
from common.utils import resize_img
import matplotlib.pyplot as plt
from data.label_spec import SEMSEG_CLASS_MAPPING


@jit(nopython=True)
def remap_mask(data, map):
    new_data = np.zeros(data.shape, dtype=np.uint8)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            px_hash = int(data[x][y][2] * 10e5 + data[x][y][1] * 10e2 + data[x][y][0])
            new_data[x][y] = map[px_hash]
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload semseg data from mapillary dataset")
    parser.add_argument("--src_path", type=str, help="Path to mapillary dataset e.g. /home/user/mapillary")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="mapillary", help="MongoDB collection")
    parser.add_argument("--dataset", type=str, default="training", help="training or validation")
    parser.add_argument("--resize", nargs='+', type=int, default=None, help="If set, will resize images and masks to [width, height, offset_bottom]")
    args = parser.parse_args()

    # args.src_path = "/home/jo/training_data/mapillary"
    # args.dataset = "validation"
    args.resize = [640, 256, -300]
    
    f = open(args.src_path + "/config_v2.0.json")
    config = json.loads(f.read())
    
    mask_map = Dict() # e.g. "250120111": [255, 0, 1] 
    for label in config["labels"]:
        # color in RGB
        hash = int(label["color"][0] * 10e5  + label["color"][1] * 10e2 + label["color"][2])
        semseg_label_color = (0, 0, 0)
        if label["name"].startswith((
            "void--car-mount",
            "void--ego-vehicle"
        )):
            semseg_label_color = SEMSEG_CLASS_MAPPING["ego_car"]
        elif label["name"].startswith((
            "marking--continuous"
        )):
            semseg_label_color = SEMSEG_CLASS_MAPPING["lane_markings"]
        elif label["name"].startswith((
            "animal--ground-animal",
            "human", "object--vehicle"
        )):
            semseg_label_color = SEMSEG_CLASS_MAPPING["movable"]
        elif label["name"].startswith((
            "construction--flat--bike-lane",
            "construction--flat--crosswalk-plain",
            "construction--flat--driveway",
            "construction--flat--parking",
            "construction--flat--parking-aisle",
            "construction--flat--pedestrian-area",
            "construction--flat--rail-track",
            "construction--flat--road",
            "construction--flat--road-shoulder",
            "construction--flat--service-lane",
            "marking--discrete",
            "marking-only",
            "object--manhole",
            "void--ground"
        )):
            semseg_label_color = SEMSEG_CLASS_MAPPING["road"]
        else:
            semseg_label_color = SEMSEG_CLASS_MAPPING["undriveable"]
        if hash not in mask_map:
            mask_map[hash] = semseg_label_color

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection + "_" + args.dataset]

    img_dir = args.src_path + "/" + args.dataset + "/images"
    mask_dir = args.src_path + "/" + args.dataset + "/v2.0/labels"
    file_list = os.listdir(img_dir)
    
    for img_name in tqdm(file_list):
        base_name = img_name.split(".")[0]
        img_path = img_dir + "/" + img_name
        mask_path = mask_dir + "/" + base_name + ".png"

        mask_data = cv2.imread(mask_path)
        img_data = cv2.imread(img_path)
        if args.resize is not None:
            mask_data, _ = resize_img(mask_data, args.resize[0], args.resize[1], args.resize[2], interpolation=cv2.INTER_NEAREST)
            img_data, _ = resize_img(img_data, args.resize[0], args.resize[1], args.resize[2])
        new_mask_data = remap_mask(mask_data, mask_map)
        mask_bytes = cv2.imencode('.png', new_mask_data)[1].tobytes()
        img_bytes = cv2.imencode('.png', img_data)[1].tobytes()
        entry = Entry(
            img=img_bytes,
            mask=mask_bytes,
            content_type="image/png",
            org_source="mapillary",
            org_id=base_name,
        )
        collection.insert_one(entry.get_dict())

        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        # ax2.imshow(cv2.cvtColor(mask_data, cv2.COLOR_BGR2RGB))
        # ax3.imshow(cv2.cvtColor(new_mask_data, cv2.COLOR_BGR2RGB))
        # plt.show()


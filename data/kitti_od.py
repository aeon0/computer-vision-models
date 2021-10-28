import os
import cv2
import math
import numpy as np
import argparse
from tqdm import tqdm
from pymongo import MongoClient
from data.label_spec import Object, Entry, OD_CLASS_MAPPING
from common.utils import calc_cuboid_from_3d, wrap_angle

# Mapping kitti classes to od_spec classes
CLASS_MAP = {"Pedestrian": "ped"}
# Ignore these kitti classes, note DontCare will be added to the ignore areas of the od spec
IGNORE_KITTY_CLASSES = ["Person_sitting", "Tram", "Misc", "DontCare"]


def class_mapper(class_name: str) -> str:
  if class_name in CLASS_MAP:
    class_name = CLASS_MAP[class_name]
  return class_name.lower()

def main(args):
  client = MongoClient(args.conn)
  collection = client[args.db][args.collection]

  label_file_list = []
  if args.label_path is not None:
    label_file_list = os.listdir(args.label_path)
  else:
    print("WARNING: Label path is None, training data will not be uploaded!")

  for filename in tqdm(label_file_list):
    kitty_id = filename[:-4]

    # check if already exists, if yes, continue with next
    # check_db_entry = collection.find_one({ "org_source": "kitti", "org_id": kitty_id})
    # if check_db_entry is not None:
    #   print("WARNING: Entry " + str(kitty_id) + " already exists, continue with next image")
    #   continue

    with open(args.label_path + "/" + filename) as f, open(args.calib_path + "/" + filename) as fc:
      raw_label_data = f.read().split("\n")[:-1]

      # load image
      img_path = args.image_path + "/" + str(kitty_id) + ".png"
      if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_bytes = cv2.imencode('.png', img)[1].tobytes()
        content_type = "image/png"
      else:
        print("WARNING: file not found: " + img_path + ", continue with next image")
        continue

      obj_list = []
      ignore_areas = []
      entry = Entry(
        img=img_bytes,
        content_type=content_type,
        org_source="kitti",
        org_id=kitty_id,
        objects=obj_list,
        ignore=ignore_areas,
        has_3D_info=True,
        has_track_info=False
      )

      # Create calibration matrix for P2 image from calibration data
      calib_lines = fc.readlines()
      calib_lines = [line.strip().split(" ") for line in calib_lines if line.strip()]
      p2_raw = calib_lines[2]
      P2 = np.array([
        [float(p2_raw[1]), float(p2_raw[2]),  float(p2_raw[3]),  float(p2_raw[4])],
        [float(p2_raw[5]), float(p2_raw[6]),  float(p2_raw[7]),  float(p2_raw[8])],
        [float(p2_raw[9]), float(p2_raw[10]), float(p2_raw[11]), float(p2_raw[12])],
        [0.0,              0.0,              0.0,                1.0],
      ])

      for obj_data in raw_label_data:
        obj_data = obj_data.split(" ")
        # decode the label info
        # [0] = object class, one of: ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Person_sitting', 'Tram', 'Misc', 'DontCare']
        # [1] = truncation
        # [2] = occlusion 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
        # [3] = alpha, observation angle of object [-pi...pi]
        # [4,5,6,7] = bbox 2D, 0-based index (left, top, right, bottom)
        # kitti camera coordinate system has z at optical axis, y as down vector, left hand system (x to the right)
        # http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf
        # [8,9,10] = dimensions (height, width, length) in [m]
        # [11,12,13] = location (x, y, z) in camera coordinates in [m]
        # [14] = rotation around y-axis in camera coordinates [-pi..pi]
        # [15] = score

        rot_angle = wrap_angle(float(obj_data[14]) + math.pi*0.5) # because parallel to optical view of camera = 90 deg
        rot_mat = np.array([
          [math.cos(rot_angle), 0.0, math.sin(rot_angle), 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [-math.sin(rot_angle), 0.0, math.cos(rot_angle), 0.0],
          [0.0, 0.0, 0.0, 1.0],
        ])
        pos_3d = np.array([float(obj_data[11]), float(obj_data[12]), float(obj_data[13]), 1.0])
        width = float(obj_data[9])
        height = float(obj_data[8])
        length = float(obj_data[10])

        kitty_class = obj_data[0]
        topLeft = [float(obj_data[4]), float(obj_data[5])]  # [x, y]
        bottomRight = [float(obj_data[6]), float(obj_data[7])]  # [x, y]
        bbox = [*topLeft, bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]]
        if kitty_class not in IGNORE_KITTY_CLASSES:
          obj_class = class_mapper(kitty_class)
          if obj_class in list(OD_CLASS_MAPPING.keys()):
            occlusion = int(obj_data[2])
            if occlusion == 3:
              occlusion = 4
            entry.objects.append(Object(
              obj_class=obj_class,
              box2d=bbox,
              box3d=calc_cuboid_from_3d(pos_3d, P2, rot_mat, width, height, length),
              box3d_valid=True,
              truncated=bool(obj_data[1]),
              occluded=occlusion,
              # 3d info
              height=float(obj_data[8]),
              width=float(obj_data[9]),
              length=float(obj_data[10]),
              orientation=rot_angle,
              x=float(obj_data[13]), # converting to autosar
              y=-float(obj_data[11]), # converting to autosar
              z=-float(obj_data[12]), # converting to autosar
            ))
          else:
            print("WARNING: Unknown mapped class " + str(obj_class) + ", continue with next image")
        elif kitty_class == "DontCare":
          # add rectangle to ignore list
          entry.ignore.append(bbox)

      # cv2.imshow("Debug", img)
      # cv2.waitKey(0)

      # upload to mongodb
      collection.insert_one(entry.get_dict())
  
  # Upload test data
  test_collection = client[args.db][args.collection + "_test"]
  if args.test_image_path is not None:
    test_images_files_list = os.listdir(args.test_image_path)
  else:
    print("WARNING: Test Image path is None, test data will not be uploaded!")
    return

  for filename in tqdm(test_images_files_list):
    kitty_id = filename[:-4]

    # check if already exists, if yes, continue with next
    check_db_entry = test_collection.find_one({ "org_source": "kitti_test", "org_id": kitty_id})
    if check_db_entry is not None:
      print("WARNING: Entry " + str(kitty_id) + " already exists, continue with next image")
      continue

    # load image
    img_path = args.test_image_path + "/" + str(kitty_id) + ".png"
    if os.path.exists(img_path):
      img = cv2.imread(img_path)
      img_bytes = cv2.imencode('.png', img)[1].tobytes()
      content_type = "image/png"
    else:
      print("WARNING: file not found: " + img_path + ", continue with next image")
      continue

    obj_list = []
    ignore_areas = []
    entry = Entry(
      img=img_bytes,
      content_type=content_type,
      org_source="kitti_test",
      org_id=kitty_id,
      objects=obj_list,
      ignore=ignore_areas,
      has_3D_info=False,
      has_track_info=False
    )

    # upload to mongodb
    test_collection.insert_one(entry.get_dict())


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Upload 2D and 3D data from kitti dataset")
  parser.add_argument("--image_path", type=str, help="Path to kitti training images e.g. /path/to/kitti/data_object_image/training/image_2")
  parser.add_argument("--label_path", type=str, help="Path to kitti labels e.g. /path/to/kitti/data_object_label/training/label_2")
  parser.add_argument("--calib_path", type=str, help="Path to kitti calibration data e.g. /path/to/kitti/data_object_calib/training/calib")
  parser.add_argument("--test_image_path", type=str, help="Path to kitti test images e.g. /path/to/kitti/data_object_image/testing/image_2")
  parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
  parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
  parser.add_argument("--collection", type=str, default="kitti", help="MongoDB collection")
  args = parser.parse_args()

  main(parser.parse_args())

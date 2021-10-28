import argparse
import cv2
import numpy as np
import math
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymongo import MongoClient
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from common.utils import calc_cuboid_from_3d, bbox_from_cuboid, wrap_angle, resize_img, Roi
from data.nuimages_od import CLASS_MAP, MyQuaternion
from data.label_spec import Object, Entry, OD_CLASS_MAPPING


def map_pointcloud_to_image(
    nusc,
    sample_token,
    pointsensor_channel,
    camera_channel,
    min_dist: float = 1.0,
    depth_map = None
):
    sample_record = nusc.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    width = im.size[0]
    height = im.size[1]
    if depth_map is None:
        depth_map = np.zeros((height, width), dtype=np.uint16)
    
    for x, y, depth in zip(points[0], points[1], coloring):
        depth_val = int(depth.real * 255.0)
        iy = int(y)
        ix = int(x)
        cv2.circle(depth_map, (ix, iy), 5, depth_val, -1)

    return depth_map


def main(args):
    args.path = "/home/jo/training_data/nuscenes/nuscenes-v1.0"
    args.resize = [640, 256, 0]

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    nusc = NuScenes(version="v1.0-mini", dataroot=args.path, verbose=True)
    nusc.list_scenes()

    for scene in tqdm(nusc.scene):
        next_sample_token = scene["first_sample_token"]

        while True:
            sample = nusc.get('sample', next_sample_token)
            next_sample_token = sample["next"]
            
            sample_data_token = sample["data"]["CAM_FRONT"]
            sample_data = nusc.get_sample_data(sample_data_token)
            cam_front_data = nusc.get('sample_data', sample_data_token)

            # Create image data
            img_path = sample_data[0]
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                nusc.render_pointcloud_in_image
                depth_map = map_pointcloud_to_image(nusc, sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT')
                # Not sure about radar data, seems to not allign with lidar data very often, leaving it out for now
                # depth_map = map_pointcloud_to_image(nusc, sample['token'], pointsensor_channel='RADAR_FRONT', camera_channel='CAM_FRONT', depth_map=depth_map)
                roi = Roi()
                if args.resize:
                    img, roi = resize_img(img, args.resize[0], args.resize[1], args.resize[2])
                    depth_map, _ = resize_img(depth_map, args.resize[0], args.resize[1], args.resize[2], cv2.INTER_NEAREST)
                img_bytes = cv2.imencode('.jpeg', img)[1].tobytes()
                depth_bytes = cv2.imencode(".png", depth_map)[1].tobytes()
                content_type = "image/jpeg"
            else:
                print("WARNING: file not found: " + img_path + ", continue with next image")
                continue

            # Get sensor extrinsics, Not sure why roll and yaw seem to be PI/2 off compared to nuImage calibarted sensor
            sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            q = MyQuaternion(sensor["rotation"][0], sensor["rotation"][1], sensor["rotation"][2], sensor["rotation"][3])
            roll  = math.atan2(2.0 * (q.z * q.y + q.w * q.x) , 1.0 - 2.0 * (q.x * q.x + q.y * q.y)) + math.pi * 0.5
            pitch = math.asin(2.0 * (q.y * q.w - q.z * q.x)) 
            yaw   = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x)) + math.pi * 0.5
            # print(sensor["translation"])
            # print(f"Pitch: {pitch*57.2} Yaw: {yaw*57.2} Roll: {roll*57.2}")

            # Sensor calibration is static, pose would be dynamic. TODO: Somehow also add some sort of cam to cam motion to be learned
            # ego_pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            # q = MyQuaternion(ego_pose["rotation"][0], ego_pose["rotation"][1], ego_pose["rotation"][2], ego_pose["rotation"][3])
            # roll  = math.atan2(2.0 * (q.z * q.y + q.w * q.x) , 1.0 - 2.0 * (q.x * q.x + q.y * q.y)) + math.pi * 0.5
            # pitch = math.asin(2.0 * (q.y * q.w - q.z * q.x)) 
            # yaw   = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x)) + math.pi * 0.5
            # print(ego_pose["translation"])
            # print(f"Pitch: {pitch*57.2} Yaw: {yaw*57.2} Roll: {roll*57.2}")

            entry = Entry(
                img=img_bytes,
                content_type=content_type,
                depth=depth_bytes,
                org_source="nuscenes",
                org_id=img_path,
                objects=[],
                ignore=[],
                has_3D_info=True,
                has_track_info=True,
                sensor_valid=True,
                yaw=wrap_angle(yaw),
                roll=wrap_angle(roll),
                pitch=wrap_angle(pitch),
                translation=sensor["translation"],
                scene_token=sample["scene_token"],
                timestamp=sample["timestamp"]
            )

            labels = sample_data[1]
            idx_counter = 0
            for box in labels:
                if box.name in CLASS_MAP.keys():
                    box.translate(np.array([0, box.wlh[2] / 2, 0])) # translate center center to bottom center
                    pos_3d = np.array([*box.center, 1.0])
                    cam_mat = np.hstack((sample_data[2], np.zeros((3, 1))))
                    cam_mat[0][2] += roi.offset_left
                    cam_mat[1][2] += roi.offset_top
                    cam_mat *= roi.scale
                    cam_mat[2][2] = 1.0
                    # create rotation matrix, somehow using the rotation matrix directly from nuscenes does not work
                    # instead, we calc the rotation as it is in kitti and use the same code
                    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                    yaw = -np.arctan2(v[2], v[0])
                    rot_angle = wrap_angle(float(yaw) + math.pi * 0.5) # because parallel to optical view of camera = 90 deg
                    rot_mat = np.array([
                        [math.cos(rot_angle), 0.0, math.sin(rot_angle), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-math.sin(rot_angle), 0.0, math.cos(rot_angle), 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    pos_2d = np.matmul(cam_mat, pos_3d)
                    pos_2d /= pos_2d[2]
                    box3d = calc_cuboid_from_3d(pos_3d, cam_mat, rot_mat, box.wlh[0], box.wlh[2], box.wlh[1])
                    box2d = bbox_from_cuboid(box3d)

                    annotation = nusc.get("sample_annotation", box.token)
                    instance_token = annotation["instance_token"]

                    entry.objects.append(Object(
                        obj_class=CLASS_MAP[box.name],
                        box2d=box2d,
                        box3d=box3d,
                        box3d_valid=True,
                        instance_token=instance_token,
                        truncated=None,
                        occluded=None,
                        width=box.wlh[0],
                        length=box.wlh[1],
                        height=box.wlh[2],
                        orientation=rot_angle,
                        # converted to autosar coordinate system
                        x=pos_3d[2],
                        y=-pos_3d[0],
                        z=-pos_3d[1]
                    ))

                    # For debugging show the data in the image
                    box2d = list(map(int, box2d))
                    cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), 3, (255, 0, 0))
                    cv2.rectangle(img, (box2d[0], box2d[1]), (box2d[0] + box2d[2], box2d[1] + box2d[3]), (255, 255, 0), 1)
                    cv2.putText(img,f"{idx_counter}", (box2d[0], box2d[1] + int(box2d[3])), 0, 0.4, (255, 255, 255))
                    idx_counter += 1
                    print(f"{idx_counter}: {math.degrees(rot_angle)}")

            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            max_val = np.amax(depth_map)
            ax2.imshow(depth_map, cmap="gray", vmin=1, vmax=10000)
            plt.show()

            # collection.insert_one(entry.get_dict())

            if next_sample_token == "":
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload 2D and 3D data from nuscenes dataset")
    parser.add_argument("--path", type=str, help="Path to nuscenes data, should contain samples/CAMERA_FRONT/*.jpg and v1.0-trainval/*.json folder e.g. /path/to/nuscenes")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuscenes_train", help="MongoDB collection")
    parser.add_argument("--resize", nargs='+', type=int, default=None, help="If set, will resize images and masks to [width, height, offset_bottom]")

    main(parser.parse_args())

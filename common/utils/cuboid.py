import numpy as np
import cv2
from typing import List


def bbox_from_cuboid(cuboid: List[float]) -> List[float]:
    """
    Convert a image cuboid to a bounding box
    :param cuboid (List[float]): With points in x, y, 1d list [back_left_top, back_left_bottom, back_right_bottom_2d,
                back_right_top_2d, front_left_top_2d, front_left_bottom_2d, front_right_bottom_2d, front_right_top_2d]
    :return: List[float]: List of bounding box [top_left_x, top_left_y, width, height]
    """
    np_cuboid = np.array(cuboid)
    np_cuboid = np.reshape(np_cuboid, (-1, 2))
    max_vals = np.amax(np_cuboid, axis=0)
    min_vals = np.amin(np_cuboid, axis=0)
    bbox = [*min_vals, max_vals[0]- min_vals[0], max_vals[1]- min_vals[1]]
    return bbox

def calc_cuboid_from_3d(
        center_ground_3d: np.ndarray,
        cam_mat: np.ndarray,
        obj_rot_mat: np.ndarray,
        width: float,
        height: float,
        length: float,
        debug_img: np.ndarray = None
    ) -> List[float]:
    """
    Calculate the 3D bounding box in an image from the 3D object information
    :param center_ground_3d (np.ndarray): 3D point of the center ground coordinate of the obj in cam coordinate system
    :param cam_mat (np.ndarray): Camera matrix that will convert 3D points (camera coordinate system) to image
    :param obj_rot_mat (np.ndarray): Rotation matrix of the object
    :param width (float): Width of object
    :param height (float): Height of object
    :param length (float): Length of object
    :param roi (Roi): Region of intereset in case image was resized
    :param debug_img (np.ndarray, optional): If a cv2 image is provided, the 3D box will be drawn. Defaults to None.
    :return: List[float]: Cuboid with points in x, y, 1d list [back_left_top, back_left_bottom, back_right_bottom_2d,
        back_right_top_2d, front_left_top_2d, front_left_bottom_2d, front_right_bottom_2d, front_right_top_2d]
    """
    pos_2d = np.matmul(cam_mat, center_ground_3d)
    pos_2d /= pos_2d[2]

    # create object in object coordinates
    height = float(height)
    half_width = float(width) * 0.5
    half_length = float(length) * 0.5
    back_left_bottom_3d = np.array(  [ half_width, 0.0,     -half_length, 1.0])
    back_left_top_3d = np.array(     [ half_width, -height, -half_length, 1.0])
    back_right_bottom_3d = np.array( [-half_width, 0.0,     -half_length, 1.0])
    back_right_top_3d = np.array(    [-half_width, -height, -half_length, 1.0])
    front_left_bottom_3d = np.array( [ half_width, 0.0,      half_length, 1.0])
    front_left_top_3d = np.array(    [ half_width, -height,  half_length, 1.0])
    front_right_bottom_3d = np.array([-half_width, 0.0,      half_length, 1.0])
    front_right_top_3d = np.array(   [-half_width, -height,  half_length, 1.0])

    # rotate corner positions around the objects up vector (y-axis)
    back_left_bottom_3d = np.matmul(obj_rot_mat, back_left_bottom_3d)
    back_left_top_3d = np.matmul(obj_rot_mat, back_left_top_3d)
    back_right_bottom_3d = np.matmul(obj_rot_mat, back_right_bottom_3d)
    back_right_top_3d = np.matmul(obj_rot_mat, back_right_top_3d)
    front_left_bottom_3d = np.matmul(obj_rot_mat, front_left_bottom_3d)
    front_left_top_3d = np.matmul(obj_rot_mat, front_left_top_3d)
    front_right_bottom_3d = np.matmul(obj_rot_mat, front_right_bottom_3d)
    front_right_top_3d = np.matmul(obj_rot_mat, front_right_top_3d)

    # move object from object coordinate system to camera coordinate system
    back_left_bottom_3d += center_ground_3d
    back_left_top_3d += center_ground_3d
    back_right_bottom_3d += center_ground_3d
    back_right_top_3d += center_ground_3d
    front_left_bottom_3d += center_ground_3d
    front_left_top_3d += center_ground_3d
    front_right_bottom_3d += center_ground_3d
    front_right_top_3d += center_ground_3d

    # convert to 2d image coordinates, note that left and right are changed on conversion since x in 3d -> -x in 2d
    back_right_bottom_2d = np.matmul(cam_mat, back_left_bottom_3d)
    back_right_bottom_2d /= back_right_bottom_2d[2]
    back_right_top_2d = np.matmul(cam_mat, back_left_top_3d)
    back_right_top_2d /= back_right_top_2d[2]
    back_left_bottom_2d = np.matmul(cam_mat, back_right_bottom_3d)
    back_left_bottom_2d /= back_left_bottom_2d[2]
    back_left_top_2d = np.matmul(cam_mat, back_right_top_3d)
    back_left_top_2d /= back_left_top_2d[2]
    front_right_bottom_2d = np.matmul(cam_mat, front_left_bottom_3d)
    front_right_bottom_2d /= front_right_bottom_2d[2]
    front_right_top_2d = np.matmul(cam_mat, front_left_top_3d)
    front_right_top_2d /= front_right_top_2d[2]
    front_left_bottom_2d = np.matmul(cam_mat, front_right_bottom_3d)
    front_left_bottom_2d /= front_left_bottom_2d[2]
    front_left_top_2d = np.matmul(cam_mat, front_right_top_3d)
    front_left_top_2d /= front_left_top_2d[2]

    # draw 3d box for debugging
    if debug_img is not None:
        cv2.circle(debug_img, (int(pos_2d[0]), int(pos_2d[1])), 3, (255, 0, 0))
        # back trapezoid
        color = (0, 255, 0)
        cv2.line(debug_img, (int(back_left_top_2d[0]),     int(back_left_top_2d[1])),
                            (int(back_left_bottom_2d[0]),  int(back_left_bottom_2d[1])), color)
        cv2.line(debug_img, (int(back_left_bottom_2d[0]),  int(back_left_bottom_2d[1])),
                            (int(back_right_bottom_2d[0]), int(back_right_bottom_2d[1])), color)
        cv2.line(debug_img, (int(back_right_bottom_2d[0]), int(back_right_bottom_2d[1])),
                            (int(back_right_top_2d[0]),    int(back_right_top_2d[1])), color)
        cv2.line(debug_img, (int(back_right_top_2d[0]),    int(back_right_top_2d[1])),
                            (int(back_left_top_2d[0]),     int(back_left_top_2d[1])), color)
        # front trapezoid
        color = (0, 0, 255)
        cv2.line(debug_img, (int(front_left_top_2d[0]),     int(front_left_top_2d[1])),
                            (int(front_left_bottom_2d[0]),  int(front_left_bottom_2d[1])), color)
        cv2.line(debug_img, (int(front_left_bottom_2d[0]),  int(front_left_bottom_2d[1])),
                            (int(front_right_bottom_2d[0]), int(front_right_bottom_2d[1])), color)
        cv2.line(debug_img, (int(front_right_bottom_2d[0]), int(front_right_bottom_2d[1])),
                            (int(front_right_top_2d[0]),    int(front_right_top_2d[1])), color)
        cv2.line(debug_img, (int(front_right_top_2d[0]),    int(front_right_top_2d[1])),
                            (int(front_left_top_2d[0]),     int(front_left_top_2d[1])), color)
        # trapezoid connections
        color = (255, 255, 255)
        cv2.line(debug_img, (int(front_left_top_2d[0]),     int(front_left_top_2d[1])),
                            (int(back_left_top_2d[0]),      int(back_left_top_2d[1])), color)
        cv2.line(debug_img, (int(front_right_top_2d[0]),    int(front_right_top_2d[1])),
                            (int(back_right_top_2d[0]),     int(back_right_top_2d[1])), color)
        cv2.line(debug_img, (int(front_left_bottom_2d[0]),  int(front_left_bottom_2d[1])),
                            (int(back_left_bottom_2d[0]),   int(back_left_bottom_2d[1])), color)
        cv2.line(debug_img, (int(front_right_bottom_2d[0]), int(front_right_bottom_2d[1])),
                            (int(back_right_bottom_2d[0]),  int(back_right_bottom_2d[1])), color)

    # [0,1]: back_left_top, [2,3]: back_left_bottom, [4,5]: back_right_bottom, [6,7]: back_right_top,
    # [8,9]: front_left_top, [10,11]: front_left_bottom, [12,13]: front_right_bottom, [14,15]: front_right_top
    return [*back_left_top_2d[:2], *back_left_bottom_2d[:2], *back_right_bottom_2d[:2], *back_right_top_2d[:2],
            *front_left_top_2d[:2], *front_left_bottom_2d[:2], *front_right_bottom_2d[:2], *front_right_top_2d[:2]]

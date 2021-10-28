from dataclasses import dataclass
from typing import List
from collections import OrderedDict

# Colours are in BGR!
OD_CLASS_MAPPING = OrderedDict([
  ("car", (96, 96, 192)), # red
  ("truck", (96, 192, 192)), # yellow
  ("van", (128, 192, 96)), # turquoise
  ("motorbike", (194, 96, 64)), # blue
  ("cyclist", (64, 194, 128)), # green
  ("ped", (196, 64, 196)), # pink
])
OD_CLASS_IDX = {k: pos for pos, k in enumerate(OD_CLASS_MAPPING)}

# These are the same as from the comma10k label spec (https://github.com/commaai/comma10k)
# Hex values are in RGB, Tuples are in BGR!
#  1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
#  2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
#  3 - #808060 - undrivable
#  4 - #00ff66 - movable (vehicles and people/animals)
#  5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
SEMSEG_CLASS_MAPPING = OrderedDict([
  ("road", (32, 32, 64)), # dark red
  ("lane_markings", (0, 0, 255)), # red
  ("undriveable", (96, 128, 128)), # green-brownish
  ("movable", (102, 255, 0)), # green
  ("ego_car", (255, 0, 204)), # purple
])
SEMSEG_CLASS_IDX = {k: pos for pos, k in enumerate(SEMSEG_CLASS_MAPPING)}


@dataclass
class Object:
  obj_class: str # describe class as string from OD_CLASS_MAPPING
  box2d: List[float] # [x, y, width, height] top left corner, width, height
  box3d: List[float] # [0,1]: back_top_left, [2,3]: back_bottom_left, [4,5]: back_bottom_right, [6,7]: back_top_right,
                     # [8,9]: front_top_left, [10,11]: front_bottom_left, [12,13]: front_bottom_right, [14,15]: front_top_right
                     # points are stored in with [x, y] in pixel
  box3d_valid: bool = False

  truncated: bool = False  # true if image leaves image boundaries
  occluded: int = 4 # 0 = fully visible (0-40%), 1 = partly occluded (40-60%), 2 = largely occluded (60-80%), 3 = almost all occluded (80-100%) 4 = unknown

  # properties only apply if has_3D_info on the OdEntry is true
  height: float = None # in [m]
  width: float = None # in [m]
  length: float = None # in [m]
  # positions and rotation are in camera coordinate system with autosar axis (x: optical axis, z: up vector, left handed system thus y to left)
  orientation: float = None # rotation around up vector (z-axis) of the object in [rad] in range of [-pi..pi] where 0 means same direction as camera
  x: float = None # position on the x-axis in [m] in camera coordinate system
  y: float = None  # position on the y-axis in [m] in camera coordinate system
  z: float = None  # position on the z-axis in [m] in camera coordinate system

  # properties only available if has_track_info on the OdEntry is true
  instance_token: int = None # token of the track

  def get_dict(self):
    return self.__dict__

@dataclass
class Entry:
  img: bytes
  content_type: str # e.g. "image/png", "image/jpg", etc.
  org_source: str # describe original source of the data e.g. KITTI
  org_id: str # describe original identifier
  mask: bytes = None
  depth: bytes = None
  objects: List[Object] = None
  ignore: List[List[float]] = None # bounding boxes that should be ignored [x, y, width, height] with (x, y) being top left
  has_3D_info: bool = False # flag weather 3D info is available on objects
  has_track_info: bool = False # flag weather the entry has tracking info
  scene_token: str = None # token to which scene this entry belongs to (only applicable for tracking data)
  timestamp: str = None # timestamp for this frame (only applicable for tracking data)
  next_timestamp: str = None # timestamp of next frame, if None there is no next frame

  # Sensor Info
  sensor_valid: bool = False
  pitch: float = 0.0
  yaw: float = 0.0
  roll: float = 0.0
  translation: List[float] = None # x, y, z in autosar coordinate system, bumper origin
  focal_length = (float, float) # focal length (x, y) in [px]
  principal_point_offset = (float, float) # principal point offset (x, y) in [px]

  def get_dict(self):
    data_as_dict = self.__dict__
    if data_as_dict["objects"] is not None:
      for i, obj in enumerate(data_as_dict["objects"]):
        data_as_dict["objects"][i] = obj.get_dict()
    return data_as_dict

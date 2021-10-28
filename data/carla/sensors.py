import carla
import numpy as np
import cv2


class Sensors():
    def __init__(self, carla_world: carla.World, ego_vehicle: carla.Vehicle):
        # Params for images
        self.img_width = 1280 # in [px]
        self.img_height = 720 # in [px]
        self.fov_horizontal = 120 # in [deg]
        self.display_width = 640 # in [px]
        self.display_height = 360 # in [px]

        # Params for static camera calibration
        self.translation_x = 0.4 # in [m] from center of car
        self.translation_z = 1.5 # in [m] from bottom of car
        self.translation_y = 0.0 # in [m] from center of car
        self.pitch = 0.0 # in [deg]
        self.roll = 0.0 # in [deg]
        self.yaw = 0.0 # in [deg]

        # Images that are stored on disk per frame
        self.rgb_img: np.array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.depth_img: np.array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.semseg_img: np.array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Images that are displayed in pygame for visualization (note spectator img is just needed for visu)
        self.display_spectator_img: np.array = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        self.display_rgb_img: np.array = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        self.display_depth_img: np.array = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        self.display_semseg_img: np.array = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Create sensors
        blueprint_library: carla.BlueprintLibrary = carla_world.get_blueprint_library()
        cam_transform = carla.Transform(
            carla.Location(x=self.translation_x, z=self.translation_z, y=self.translation_y),
            carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)
        )

        # Attach dummy rgb sensor to follow the ego vehicle (only for visulization)
        spectator_bp = blueprint_library.find('sensor.camera.rgb')
        self._add_attr_to_blueprint(spectator_bp)
        spectator_transform = carla.Transform(carla.Location(x=-6, z=3))
        self.spectator: carla.Actor = carla_world.spawn_actor(spectator_bp, spectator_transform, attach_to=ego_vehicle)
        self.spectator.listen(self._read_spectator_img)

        # Attach rgb cam sensors to ego vehicle
        cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        self._add_attr_to_blueprint(cam_rgb_bp)
        self.cam_rgb: carla.Actor = carla_world.spawn_actor(cam_rgb_bp, cam_transform, attach_to=ego_vehicle)
        self.cam_rgb.listen(self._read_rgb_img)

        # Attach depth sensor to ego vehicle
        cam_depth_bp = blueprint_library.find("sensor.camera.depth")
        self._add_attr_to_blueprint(cam_depth_bp)
        self.cam_depth: carla.Actor = carla_world.spawn_actor(cam_depth_bp, cam_transform, attach_to=ego_vehicle)
        self.cam_depth.listen(self._read_depth_img)

        # Attach semseg sensor to ego vehicle
        cam_semseg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        self._add_attr_to_blueprint(cam_semseg_bp)
        self.cam_semseg: carla.Actor = carla_world.spawn_actor(cam_semseg_bp, cam_transform, attach_to=ego_vehicle)
        self.cam_semseg.listen(self._read_semseg_img)

        # TODO: Save all of it also to disk each frame
        # e.g. camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame)

    def destroy(self):
        self.spectator.destroy()
        self.cam_rgb.destroy()
        self.cam_depth.destroy()
        self.cam_semseg.destroy()

    def _add_attr_to_blueprint(self, bp: carla.ActorBlueprint):
        bp.set_attribute('image_size_x', str(self.img_width))
        bp.set_attribute('image_size_y', str(self.img_height))
        bp.set_attribute('fov', str(self.fov_horizontal))

    def _read_spectator_img(self, image: carla.Image):
        np_raw = np.array(image.raw_data).astype('uint8')
        tmp_img = np_raw.reshape((image.height, image.width, 4))
        tmp_img = tmp_img[:, :, :3]
        tmp_img = tmp_img[:, :, ::-1]
        self.display_spectator_img = cv2.resize(tmp_img, (self.display_width, self.display_height))

    def _read_rgb_img(self, image):
        np_raw = np.array(image.raw_data).astype('uint8')
        tmp_img = np_raw.reshape((image.height, image.width, 4))
        tmp_img = tmp_img[:, :, :3]
        tmp_img = tmp_img[:, :, ::-1]
        self.rgb_img = cv2.resize(tmp_img, (self.img_width, self.img_height))
        self.display_rgb_img = cv2.resize(self.rgb_img, (self.display_width, self.display_height))

    def _read_depth_img(self, image):
        np_raw = np.array(image.raw_data).astype('uint8')
        tmp_img = np_raw.reshape((image.height, image.width, 4))
        tmp_img = tmp_img[:, :, :3]
        tmp_img = tmp_img[:, :, ::-1]
        tmp_depth = np.zeros((image.height, image.width), dtype=np.float32)
        # The depth is stored in the RGB values using its full resolution, scaling it down to one grayscale value
        tmp_depth  = ((tmp_img[:,:,0] +  tmp_img[:,:,1]*256.0 +  tmp_img[:,:,2]*256.0*256.0)/(256.0*256.0*256.0 - 1))
        # The depth far clip is at 1000m = 1.0f, no need to look that far, 250m = 0.25f is enough and increases resolution
        tmp_depth = np.clip(tmp_depth, 0.0, 0.25)
        tmp_depth = tmp_depth * (1.0 / 0.25)
        tmp_depth = np.clip(tmp_depth, 0.0, 1.0)
        # Rescale with log to have better resolution in near range
        logdepth = np.ones(tmp_depth.shape) + (np.log(tmp_depth) / 4.7)
        logdepth = np.clip(logdepth, 0.0, 1.0)
        logdepth *= 255.0
        logdepth = cv2.resize(logdepth, (self.img_width, self.img_height))
        self.depth_img = np.stack((logdepth.astype(np.uint8),)*3, axis=-1)
        self.display_depth_img = cv2.resize(self.depth_img, (self.display_width, self.display_height))

    def _read_semseg_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        np_raw = np.array(image.raw_data).astype('uint8')
        tmp_img = np_raw.reshape((image.height, image.width, 4))
        tmp_img = tmp_img[:, :, :3]
        tmp_img = tmp_img[:, :, ::-1]
        self.semseg_img = cv2.resize(tmp_img, (self.img_height, self.img_width))
        self.display_semseg_img = cv2.resize(self.semseg_img, (self.display_width, self.display_height))

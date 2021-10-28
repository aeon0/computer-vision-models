import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams
import numpy as np
from pymongo import MongoClient
import cv2
import matplotlib.pyplot as plt
from common.utils import resize_img
from models.dmds_ref import regularizers, resampler, intrinsics_utils
from models.depth.loss import DepthLoss


class DmdsLoss:
    def __init__(self, params):
        self.params: DmdsParams = params

        self.loss_vals = {
            # "depth_abs": 0,
            # "depth_smooth": 0,
            # "depth_var": 0,
            "mm_sparsity": 0,
            "mm_smooth": 0,
            # "depth": 0,
            "rgb": 0,
            "ssim": 0,
            "rot": 0,
            "tran": 0,
        }

        # some debug data
        self.resampled_img1 = None
        self.warp_mask = None
        self.depth_weights = None

    def norm(self, x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    def expand_dims_twice(self, x, dim):
        return tf.expand_dims(tf.expand_dims(x, dim), dim)

    def combine(self, rot_mat1, trans_vec1, rot_mat2, trans_vec2):
        r2r1 = tf.matmul(rot_mat2, rot_mat1)
        r2t1 = tf.matmul(rot_mat2, tf.expand_dims(trans_vec1, -1))
        r2t1 = tf.squeeze(r2t1, axis=-1)
        return r2r1, r2t1 + trans_vec2

    def construct_rotation_matrix(self, rot):
        sin_angles = tf.sin(rot)
        cos_angles = tf.cos(rot)
        # R = R_z * R_y * R_x
        sin_angles.shape.assert_is_compatible_with(cos_angles.shape)
        sx, sy, sz = tf.unstack(sin_angles, axis=-1)
        cx, cy, cz = tf.unstack(cos_angles, axis=-1)
        m00 = cy * cz
        m01 = (sx * sy * cz) - (cx * sz)
        m02 = (cx * sy * cz) + (sx * sz)
        m10 = cy * sz
        m11 = (sx * sy * sz) + (cx * cz)
        m12 = (cx * sy * sz) - (sx * cz)
        m20 = -sy
        m21 = sx * cy
        m22 = cx * cy
        matrix = tf.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
        output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)

    def warp_it(self, depth, translation, rotation, intrinsic_mat, intrinsic_mat_inv):
        _, height, width = tf.unstack(tf.shape(depth))
        grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)
        grid = tf.cast(grid, tf.float32)

        cam_coords = tf.einsum('bij,jhw,bhw->bihw', intrinsic_mat_inv, grid, depth)
        xyz = (tf.einsum('bij,bjk,bkhw->bihw', intrinsic_mat, rotation, cam_coords) + tf.einsum('bij,bhwj->bihw', intrinsic_mat, translation))

        x, y, z = tf.unstack(xyz, axis=1)
        pixel_x = x / z
        pixel_y = y / z

        def _tensor(x):
            return tf.cast(tf.convert_to_tensor(x), tf.float32)

        x_not_underflow = pixel_x >= 0.0
        y_not_underflow = pixel_y >= 0.0
        x_not_overflow = pixel_x < _tensor(width - 1)
        y_not_overflow = pixel_y < _tensor(height - 1)
        z_positive = z > 0.0
        x_not_nan = tf.math.logical_not(tf.compat.v1.is_nan(pixel_x))
        y_not_nan = tf.math.logical_not(tf.compat.v1.is_nan(pixel_y))
        not_nan = tf.logical_and(x_not_nan, y_not_nan)
        not_nan_mask = tf.cast(not_nan, tf.float32)
        pixel_x = tf.math.multiply_no_nan(pixel_x, not_nan_mask)
        pixel_y = tf.math.multiply_no_nan(pixel_y, not_nan_mask)
        pixel_x = tf.clip_by_value(pixel_x, 0.0, _tensor(width - 1))
        pixel_y = tf.clip_by_value(pixel_y, 0.0, _tensor(height - 1))
        mask_stack = tf.stack([x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow, z_positive, not_nan], axis=0)
        mask = tf.reduce_all(mask_stack, axis=0)
        mask = tf.cast(mask, tf.float32)

        return pixel_x, pixel_y, z, mask

    def calc_warp_error(self, img0, img1, x1, px, py, z, mask):
        frame1_rgbd = tf.concat([img1, x1], axis=-1)
        frame1_rgbd_resampled = resampler.resampler_with_unstacked_warp(frame1_rgbd, px, py)
        img1_resampled, x1_resampled = tf.split(frame1_rgbd_resampled, [3, 1], axis=-1)
        x1_resampled = tf.squeeze(x1_resampled, axis=-1)

        mask = tf.stop_gradient(mask)
        frame0_closer_to_camera = tf.cast(tf.less_equal(z, x1_resampled), tf.float32)
        frame0_closer_to_camera *= mask
        # frame0_closer_to_camera = tf.cast(mask, tf.float32)
        n = tf.reduce_sum(frame0_closer_to_camera)
        depth_l1_diff = tf.abs(x1_resampled - z)
        depth_error = tf.reduce_mean(tf.math.multiply_no_nan(depth_l1_diff, frame0_closer_to_camera))

        rgb_l1_diff = tf.abs(img1_resampled - img0)
        rgb_error = tf.reduce_sum(tf.math.multiply_no_nan(rgb_l1_diff, tf.expand_dims(frame0_closer_to_camera, -1)))
        rgb_error = tf.cond(tf.greater(n, 0), lambda: rgb_error / n, lambda: 2550.0)

        frame0_closer_to_camera_3c = tf.stack([frame0_closer_to_camera] * 3, axis=-1)
        self.resampled_img1 = tf.math.multiply_no_nan(img1_resampled, frame0_closer_to_camera_3c)
        self.warp_mask = frame0_closer_to_camera

        def weighted_average(x, w, epsilon=1.0):
            weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
            sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)
            return weighted_sum / (sum_of_weights + epsilon)

        def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
            def _avg_pool3x3(x):
                return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

            def weighted_avg_pool3x3(z):
                wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
                return wighted_avg * inverse_average_pooled_weight

            weight = tf.expand_dims(weight, -1)
            average_pooled_weight = _avg_pool3x3(weight)
            weight_plus_epsilon = weight + weight_epsilon
            inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

            mu_x = weighted_avg_pool3x3(x)
            mu_y = weighted_avg_pool3x3(y)
            sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
            sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
            sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
            if c1 == float('inf'):
                ssim_n = (2 * sigma_xy + c2)
                ssim_d = (sigma_x + sigma_y + c2)
            elif c2 == float('inf'):
                ssim_n = 2 * mu_x * mu_y + c1
                ssim_d = mu_x**2 + mu_y**2 + c1
            else:
                ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
                ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            result = ssim_n / ssim_d
            return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight

        sec_moment = weighted_average(tf.square(x1_resampled - z), frame0_closer_to_camera) + 1e-4
        depth_proximity_weight = tf.math.multiply_no_nan(sec_moment * 
            tf.math.reciprocal_no_nan(tf.square(x1_resampled - z) + sec_moment), tf.cast(mask, tf.float32))
        depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)
        self.depth_weights = depth_proximity_weight

        ssim_error, avg_weight = weighted_ssim(img1_resampled, img0, depth_proximity_weight, c1=float('inf'), c2=9e-6)
        n_ssim = tf.reduce_sum(avg_weight)
        ssim_error = tf.reduce_sum(tf.math.multiply_no_nan(ssim_error, avg_weight))
        ssim_error = tf.cond(tf.greater(n_ssim, 0), lambda: ssim_error / n_ssim, lambda: 10)

        self.loss_vals["rgb"] = self.params.rgb_cons * rgb_error
        self.loss_vals["depth"] = self.params.depth_cons * depth_error
        self.loss_vals["ssim"] = self.params.ssim_cons * ssim_error

    def calc_motion_field_consistency_loss(self, T, T_inv, rot, rot_inv, mask, px, py):
        T_inv_resampled = resampler.resampler_with_unstacked_warp(T_inv, tf.stop_gradient(px), tf.stop_gradient(py), safe=False)
        
        rot_field = tf.broadcast_to(self.expand_dims_twice(rot, -2), tf.shape(T))
        rot_field_inv = tf.broadcast_to(self.expand_dims_twice(rot_inv, -2), tf.shape(T_inv))
        R_mat = self.construct_rotation_matrix(rot_field)
        R_mat_inv = self.construct_rotation_matrix(rot_field_inv)

        rot_unit, trans_zero = self.combine(R_mat_inv, T_inv_resampled, R_mat, T)
        eye = tf.eye(3, batch_shape=tf.shape(rot_unit)[:-2])

        rot_error = tf.reduce_mean(tf.square(rot_unit - eye), axis=(3, 4))
        rot1_scale = tf.reduce_mean(tf.square(R_mat - eye), axis=(3, 4))
        rot2_scale = tf.reduce_mean(tf.square(R_mat_inv - eye), axis=(3, 4))
        rot_error /= (1e-10 + rot1_scale + rot2_scale)
        rotation_error = tf.reduce_mean(rot_error)
        self.loss_vals["rot"] = rotation_error * self.params.rot_cyc

        t = tf.math.multiply_no_nan(mask, self.norm(trans_zero))
        translation_error = tf.reduce_mean(t / (1e-10 + self.norm(T) + self.norm(T_inv_resampled)))
        self.loss_vals["tran"] = translation_error * self.params.tran_cyc

    def calc(self, img0, img1, depth0, depth1, obj_tran, obj_tran_inv, bg_tran, bg_tran_inv, rot, rot_inv, K, gt_x0, gt_x1, step_number):
        # obj_tran *= tf.clip_by_value(((step_number - 5000.0) / 10000.0), 0.0, 1.0)
        obj_tran *= 0.0

        np_rot = rot.numpy()
        np_obj_tran = obj_tran.numpy()
        np_bg_tran = bg_tran.numpy()
        bg_tran *= 0.0
        rot *= 0.0
        bg_tran_inv *= 0.0
        rot_inv *= 0.0

        # some attempt to make sure rotation and translation are not getting too big
        # bg_abs = tf.abs(bg_tran)
        # bg_mask = tf.cast(tf.greater(bg_abs, 5.0), tf.float32)
        # n_bg = tf.reduce_sum(bg_mask)
        # bg_loss = tf.reduce_sum(bg_mask * (bg_abs - 5.0))
        # self.loss_vals["bg_loss"] = tf.cond(tf.greater(n_bg, 0), lambda: bg_loss / n_bg, lambda: 0)
        # rot_abs = tf.abs(rot)
        # rot_mask = tf.cast(tf.greater(rot_abs, 0.3), tf.float32)
        # n_rot = tf.reduce_sum(rot_mask)
        # rot_loss = tf.reduce_sum(rot_mask * (rot_abs - 0.3))
        # self.loss_vals["rot_loss"] = tf.cond(tf.greater(n_rot, 0), lambda: rot_loss / n_rot, lambda: 0)

        # Data generation
        # -------------------------------
        # resize since we output a smaller resudial_translation map
        obj_tran = tf.image.resize(obj_tran, img0.shape[1:3], method='nearest')
        obj_tran_inv = tf.image.resize(obj_tran_inv, img0.shape[1:3], method='nearest')

        # to avoid division by zero (final depth map layer has relu activation)
        depth_mask0 = tf.cast(tf.greater(depth0, 0.1), tf.float32)
        depth_mask1 = tf.cast(tf.greater(depth1, 0.1), tf.float32)
        depth_mask = depth_mask0 * depth_mask1
        depth_mask = tf.squeeze(depth_mask, axis=-1)
        depth0 += 0.02
        depth1 += 0.02

        T = obj_tran + bg_tran
        T_inv = obj_tran_inv + bg_tran_inv

        rot = tf.squeeze(rot)
        rot_inv = tf.squeeze(rot_inv)
        R = self.construct_rotation_matrix(rot)
        R_inv = self.construct_rotation_matrix(rot_inv)

        K_inv = intrinsics_utils.invert_intrinsics_matrix(K)

        # Depth regulizers
        # -------------------------------
        # mean_depth = tf.reduce_mean(depth1)
        # depth_var = tf.reduce_mean(tf.square(depth1 / mean_depth - 1.0))
        # self.loss_vals["depth_var"] = tf.math.reciprocal_no_nan(depth_var) * self.params.var_depth

        # disp = tf.math.reciprocal_no_nan(depth1)
        # mean_disp = tf.reduce_mean(disp, axis=[1, 2, 3], keepdims=True)
        # self.loss_vals["depth_smooth"] = regularizers.joint_bilateral_smoothing(disp * tf.math.reciprocal_no_nan(mean_disp), img1) * self.params.depth_smoothing

        self.loss_vals["depth_abs"] = DepthLoss.calc(tf.concat([gt_x0, gt_x1], axis=0), depth0) * 10.0

        # Motionmap regulizers
        # -------------------------------
        normalized_trans = regularizers.normalize_motion_map(obj_tran, T)
        self.loss_vals["mm_smooth"] = regularizers.l1smoothness(normalized_trans) * self.params.mot_smoothing
        self.loss_vals["mm_sparsity"] = regularizers.sqrt_sparsity(normalized_trans) * self.params.mot_drift

        # Cyclic and RGB Loss
        # -------------------------------
        px, py, z, mask = self.warp_it(tf.squeeze(depth0, axis=-1), T, R, K, K_inv)
        mask *= depth_mask
        depth1 = tf.stop_gradient(depth1)

        self.calc_warp_error(img0, img1, depth1, px, py, z, mask)
        self.calc_motion_field_consistency_loss(T, T_inv, rot, rot_inv, self.warp_mask, px, py)

        result = 0
        for key in self.loss_vals.keys():
            if key != "sum":
                result += self.loss_vals[key]

        return result


def test():
    params = DmdsParams()
    loss = DmdsLoss(params)
    batch_size = 2

    client = MongoClient("mongodb://localhost:27017")
    collection = client["depth"]["driving_stereo"]
    documents = collection.find({}).limit(10).skip(300)
    documents = list(documents)
    for i in range(0, len(documents)-1):
        intr = np.array([
            [375.0,  0.0, 160.0],
            [ 0.0, 375.0, 128.0],
            [ 0.0,   0.0,   1.0]
        ], dtype=np.float32)
        intr = np.stack([intr]*batch_size)

        img0 = cv2.imdecode(np.frombuffer(documents[i]["img"], np.uint8), cv2.IMREAD_COLOR)
        img0, _ = resize_img(img0, 320, 128, 0)
        img0 = np.stack([img0]*batch_size, axis=0)
        img0 = img0.astype(np.float32)

        img1 = cv2.imdecode(np.frombuffer(documents[i+1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img1, _ = resize_img(img1, 320, 128, 0)
        img1 = np.stack([img1]*batch_size, axis=0)
        img1 = img1.astype(np.float32)

        # create gt depth_maps
        x0 = cv2.imdecode(np.frombuffer(documents[i]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x0, _ = resize_img(x0, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x0 = np.expand_dims(x0, axis=-1)
        x0 = np.stack([x0]*batch_size, axis=0)
        x0 = x0.astype(np.float32)
        x0 /= 255.0

        x1 = cv2.imdecode(np.frombuffer(documents[i+1]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x1, _ = resize_img(x1, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x1 = np.expand_dims(x1, axis=-1)
        x1 = np.stack([x1]*batch_size, axis=0)
        x1 = x1.astype(np.float32)
        x1 /= 255.0

        mm = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm[:, 23:85, 132:320, :] = [0.0, 0.0, 0.0]

        mm_inv = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm_inv[:, 23:85, 172:320, :] = [0.0, 0.0, 0.0]

        rot = np.zeros((batch_size, 1, 1, 3), dtype=np.float32)
        rot[:, 0, 0, :] = np.array([-0.1, 0.0, 0.1])
        rot_inv = np.zeros((batch_size, 1, 1, 3), dtype=np.float32)
        rot_inv[:, 0, 0, :] = np.array([0.1, 0.0, -0.1])

        tran = np.zeros((batch_size, 1, 1, 3), dtype=np.float32)
        tran[:, 0, 0, :] = np.array([0.0, 0.0, 10]) # [left,right | up,down | forward,backward]
        tran_inv = np.zeros((batch_size, 1, 1, 3), dtype=np.float32)
        tran_inv[:, 0, 0, :] = np.array([0.0, 0.0, -10])

        loss.calc(img1, img0, x1, x0, mm_inv, mm, tran_inv, tran, rot_inv, rot, intr, x0, x1, 0)

        for key in loss.loss_vals.keys():
            print(f"{key}: {loss.loss_vals[key].numpy()}")
        print(" - - - - - - - -")

        for i in range(len(img0)):
            f, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)
            # img0, img1
            ax11.imshow(cv2.cvtColor((img0[i]).astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax12.imshow(cv2.cvtColor((img1[i]).astype(np.uint8), cv2.COLOR_BGR2RGB))
            # x0, mm
            ax21.imshow(x1[i], cmap='gray', vmin=0, vmax=170)
            ax22.imshow((mm[i] * (255.0 / np.amax(mm[i]))).astype(np.uint8))
            # mask, frame closer mask
            ax31.imshow(loss.warp_mask[i], cmap='gray', vmin=0, vmax=1)
            ax32.imshow(cv2.cvtColor((loss.resampled_img1[i].numpy()).astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()


if __name__ == "__main__":
    test()

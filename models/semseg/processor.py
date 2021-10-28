import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.semseg.params import Params
import albumentations as A
from numba import jit
from numba.typed import List
import time


@jit(nopython=True)
def hex_to_one_hot(hex_mask, pos_mask, hex_colours):
    for i in range(hex_mask.shape[0]):
        for j in range(hex_mask.shape[1]):
            idx = 0
            if hex_mask[i][j] == 0:
                pos_mask[i][j] = 0.0
            else:
                for hex_colour in hex_colours:
                    if hex_mask[i][j] == hex_colour:
                        hex_mask[i][j] = idx
                        found_it = True
                        continue
                    idx += 1
                if not found_it:
                    assert(False and "colour does not exist: " + str(hex_colour))
    return hex_mask, pos_mask

def to_hex(img):
    """
    Convert 3 channel representation to single hex 
    channel
    """
    img = np.asarray(img, dtype='uint32')
    return (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]

class ProcessImages(IPreProcessor):
    def __init__(self, params: Params, start_augment = None):
        self._params: Params = params
        self._start_augment = start_augment

    def augment(self, img, mask, do_img_augmentation = True, do_affine_transform = True):
        if do_affine_transform:
            afine_transform = A.Compose([
                A.HorizontalFlip(p=0.4),
                A.OneOf([
                    A.GridDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ElasticTransform(interpolation=cv2.INTER_NEAREST, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, shift_limit=0.035, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                ], p=0.5),
            ], additional_targets={'mask': 'image'})
            afine_transformed = afine_transform(image=img, mask=mask)
            img = afine_transformed["image"]
            mask = afine_transformed["mask"]

        if do_img_augmentation:
            transform = A.Compose([
                A.GaussNoise(p=0.05),
                A.OneOf([
                    A.Sharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ] , p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomGamma(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomFog(p=1.0),
                    A.RandomRain(p=1.0),
                    A.RandomShadow(p=1.0),
                    A.RandomSnow(p=1.0)
                ], p=0.05),
            ])
            transformed = transform(image=img)
            img = transformed["image"]

        return img, mask

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # start_time = time.time()
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi_img = resize_img(input_data, self._params.INPUT_WIDTH, self._params.INPUT_HEIGHT, offset_bottom=self._params.OFFSET_BOTTOM)
        piped_params["roi_img"] = roi_img

        # Add ground_truth mask
        mask_encoded = np.frombuffer(raw_data["mask"], np.uint8)
        mask_img = cv2.imdecode(mask_encoded, cv2.IMREAD_COLOR)
        mask_img, _ = resize_img(mask_img, self._params.INPUT_WIDTH, self._params.INPUT_HEIGHT, offset_bottom=self._params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)

        # augment and resize mask to real size
        if "replay_img_aug" in piped_params and len(piped_params["replay_img_aug"]) > 0:
            for replay in piped_params["replay_img_aug"]:
                transformed = A.ReplayCompose.replay(replay, image=input_data, mask=mask_img)
                input_data = transformed["image"]
                mask_img = transformed["mask"]
        elif self._start_augment is not None and len(self._start_augment) == 2:
            do_img_augmentation = piped_params["epoch"] >= self._start_augment[0]
            do_affine_augmentation = piped_params["epoch"] >= self._start_augment[1]
            input_data, mask_img = self.augment(input_data, mask_img, do_img_augmentation, do_affine_augmentation)
        mask_img, _ = resize_img(mask_img, self._params.MASK_WIDTH, self._params.MASK_HEIGHT, offset_bottom=0, interpolation=cv2.INTER_NEAREST)

        # one hot encode based on class mapping from semseg spec
        mask_img = to_hex(mask_img) # convert 3 channel representation to single hex channel
        colours = List()
        for _, colour in list(SEMSEG_CLASS_MAPPING.items()):
            hex_colour = (colour[0] << 16) + (colour[1] << 8) + colour[2]
            colours.append(hex_colour)
        pos_mask = np.ones((self._params.MASK_HEIGHT, self._params.MASK_WIDTH), dtype=np.float32)
        mask_img, pos_mask = hex_to_one_hot(mask_img, pos_mask, colours)
        nb_classes = len(SEMSEG_CLASS_MAPPING)
        y_true_mask = to_categorical(mask_img, nb_classes)

        input_data = input_data.astype(np.float32)
        ground_truth = np.concatenate((y_true_mask, np.expand_dims(pos_mask, axis=-1)), axis=-1)
        # elapsed_time = time.time() - start_time
        # print(str(elapsed_time) + " s")
        return raw_data, input_data, ground_truth, piped_params

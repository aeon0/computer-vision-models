import numpy as np
import cv2
from dataclasses import dataclass
from common.processors import IPreProcessor
from common.utils import resize_img
import albumentations as A
import matplotlib.pyplot as plt


class ProcessImages(IPreProcessor):
    def __init__(self, params, do_augmentation: bool = True):
        self.params = params
        self.do_augmentation = do_augmentation

    def augment(self, img, mask, do_affine_transform = True):
        if do_affine_transform:
            afine_transform = A.Compose([
                A.HorizontalFlip(p=0.4),
                A.OneOf([
                    A.GridDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ElasticTransform(interpolation=cv2.INTER_NEAREST, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, shift_limit=0.03, rotate_limit=4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0)
                ], p=0.6),
            ], additional_targets={'mask': 'image'})
            afine_transformed = afine_transform(image=img, mask=mask)
            img = afine_transformed["image"]
            mask = afine_transformed["mask"]

        transform = A.Compose([
            A.IAAAdditiveGaussianNoise(p=0.05),
            A.OneOf([
                A.IAASharpen(p=1.0),
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
                A.RandomSnow(p=1.0),
                A.RandomSunFlare(p=1.0)
            ], p=0.05),
        ])
        transformed = transform(image=img)
        img = transformed["image"]
        return img, mask

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_t0 = cv2.imdecode(np.frombuffer(raw_data["img"], np.uint8), cv2.IMREAD_COLOR)
        img_t0, _ = resize_img(img_t0, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)

        # Add ground_truth mask
        mask_t1 = cv2.imdecode(np.frombuffer(raw_data["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        mask_t1, _ = resize_img(mask_t1, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)

        # agument
        if "replay_img_aug" in piped_params and len(piped_params["replay_img_aug"]) > 0:
            for replay in piped_params["replay_img_aug"]:
                transformed = A.ReplayCompose.replay(replay, image=img_t0, mask=mask_t1)
                img_t0 = transformed["image"]
                mask_t1 = transformed["mask"]
        elif self.do_augmentation:
            img_t0, mask_t1 = self.augment(img_t0, mask_t1)
        img_t0 = img_t0.astype(np.float32)

        mask_t1, _ = resize_img(mask_t1, self.params.MASK_WIDTH, self.params.MASK_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)

        # adjust mask values
        mask_t1 = mask_t1.astype(np.float32)
        mask_t1 /= 255.0
        pos_mask = np.where(mask_t1 > 0.0, 1.0, 0.0) 
        mask_t1 = np.clip(mask_t1, 3.1, 137.0)
        mask_t1 = 22 * np.sqrt(mask_t1 - 3.0)
        mask_t1 /= 255.0  # norm between [0, 1]

        mask_t1 *= pos_mask

        input_data = img_t0
        ground_truth = mask_t1

        return raw_data, input_data, ground_truth, piped_params

import numpy as np
import cv2
from dataclasses import dataclass
from common.processors import IPreProcessor
from common.utils import resize_img
from models.dmds.params import DmdsParams
import albumentations as A
import matplotlib.pyplot as plt


class ProcessImages(IPreProcessor):
    def __init__(self, params: DmdsParams):
        self.params: DmdsParams = params

    def augment(self, img0, img1):
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
        ], additional_targets={'img1': 'image'})
        transformed = transform(image=img0, img1=img1)
        img0 = transformed["image"]
        img1 = transformed["img1"]

        return img0, img1

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_t0 = cv2.imdecode(np.frombuffer(raw_data[0]["img"], np.uint8), cv2.IMREAD_COLOR)
        img_t0, _ = resize_img(img_t0, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)

        img_t1 = cv2.imdecode(np.frombuffer(raw_data[1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img_t1, _ = resize_img(img_t1, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)

        # img_t0, img_t1 = self.augment(img_t0, img_t1)
        img_t0 = img_t0.astype(np.float32)
        img_t1 = img_t1.astype(np.float32)

        # Add ground_truth mask
        mask_t0 = cv2.imdecode(np.frombuffer(raw_data[0]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        mask_t0, _ = resize_img(mask_t0, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)
        mask_t0 = mask_t0.astype(np.float32)
        mask_t0 /= 255.0
        mask_t0 = np.expand_dims(mask_t0, axis=-1)

        mask_t1 = cv2.imdecode(np.frombuffer(raw_data[1]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        mask_t1, _ = resize_img(mask_t1, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)
        mask_t1 = mask_t1.astype(np.float32)
        mask_t1 /= 255.0
        mask_t1 = np.expand_dims(mask_t1, axis=-1)

        # currently hardcoded for driving stereo dataset
        intr_scale = (self.params.INPUT_WIDTH / 1758.0)
        intr = np.array([
            [2063.0 * intr_scale,                 0.0, 978.0 * intr_scale],
            [                0.0, 2063.0 * intr_scale, 584.0 * intr_scale],
            [                0.0,                 0.0,                1.0]
        ], dtype=np.float32)

        input_data = [img_t0, img_t1, intr]
        ground_truth = [mask_t0, mask_t1]

        return raw_data, input_data, ground_truth, piped_params

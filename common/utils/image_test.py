import cv2
import pytest
import numpy as np
from common.utils import Logger, resize_img, cmap_depth
import matplotlib.pyplot as plt


class TestImage:
    @pytest.mark.parametrize("org_width, org_height, goal_width, goal_height, offset_bottom", [
        (1200, 1200, 480, 320, 0),
        (1200, 1200, 480, 320, -60),
        (1200, 1200, 480, 320, 60),
        (1200, 320, 480, 320, 0)
    ])
    def test_resize_img(self, org_width, org_height, goal_width, goal_height, offset_bottom):
        img = np.zeros((org_height, org_width))
        resized_img, roi = resize_img(img, goal_width, goal_height, offset_bottom)
        assert resized_img.shape[0] == goal_height
        assert resized_img.shape[1] == goal_width

    def test_cmap_depth(self):
        dummy_depth = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            dummy_depth[i][:] = i * 5.0
            rgb_depth = cmap_depth(dummy_depth, 1.0, 70.0)

        f, (ax11, ax22) = plt.subplots(1, 2)
        ax11.imshow(cv2.cvtColor(rgb_depth, cv2.COLOR_BGR2RGB))
        ax22.imshow(dummy_depth, cmap='gray', vmin=0, vmax=100)
        plt.show()

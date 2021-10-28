import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel, cmap_depth
from models.multitask import MultitaskParams, ProcessImages
from data.label_spec import SEMSEG_CLASS_MAPPING, OD_CLASS_MAPPING
from numba.typed import List


class TestProcessors:
    def setup_method(self):
        Logger.init()
        Logger.remove_file_logger()

        self.params = MultitaskParams(len(OD_CLASS_MAPPING.items()))

        # get one entry from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "labels", "nuscenes_train")

        # Create Data Generators
        self.td, self.vd = load_ids(
            self.collection_details,
            data_split=(70, 30),
            shuffle_data=True,
            limit=30
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            [self.collection_details],
            [self.td],
            batch_size=10,
            processors=[ProcessImages(self.params, [0, 0])],
            shuffle_data=True
        )

        for batch_x, batch_y in train_gen:
            print("New batch")
            for i in range(len(batch_x[0])):
                assert len(batch_x[0]) > 0
                img1 = batch_x[0][i]
                # [0: centernet_channel]: centernet
                # [centernet_channels+1]: centernet weights
                # [centernet_channels+1: semseg_classes]: semseg_masks
                # [semseg_classes+1]: semseg weights
                # [-1]: depth
                semseg_cls_items = List(SEMSEG_CLASS_MAPPING.items())
                cn_idx_end = self.params.cn_params.mask_channels()
                semseg_idx_end = (cn_idx_end+1) + len(semseg_cls_items)
                cn_heatmap = np.array(batch_y[i][:, :, :cn_idx_end])
                cn_weights = np.stack([batch_y[i][:, :, cn_idx_end]]*3, axis=-1)
                cn_heatmap = to_3channel(cn_heatmap, List([("object", (0, 0, 255))]), 0.01, True, False)
                semseg_mask = np.array(batch_y[i][:, :, cn_idx_end+1:semseg_idx_end])
                semseg_mask = to_3channel(semseg_mask, List(SEMSEG_CLASS_MAPPING.items()), 0.01, True, False)
                semseg_weights = np.stack([batch_y[i][:, :, semseg_idx_end]]*3, axis=-1)
                depth_map = np.array(batch_y[i][:, :, -1])
   
                f, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)
                ax11.imshow(cv2.cvtColor(batch_x[0][i].astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax12.imshow(cv2.cvtColor(cmap_depth(depth_map, vmin=0.1, vmax=255.0), cv2.COLOR_BGR2RGB))
                ax21.imshow(cv2.cvtColor(cn_heatmap, cv2.COLOR_BGR2RGB))
                ax22.imshow(cn_weights)
                ax31.imshow(cv2.cvtColor(semseg_mask, cv2.COLOR_BGR2RGB))
                ax32.imshow(semseg_weights)
                plt.show()

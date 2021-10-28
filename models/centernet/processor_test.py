import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel, Roi
from data.label_spec import OD_CLASS_MAPPING
from numba.typed import List
from models.centernet.processor import ProcessImages
from models.centernet.params import CenternetParams
from models.centernet.post_processing import process_2d_output


class TestProcessors:
    def setup_method(self):
        Logger.init()
        Logger.remove_file_logger()

        self.params = CenternetParams(len(OD_CLASS_MAPPING))
        self.params.REGRESSION_FIELDS["l_shape"].active = False
        self.params.REGRESSION_FIELDS["3d_info"].active = False

        # get some entries from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "labels", "nuscenes_train")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=250
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            [self.collection_details],
            [self.train_data],
            batch_size=30,
            processors=[ProcessImages(self.params, start_augmentation=[0, 0], show_debug_img=False)]
        )

        for batch_x, batch_y in train_gen:
            print("New batch")
            for i in range(len(batch_x[0])):
                assert len(batch_x[0]) > 0
                img1 = batch_x[0][i]
                heatmap = np.array(batch_y[i][:, :, :1]) # needed because otherwise numba makes mimimi
                heatmap = to_3channel(heatmap, List([("object", (0, 0, 255))]), 0.01, True, False)
                weights = np.stack([batch_y[i][:, :, -1]]*3, axis=-1)

                roi = Roi()
                objects = process_2d_output(batch_y[i], roi, self.params, 0.2)
                for obj in objects:
                    color = list(OD_CLASS_MAPPING.values())[obj["cls_idx"]]
                    top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
                    bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
                    cv2.rectangle(img1, top_left, bottom_right, color, 1)
                    cv2.circle(img1, (int(obj["center"][0]), int(obj["center"][1])), 2, color, 1)

                f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(cv2.cvtColor(batch_x[0][i].astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax2.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                ax3.imshow(weights)
                plt.show()

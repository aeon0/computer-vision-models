import cv2
import numpy as np
from collections import OrderedDict
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from data.label_spec import OD_CLASS_MAPPING
from models.centertracker.processor import CenterTrackerProcess
from models.centertracker.params import CentertrackerParams


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = CentertrackerParams(len(OD_CLASS_MAPPING))

        # get some entries from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "labels", "kitti")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=100
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=12,
            processors=[CenterTrackerProcess(self.params)]
        )

        batch_x, batch_y = train_gen[0]

        batch_size = len(batch_x["img"])
        for i in range(batch_size):
            # input_data is a list in this case and consist of:
            # [0]: img current frame
            # [1]: img prev frame
            # [3]: single channel heatmap
            curr_img = batch_x["img"][i]
            prev_img = batch_x["prev_img"][i]

            mask_img = to_3channel(batch_y[i], OD_CLASS_MAPPING, 0.1, True)
            prev_mask_img = to_3channel(batch_x["prev_heatmap"][i], OrderedDict([("obj", (255, 255, 255))]), 0.1, True)
            cv2.imshow("img", curr_img.astype(np.uint8))
            cv2.imshow("prev_img", prev_img.astype(np.uint8))
            cv2.imshow("prev_heatmap", prev_mask_img)
            cv2.imshow("mask", mask_img)
            cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from models.semseg.params import Params
from models.semseg.processor import ProcessImages
from data.label_spec import SEMSEG_CLASS_MAPPING
from numba.typed import List


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = Params()

        # get one entry from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "labels", "comma10k")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=30
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            [self.collection_details],
            [self.train_data],
            batch_size=10,
            processors=[ProcessImages(self.params)]
        )

        batch_x, batch_y = train_gen[0]

        for i, input_data in enumerate(batch_x):
            assert len(input_data) > 0
            cls_items = List(SEMSEG_CLASS_MAPPING.items())
            nb_classes = len(cls_items)
            semseg_mask = np.array(batch_y[i][:, :, :-1]) # needed because otherwise numba makes mimimi
            mask_img = to_3channel(semseg_mask, cls_items, threshold=0.999)
            pos_mask = batch_y[i][:, :, -1]

            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
            ax3.imshow(pos_mask, cmap="gray", vmin=0.0, vmax=1.0)
            plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, cmap_depth
from models.depth.params import Params
from models.depth.processor import ProcessImages


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
        collection_details = ("local_mongodb", "labels", "nuscenes_train")

        # get ids
        td, vd = load_ids(
            collection_details,
            data_split=(70, 30),
            limit=100,
            shuffle_data=True,
        )
        self.train_data = [td]
        self.val_data = [vd]
        self.collection_details = [collection_details]

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=8,
            processors=[ProcessImages(self.params)],
            shuffle_data=True
        )

        for batch_x, batch_y in train_gen:
            for i in range(len(batch_x)):
                assert len(batch_x[0]) > 0
                img_t0 = batch_x[i]
                mask_t0 = batch_y[i]

                f, (ax11, ax22) = plt.subplots(2, 1)
                ax11.imshow(cv2.cvtColor(img_t0.astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax22.imshow(cmap_depth(mask_t0, vmin=0.1, vmax=255.0))
                plt.show()

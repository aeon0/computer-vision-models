import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger
from models.dmds.params import DmdsParams
from models.dmds.processor import ProcessImages


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = DmdsParams()

        # get one entry from the database
        Config.add_config('./config.ini')
        collection_details = ("local_mongodb", "depth", "driving_stereo")
        scenes = [
            "2018-10-26-15-24-18",
            "2018-10-19-09-30-39",
        ]
        self.train_data = []
        self.val_data = []
        self.collection_details = []

        # get ids
        for scene_token in scenes:
            td, vd = load_ids(
                collection_details,
                data_split=(80, 20),
                limit=100,
                shuffle_data=False,
                mongodb_filter={"scene_token": scene_token},
                sort_by={"timestamp": 1}
            )
            self.train_data.append(td)
            self.val_data.append(vd)
            self.collection_details.append(collection_details)

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=8,
            processors=[ProcessImages(self.params)],
            data_group_size=2,
            continues_data_selection=True,
            shuffle_data=False
        )

        for batch_x, batch_y in train_gen:
            print("New BATCH!")
            for i in range(len(batch_x[0])):
                assert len(batch_x[0]) > 0
                img_t0 = batch_x[0][i]
                img_t1 = batch_x[1][i]

                mask_t0 = batch_y[0][i]
                mask_t1 = batch_y[1][i]

                f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
                ax11.imshow(cv2.cvtColor(img_t0.astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax12.imshow(cv2.cvtColor(img_t1.astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax21.imshow(mask_t0, cmap='gray', vmin=0, vmax=170)
                ax22.imshow(mask_t1, cmap='gray', vmin=0, vmax=170)
                #plt.show()
                plt.draw()
                plt.waitforbuttonpress(0) # this will wait for indefinite time
                plt.close(f)

import tensorflow as tf
from typing import List


class BaseDataGenerator(tf.keras.utils.Sequence):
    """
    Base Class for Data Generators which implements data processors, but data generators can also be created
    without the use of this Base Class
    More information on the usage of Sequence: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    def __init__(self,
                 batch_size: int = 32,
                 processors: List[any] = list()):
        """
        :param batch_size: size of the batch as integer, default = 32
        :param processors: a list of processors which implement the IProcessor class
        """
        self.batch_size = batch_size
        self.processors = processors
        self.epoch_nb = 0 # will be increased in the generators

    def add_processors(self, processors: list):
        """
        Add a processor to the processor list
        :param processors: Data processor instance
        """
        self.processors.extend(processors)

    def _process_batch(self, batch):
        """
        Process the raw data from the data reader with the specified processors
        :param batch: raw data for this batch (must be iterable)
        :return: array of 2: [batch data input, batch data ground truth]
        """
        batch_x = []
        batch_y = []
        for data in batch:
            input_data = None
            ground_truth = None
            piped_params = { "epoch": self.epoch_nb }
            raw_data = data
            # process each entry in the batch list one by one
            for processor in self.processors:
                raw_data, input_data, ground_truth, piped_params = processor.process(raw_data, input_data, ground_truth,
                                                                                     piped_params=piped_params)
            batch_x.append(input_data)
            batch_y.append(ground_truth)
        return batch_x, batch_y

from abc import ABCMeta, abstractmethod


class IPreProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        """
        One at a time data processor interface
        :param raw_data: the raw_data fetched from a data source
        :param input_data: the input_data which will be used by the model
        :param ground_truth: the ground_truth / labels which will be used by the model
        :param piped_params: any additional parameters that need to be piped from processor to processor
        :return: same as input params as these will be the input of the next processor in the the pipline
        """
        ...
        return raw_data, input_data, ground_truth, piped_params
